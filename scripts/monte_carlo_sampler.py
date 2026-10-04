#!/usr/bin/env python3
"""
蒙特卡洛证据采样器

实现 Sirchmunk 的三阶段检索算法：
  Phase 1: EXPLORATION - 宽泛的随机化探测
  Phase 2: EXPLOITATION - 聚焦高分区域深读
  Phase 3: SYNTHESIS - 聚合、去重、持久化

支持 FAST/DEEP 两种模式：
  FAST: 检查集群缓存 -> 单轮 grep -> 返回（<5秒）
  DEEP: 完整三阶段蒙特卡洛（10-30秒）

使用方式：
    from monte_carlo_sampler import MonteCarloSampler
    from knowledge_clusters import KnowledgeClusterStore

    store = KnowledgeClusterStore(Path("~/.cache/clusters"))
    sampler = MonteCarloSampler(Path("~/legal_docs"), store)

    # 快速模式
    result = sampler.sample("善意取得", mode='fast')

    # 深度模式
    result = sampler.sample("善意取得", mode='deep')
"""

import math
import random
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent))

from knowledge_clusters import KnowledgeClusterStore

from shared_utils import load_synonym_expansion

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


@dataclass
class Evidence:
    """单条证据"""
    text: str = ""
    file: str = ""
    line_start: int = 0
    line_end: int = 0
    relevance: float = 0.0
    section_context: str = ""
    sub_topic: Optional[str] = None


@dataclass
class SamplingResult:
    """采样结果"""
    query: str = ""
    mode: str = "deep"
    evidence: List[Evidence] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    cluster_id: Optional[str] = None
    was_cached: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)


class MonteCarloSampler:
    """
    蒙特卡洛证据采样器

    Args:
        library_path: 文献库根目录
        cluster_store: 知识集群存储
        synonym_path: 同义词扩展 YAML 路径
    """

    def __init__(
        self,
        library_path: Path,
        cluster_store: KnowledgeClusterStore,
        synonym_path: str = None,
    ):
        self._library_path = Path(library_path)
        self._cluster_store = cluster_store
        self._synonym_map = load_synonym_expansion(synonym_path)

    def sample(self, query: str, mode: str = 'deep', book_type: str = None) -> SamplingResult:
        """
        主入口：对给定查询执行证据采样。

        Args:
            query: 查询字符串
            mode: 'fast' 或 'deep'
            book_type: 书籍类型提示（可选）

        Returns:
            SamplingResult
        """
        if mode == 'fast':
            return self._fast_sample(query, book_type)
        else:
            return self._deep_sample(query, book_type)

    # ─── FAST 模式 ───

    def _fast_sample(self, query: str, book_type: str = None) -> SamplingResult:
        """
        快速模式：先查集群缓存，未命中则单轮 grep。
        目标：<5 秒，最多 2 次工具调用。
        """
        # Step 1: 查集群缓存
        cached = self._cluster_store.find_similar(query, threshold=0.85)
        if cached:
            return SamplingResult(
                query=query,
                mode='fast',
                evidence=[Evidence(text=r.get('text', r.get('snippet', '')),
                                  file=r.get('file', ''),
                                  relevance=r.get('relevance', 0.5))
                         for r in cached.results],
                sources=cached.source_files,
                confidence=cached.confidence,
                cluster_id=cached.cluster_id,
                was_cached=True,
                stats={"cache_hit": True, "original_query": cached.query},
            )

        # Step 2: 单轮搜索
        keywords = self._generate_search_terms(query)
        md_files = self._list_md_files()

        if not md_files:
            return SamplingResult(query=query, mode='fast', confidence=0.0)

        hits = []
        for md_file in md_files:
            file_hits = self._grep_file(md_file, keywords, context_lines=5)
            hits.extend(file_hits)

        # 按相关性排序，取前 5
        hits.sort(key=lambda h: h['relevance'], reverse=True)
        top_hits = hits[:5]

        evidence = [
            Evidence(
                text=h['text'],
                file=h['file'],
                line_start=h.get('line', 0),
                line_end=h.get('line', 0) + 10,
                relevance=h['relevance'],
                section_context=h.get('section', ''),
            )
            for h in top_hits
        ]

        sources = list(set(h['file'] for h in top_hits))
        confidence = min(0.8, len(top_hits) * 0.15)

        # 持久化到集群
        cluster = self._cluster_store.add_cluster(
            query=query,
            results=[{'text': e.text, 'file': e.file, 'relevance': e.relevance} for e in evidence],
            source_files=sources,
            confidence=confidence,
        )

        return SamplingResult(
            query=query,
            mode='fast',
            evidence=evidence,
            sources=sources,
            confidence=confidence,
            cluster_id=cluster.cluster_id,
            was_cached=False,
            stats={"total_hits": len(hits), "files_searched": len(md_files)},
        )

    # ─── DEEP 模式 ───

    def _deep_sample(self, query: str, book_type: str = None) -> SamplingResult:
        """
        深度模式：完整三阶段蒙特卡洛采样。
        目标：10-30 秒，5-10 次工具调用。
        """
        stats = {}

        # Phase 1: EXPLORATION
        exploration_results = self._phase_exploration(query, book_type)
        stats['exploration_hits'] = len(exploration_results['hits'])
        stats['exploration_files'] = len(exploration_results['file_scores'])

        # Phase 2: EXPLOITATION
        exploitation_results = self._phase_exploitation(
            query, exploration_results
        )
        stats['exploitation_reads'] = exploitation_results['reads']
        stats['evidence_count'] = len(exploitation_results['evidence'])

        # Phase 3: SYNTHESIS
        final = self._phase_synthesis(query, exploitation_results)
        stats['final_evidence'] = len(final['evidence'])
        stats['deduplication_removed'] = exploitation_results['reads'] - len(final['evidence'])

        evidence = final['evidence']
        sources = final['sources']
        confidence = final['confidence']

        # 检查是否有已有集群可合并
        existing = self._cluster_store.find_similar(query, threshold=0.90)
        if existing:
            self._cluster_store.update_cluster(
                existing.cluster_id,
                [{'text': e.text, 'file': e.file, 'relevance': e.relevance} for e in evidence]
            )
            cluster_id = existing.cluster_id
        else:
            cluster = self._cluster_store.add_cluster(
                query=query,
                results=[{'text': e.text, 'file': e.file, 'relevance': e.relevance} for e in evidence],
                source_files=sources,
                confidence=confidence,
            )
            cluster_id = cluster.cluster_id

        return SamplingResult(
            query=query,
            mode='deep',
            evidence=evidence,
            sources=sources,
            confidence=confidence,
            cluster_id=cluster_id,
            was_cached=False,
            stats=stats,
        )

    # ─── Phase 1: EXPLORATION ───

    def _phase_exploration(self, query: str, book_type: str = None) -> Dict[str, Any]:
        """
        探索阶段：宽泛的随机化探测。

        生成多组搜索词，随机采样文件子集，记录所有命中。
        """
        search_terms = self._generate_search_terms(query)
        md_files = self._list_md_files()

        # 随机采样文件（>10 个时采样 60%）
        if len(md_files) > 10:
            sample_count = max(6, int(len(md_files) * 0.6))
            sampled_files = random.sample(md_files, sample_count)
        else:
            sampled_files = md_files

        all_hits = []
        file_scores: Dict[str, float] = {}

        for md_file in sampled_files:
            file_hits = self._grep_file(md_file, search_terms, context_lines=5)
            if file_hits:
                # 累计文件得分
                file_score = sum(h['relevance'] for h in file_hits)
                file_scores[md_file.name] = file_score
                all_hits.extend(file_hits)

        return {
            'hits': all_hits,
            'file_scores': file_scores,
            'search_terms': search_terms,
        }

    # ─── Phase 2: EXPLOITATION ───

    def _phase_exploitation(
        self, query: str, exploration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        利用阶段：聚焦高分区域进行深度阅读。
        """
        file_scores = exploration['file_scores']
        if not file_scores:
            return {'evidence': [], 'reads': 0}

        # 取得分最高的 30% 文件
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, int(len(sorted_files) * 0.3))
        top_files = [name for name, _ in sorted_files[:top_count]]

        evidence = []
        total_reads = 0

        # 收集已有证据的文本用于新颖性检查
        existing_texts: List[str] = []

        for filename in top_files:
            file_path = self._library_path / filename
            if not file_path.exists():
                # 递归查找
                found = list(self._library_path.rglob(filename))
                if found:
                    file_path = found[0]
                else:
                    continue

            try:
                content = file_path.read_text(encoding='utf-8')
            except Exception:
                continue

            lines = content.split('\n')

            # 找到该文件中的高分区域（来自探索阶段的命中）
            file_hits = [h for h in exploration['hits'] if h['file'] == filename]
            if not file_hits:
                continue

            # 按行号聚类命中点
            regions = self._cluster_hit_regions(file_hits, window=50)

            for region_start, region_end in regions[:3]:  # 每文件最多 3 个区域
                # 扩展阅读范围
                read_start = max(0, region_start - 20)
                read_end = min(len(lines), region_end + 100)
                passage = '\n'.join(lines[read_start:read_end])
                total_reads += 1

                # 提取段落
                paragraphs = self._extract_paragraphs(passage, query)

                for para_text, section in paragraphs:
                    # 新颖性检查
                    is_novel = all(
                        SequenceMatcher(None, para_text, et).ratio() < 0.5
                        for et in existing_texts
                    ) if existing_texts else True

                    if is_novel and len(para_text) > 30:
                        relevance = self._score_relevance(para_text, query)
                        evidence.append(Evidence(
                            text=para_text,
                            file=filename,
                            line_start=read_start,
                            line_end=read_end,
                            relevance=relevance,
                            section_context=section,
                        ))
                        existing_texts.append(para_text)

                # 自适应停止
                if len(evidence) >= 15:
                    break

            if len(evidence) >= 15:
                break

        return {
            'evidence': evidence,
            'reads': total_reads,
        }

    # ─── Phase 3: SYNTHESIS ───

    def _phase_synthesis(
        self, query: str, exploitation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        综合阶段：去重、分组、置信度评估。
        """
        evidence = exploitation['evidence']

        # 去重（相似度 > 0.8 的保留更长的）
        deduped = []
        for e in evidence:
            is_dup = False
            for existing in deduped:
                if SequenceMatcher(None, e.text[:200], existing.text[:200]).ratio() > 0.8:
                    if len(e.text) > len(existing.text):
                        deduped.remove(existing)
                        deduped.append(e)
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(e)

        # 按相关性排序
        deduped.sort(key=lambda e: e.relevance, reverse=True)

        # 源文件去重
        sources = list(set(e.file for e in deduped))

        # 置信度评估
        if not deduped:
            confidence = 0.0
        else:
            evidence_factor = min(1.0, len(deduped) / 5.0)
            diversity_factor = min(1.0, len(sources) / 3.0)
            quality_factor = sum(e.relevance for e in deduped[:5]) / min(5, len(deduped))
            confidence = (evidence_factor * 0.3 + diversity_factor * 0.3 + quality_factor * 0.4)

        return {
            'evidence': deduped[:10],  # 最多 10 条
            'sources': sources,
            'confidence': round(confidence, 2),
        }

    # ─── 工具方法 ───

    def _generate_search_terms(self, query: str) -> List[str]:
        """从查询生成搜索词（分词 + 同义词扩展）"""
        if HAS_JIEBA:
            words = [w for w in jieba.cut(query) if len(w) >= 2]
        else:
            words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', query)

        # 同义词扩展
        expanded = list(words)
        seen = set(words)
        for w in words:
            for syn in self._synonym_map.get(w, []):
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

        return expanded

    def _list_md_files(self) -> List[Path]:
        """列出文献库中的所有 MD 文件"""
        return sorted(self._library_path.rglob("*.md"))

    def _grep_file(
        self, file_path: Path, keywords: List[str], context_lines: int = 5
    ) -> List[Dict[str, Any]]:
        """在单个文件中搜索关键词"""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return []

        lines = content.split('\n')
        hits = []

        for kw in keywords:
            for i, line in enumerate(lines):
                if kw in line:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = '\n'.join(lines[start:end])

                    # 计算该命中点周围的关键词密度
                    window = '\n'.join(lines[max(0, i-20):min(len(lines), i+20)])
                    density = sum(1 for k in keywords if k in window) / len(keywords)

                    # 标题匹配加分
                    is_heading = line.strip().startswith('#')
                    position_weight = 1.5 if is_heading else 1.0

                    relevance = density * position_weight

                    # 获取段落标题上下文
                    section = self._find_nearest_heading(lines, i)

                    hits.append({
                        'file': file_path.name,
                        'line': i,
                        'text': context,
                        'relevance': relevance,
                        'section': section,
                        'keyword': kw,
                    })

        return hits

    @staticmethod
    def _find_nearest_heading(lines: List[str], line_idx: int) -> str:
        """向上查找最近的标题"""
        for i in range(line_idx, max(-1, line_idx - 50), -1):
            if lines[i].strip().startswith('#'):
                return lines[i].strip().lstrip('#').strip()
        return ""

    @staticmethod
    def _cluster_hit_regions(
        hits: List[Dict], window: int = 50
    ) -> List[tuple]:
        """将命中点按位置聚类为区域"""
        if not hits:
            return []

        lines = sorted(set(h.get('line', 0) for h in hits))
        regions = []
        region_start = lines[0]
        region_end = lines[0]

        for line in lines[1:]:
            if line - region_end <= window:
                region_end = line
            else:
                regions.append((region_start, region_end))
                region_start = line
                region_end = line

        regions.append((region_start, region_end))
        return regions

    def _extract_paragraphs(
        self, passage: str, query: str
    ) -> List[tuple]:
        """从段落中提取包含查询词的完整段落"""
        paragraphs = re.split(r'\n\s*\n', passage)
        results = []
        keywords = self._generate_search_terms(query)[:5]

        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue
            if any(kw in para for kw in keywords):
                # 查找段落所属的标题
                lines_before = passage[:passage.find(para)].split('\n')
                section = ""
                for line in reversed(lines_before):
                    if line.strip().startswith('#'):
                        section = line.strip().lstrip('#').strip()
                        break
                results.append((para, section))

        return results

    def _score_relevance(self, text: str, query: str) -> float:
        """评估文本与查询的相关性"""
        keywords = self._generate_search_terms(query)[:10]
        if not keywords:
            return 0.0

        # 关键词命中率
        hit_count = sum(1 for kw in keywords if kw in text)
        hit_rate = hit_count / len(keywords)

        # 内容质量（法律术语密度、长度）
        legal_markers = ['构成要件', '法律效果', '举证责任', '因果关系', '过错',
                        '根据', '规定', '条', '款', '认为', '主张']
        quality = sum(1 for m in legal_markers if m in text) / len(legal_markers)

        # 长度因子（适中长度最好）
        length_factor = min(1.0, len(text) / 200.0) if len(text) < 1000 else 0.8

        return round(hit_rate * 0.5 + quality * 0.3 + length_factor * 0.2, 3)


def _cli():
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="蒙特卡洛证据采样器 — 三阶段检索算法（Sirchmunk 风格）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速检索
  python monte_carlo_sampler.py "善意取得构成要件" --library ~/legal_docs --mode fast

  # 深度检索（三阶段蒙特卡洛）
  python monte_carlo_sampler.py "违约金调整" --library ~/legal_docs --mode deep

  # 指定书籍类型和随机种子（复现结果）
  python monte_carlo_sampler.py "合同解除" --library ~/legal_docs \
      --book-type monograph --seed 42

  # 运行自带冒烟测试
  python monte_carlo_sampler.py --test
""",
    )
    ap.add_argument("query", nargs="?", help="查询字符串")
    ap.add_argument("--library", "-L", help="文献库根目录（含 .md 文件）")
    ap.add_argument("--mode", choices=["fast", "deep"], default="deep", help="检索模式（默认 deep）")
    ap.add_argument("--book-type", dest="book_type", help="书籍类型提示，如 monograph/case_compilation")
    ap.add_argument("--cluster-store", dest="cluster_store",
                    default="~/.cache/legal-kb-builder/clusters",
                    help="知识集群缓存目录")
    ap.add_argument("--synonym-path", dest="synonym_path", help="同义词扩展 YAML 路径")
    ap.add_argument("--seed", type=int, help="随机种子，用于复现 Phase-1 采样")
    ap.add_argument("--top-k", type=int, default=10, help="返回证据条数")
    ap.add_argument("--test", action="store_true", help="运行内置冒烟测试")
    args = ap.parse_args()

    if args.test:
        _run_smoke_test()
        return 0

    if not args.query or not args.library:
        ap.error("必须提供 <query> 和 --library；或使用 --test 运行冒烟测试")

    if args.seed is not None:
        random.seed(args.seed)

    library = Path(args.library).expanduser().resolve()
    if not library.exists():
        print(f"错误：文献库目录不存在: {library}", file=sys.stderr)
        return 1

    cluster_dir = Path(args.cluster_store).expanduser()
    store = KnowledgeClusterStore(cluster_dir)
    sampler = MonteCarloSampler(library, store, synonym_path=args.synonym_path)
    result = sampler.sample(args.query, mode=args.mode, book_type=args.book_type)

    out = {
        "query": args.query,
        "mode": args.mode,
        "cached": result.was_cached,
        "confidence": result.confidence,
        "stats": result.stats,
        "evidence": [
            {
                "source": ev.file,
                "score": ev.relevance,
                "text": ev.text[:500],
            }
            for ev in result.evidence[: args.top_k]
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _run_smoke_test():
    import json
    import tempfile
    import shutil

    tmp = Path(tempfile.mkdtemp())
    cluster_dir = tmp / "clusters"
    lib_dir = tmp / "library"
    lib_dir.mkdir()

    (lib_dir / "test1.md").write_text(
        "# 善意取得\n\n善意取得是指无处分权人将不动产或者动产转让给受让人的，"
        "所有权人有权追回；除法律另有规定外，符合下列情形的，受让人取得该不动产或者动产的所有权：\n\n"
        "## 构成要件\n\n一、受让人受让该不动产或者动产时是善意的。\n"
        "二、以合理的价格转让。\n三、转让的不动产或者动产依照法律规定应当登记的已经登记。\n"
    )

    store = KnowledgeClusterStore(cluster_dir)
    sampler = MonteCarloSampler(lib_dir, store)

    result = sampler.sample("善意取得构成要件", mode='fast')
    print(f"FAST: {len(result.evidence)} evidence, conf={result.confidence}, cached={result.was_cached}")

    result2 = sampler.sample("善意取得的构成要件", mode='fast')
    print(f"FAST (2nd): cached={result2.was_cached}")

    result3 = sampler.sample("善意取得的法律效果", mode='deep')
    print(f"DEEP: {len(result3.evidence)} evidence, conf={result3.confidence}, stats={result3.stats}")

    shutil.rmtree(tmp)
    print("All tests completed.")


if __name__ == "__main__":
    sys.exit(_cli())
