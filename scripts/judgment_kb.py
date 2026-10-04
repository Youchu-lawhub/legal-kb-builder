#!/usr/bin/env python3
"""
裁判文书知识库主入口

提供 init/add/search/stats 命令，编排整个裁判文书知识库的生命周期。
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from shared_utils import file_lock, atomic_write_json, build_record_index_bundle, active_record_index

from judgment_parser import JudgmentParser, JudgmentElements


def normalize_case_number(value: str) -> str:
    """将案号用于比较的表示统一为半角括号且无空白，保留原始显示值。"""
    return "".join((value or "").translate(str.maketrans({"（": "(", "）": ")"})).split())


class JudgmentKnowledgeBase:
    """裁判文书知识库"""

    def __init__(self, kb_path: str = None):
        self.kb_path = Path(kb_path) if kb_path else Path.cwd() / "judgment_kb"
        self.parser = JudgmentParser()

    def init(self, name: str) -> None:
        """初始化知识库目录结构"""
        dirs = ["judgments", "indices", "cache"]
        for d in dirs:
            (self.kb_path / d).mkdir(parents=True, exist_ok=True)

        config = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "judgment_count": 0,
        }
        config_path = self.kb_path / "config.yaml"
        if not config_path.exists():
            import yaml
            from shared_utils import atomic_write_text
            atomic_write_text(config_path, yaml.safe_dump(config, allow_unicode=True))

        # 初始化元数据索引
        metadata = {"judgments": [], "stats": {}}
        metadata_path = self.kb_path / "indices" / "metadata.json"
        if not metadata_path.exists():
            atomic_write_json(metadata_path, metadata)

        print(f"知识库 '{name}' 初始化成功: {self.kb_path}")

    def add(self, file_path: str, rebuild: bool = True) -> Optional[JudgmentElements]:
        """添加单个裁判文书"""
        path = Path(file_path)
        if not path.exists():
            print(f"文件不存在: {file_path}")
            return None

        # 读取文本
        if path.suffix.lower() == '.pdf':
            print(f"PDF 文件需先用 MinerU 转换为文本: {path}")
            return None

        text = path.read_text(encoding='utf-8')

        # 解析
        print(f"解析: {path.name}")
        elements = self.parser.parse(text)

        if elements.parse_confidence < 0.3:
            print(f"  警告: 解析置信度过低 ({elements.parse_confidence})，可能不是裁判文书")

        # 生成安全文件名
        safe_name = elements.case_number or path.stem
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-（）() ").strip()
        if not safe_name:
            safe_name = path.stem

        # 保存解析结果
        output_path = self.kb_path / "judgments" / f"{safe_name}.json"
        with file_lock(self.kb_path / ".write.lock"):
            atomic_write_json(output_path, elements.to_dict())
            self._update_metadata(elements, output_path.name)
        if rebuild:
            self.rebuild_indices()

        print(f"  案号: {elements.case_number}")
        print(f"  法院: {elements.court}")
        print(f"  案由: {elements.cause_of_action}")
        print(f"  置信度: {elements.parse_confidence}")
        print(f"  已保存: {output_path}")

        return elements

    def add_batch(self, directory: str) -> None:
        """批量添加目录下的所有文书"""
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"目录不存在: {directory}")
            return

        files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))
        print(f"找到 {len(files)} 个文件")

        for f in files:
            self.add(str(f), rebuild=False)
            print()

        # 批量完成后重建索引
        self.rebuild_indices()

    def rebuild_indices(self) -> None:
        """Build BM25 and optional vectors in one immutable, manifest-published generation."""
        with file_lock(self.kb_path / ".write.lock", timeout=3600):
            metadata_path = self.kb_path / "indices" / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"裁判元数据不存在: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fields_yaml = Path(__file__).parent.parent / "assets" / "judgment-fields.yaml"
            weights = {"court_reasoning": 2, "dispute_focus": 1, "found_facts": 1,
                       "judgment_result": 1, "cause_of_action": 1}
            try:
                import yaml
                cfg = yaml.safe_load(fields_yaml.read_text(encoding="utf-8")) or {}
                weights = {entry["field"]: entry.get("weight", 1.0)
                           for entry in cfg.get("embedding", {}).get("embeddable_fields", [])}
            except (ImportError, OSError, ValueError):
                pass
            records = []
            for item in metadata.get("judgments", []):
                path = self.kb_path / "judgments" / item["file"]
                if not path.is_file():
                    raise FileNotFoundError(f"裁判文书缺失: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                fields = ["cause_of_action", "found_facts", "court_reasoning", "judgment_result", "dispute_focus"]
                searchable = "\n".join(str(data.get(field) or "") for field in fields)
                weighted = []
                for field, weight in weights.items():
                    value = str(data.get(field) or "")
                    weighted.extend([value] * max(1, int(weight)) if value else [])
                records.append((Path(item["file"]).stem, searchable + "\n" + "\n".join(weighted)))
            build_record_index_bundle(self.kb_path, {"judgments": records}, metadata)
            print(f"检索索引版本已发布 ({len(records)} 份)")

    def search(self, query: str = None, case_number: str = None,
               cause: str = None, court: str = None,
               party: str = None, law: str = None,
               mode: str = 'hybrid', top_k: int = 10) -> List[Dict]:
        """检索裁判文书。

        v2 修复：
        - hybrid/semantic/bm25 模式真正调用 BM25Searcher + EmbeddingManager，
          再用 RRF 融合结构化检索结果。
        - 若索引不存在，自动降级到 grep 兜底。
        """
        metadata_path = self.kb_path / "indices" / "metadata.json"
        if not metadata_path.exists():
            return []

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        judgments = metadata.get('judgments', [])
        by_file = {jm.get('file', ''): jm for jm in judgments}
        # 也建立 stem -> file 的映射（索引里存的是 stem）
        by_stem = {Path(jm.get('file', '')).stem: jm for jm in judgments if jm.get('file')}

        # ── 1. 结构化精确检索 ──
        structured_ranks: Dict[str, float] = {}
        for jm in judgments:
            score = 0.0
            stored_case = jm.get('case_number_normalized') or normalize_case_number(
                jm.get('case_number') or ''
            )
            if case_number and normalize_case_number(case_number) in stored_case:
                score += 10.0
            if cause and cause in (jm.get('cause_of_action') or ''):
                score += 5.0
            if court and court in (jm.get('court') or ''):
                score += 3.0
            if party:
                for p in jm.get('parties', []):
                    if party in (p.get('name') or ''):
                        score += 3.0
            if law:
                for l in jm.get('applied_laws', []):
                    if law in l:
                        score += 2.0
            if score > 0:
                structured_ranks[jm.get('file', '')] = score

        # 无自然语言查询：只按结构化得分返回
        if not query:
            ordered = sorted(structured_ranks.items(), key=lambda x: x[1], reverse=True)
            return [dict(by_file[f], score=s) for f, s in ordered[:top_k] if f in by_file]

        # ── 2. BM25 + Vector 索引检索 ──
        bm25_hits: Dict[str, float] = {}
        vec_hits: Dict[str, float] = {}
        from shared_utils import active_record_index
        index_generation = active_record_index(self.kb_path, metadata)
        indices_dir = ((index_generation / "judgments_index") if index_generation else
                       (self.kb_path / "indices"))

        def _chunk_id_to_stem(chunk_id: str) -> str:
            """BM25/向量索引里 chunk_id 常见形态：
               - 'abc.md#0'
               - 'abc_chunk_0'
               - 'abc.md'
            这里统一提取原始文件 stem（不含 .md 扩展）。"""
            s = chunk_id.split('#', 1)[0]
            s = Path(s).stem
            for sep in ('_chunk_', '_chunk-', '_chunk'):
                if sep in s:
                    s = s.split(sep, 1)[0]
                    break
            return s

        if mode in ('hybrid', 'bm25') and (indices_dir / "_bm25_corpus.json").exists():
            try:
                from bm25_searcher import BM25Searcher
                bm25 = BM25Searcher()
                for chunk_id, score in bm25.search(indices_dir, query, top_k=top_k * 3):
                    stem = _chunk_id_to_stem(chunk_id)
                    jm = by_stem.get(stem)
                    if jm:
                        # 同一文书多 chunk 命中取最高分
                        f = jm.get('file', '')
                        bm25_hits[f] = max(bm25_hits.get(f, 0.0), float(score))
            except ImportError:
                pass
            except Exception as e:
                print(f"  BM25 检索失败: {e}", file=sys.stderr)

        from shared_utils import vector_index_exists
        if mode in ('hybrid', 'semantic') and vector_index_exists(indices_dir):
            try:
                from embedding_manager import EmbeddingManager
                em = EmbeddingManager()
                for chunk_id, score in em.search(indices_dir, query, top_k=top_k * 3):
                    stem = _chunk_id_to_stem(chunk_id)
                    jm = by_stem.get(stem)
                    if jm:
                        f = jm.get('file', '')
                        vec_hits[f] = max(vec_hits.get(f, 0.0), float(score))
            except ImportError:
                pass
            except Exception as e:
                print(f"  向量检索失败: {e}", file=sys.stderr)

        # ── 3. RRF 融合 ──
        def _to_rank(hits: Dict[str, float]) -> Dict[str, int]:
            ordered = sorted(hits.items(), key=lambda x: x[1], reverse=True)
            return {f: i + 1 for i, (f, _) in enumerate(ordered)}

        rrf_k = 60
        fused: Dict[str, float] = {}
        for hits in (bm25_hits, vec_hits, structured_ranks):
            for f, rank in _to_rank(hits).items():
                fused[f] = fused.get(f, 0.0) + 1.0 / (rrf_k + rank)

        # ── 4. 兜底：所有索引都没结果时按关键词子集召回 ──
        if not fused:
            try:
                import jieba
                terms = [term.strip() for term in jieba.lcut(query) if len(term.strip()) >= 2]
            except ImportError:
                terms = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}', query)
            terms = list(dict.fromkeys(terms)) or [query]
            for jm in judgments:
                jf = self.kb_path / "judgments" / jm.get('file', '')
                if not jf.exists():
                    continue
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                searchable = "\n".join([
                    data.get('cause_of_action', '') or '',
                    data.get('found_facts', '') or '',
                    data.get('court_reasoning', '') or '',
                    data.get('judgment_result', '') or '',
                ])
                matched = sum(1 for term in terms if term in searchable)
                if matched:
                    fused[jm.get('file', '')] = matched / len(terms)

        # ── 5. 组装最终结果 ──
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results: List[Dict] = []
        for fname, score in ordered[:top_k]:
            if fname not in by_file:
                continue
            entry = dict(by_file[fname], score=round(score, 4))
            # 附加片段
            jf = self.kb_path / "judgments" / fname
            if jf.exists():
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    reasoning = data.get('court_reasoning', '') or ''
                    entry['snippet'] = reasoning[:200] if reasoning else (data.get('found_facts', '') or '')[:200]
                except Exception:
                    pass
            results.append(entry)
        return results

    def stats(self) -> Dict[str, Any]:
        """知识库统计"""
        metadata_path = self.kb_path / "indices" / "metadata.json"
        if not metadata_path.exists():
            return {"error": "知识库未初始化"}

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        judgments = metadata.get('judgments', [])
        courts = set(j.get('court', '') for j in judgments if j.get('court'))
        causes = set(j.get('cause_of_action', '') for j in judgments if j.get('cause_of_action'))

        return {
            "total_judgments": len(judgments),
            "courts": list(courts),
            "court_count": len(courts),
            "causes": list(causes),
            "cause_count": len(causes),
        }

    def _update_metadata(self, elements: JudgmentElements, filename: str) -> None:
        """更新元数据索引"""
        metadata_path = self.kb_path / "indices" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {"judgments": []}

        entry = {
            "file": filename,
            "case_number": elements.case_number,
            "case_number_normalized": normalize_case_number(elements.case_number),
            "court": elements.court,
            "procedure": elements.procedure,
            "document_type": elements.document_type,
            "cause_of_action": elements.cause_of_action,
            "parties": [{"role": p.role, "name": p.name} for p in elements.parties],
            "applied_laws": elements.applied_laws,
            "parse_confidence": elements.parse_confidence,
            "added_at": datetime.now().isoformat(),
        }

        # 去重（按案号）——若已存在同案号条目，告警而非静默覆盖
        normalized_case = normalize_case_number(elements.case_number)
        if normalized_case:
            existing = [j for j in metadata["judgments"]
                        if (j.get("case_number_normalized")
                            or normalize_case_number(j.get("case_number") or "")) == normalized_case]
            if existing:
                print(f"  ⚠️  案号 {elements.case_number} 已存在于知识库，"
                      f"将替换旧条目（旧文件: {existing[0].get('file', '未知')} → 新文件: {filename}）")
            metadata["judgments"] = [
                j for j in metadata["judgments"]
                if (j.get("case_number_normalized")
                    or normalize_case_number(j.get("case_number") or "")) != normalized_case
            ]
        metadata["judgments"].append(entry)

        atomic_write_json(metadata_path, metadata)


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(description="裁判文书知识库")
    parser.add_argument('--kb', default=None, help='知识库路径')
    subparsers = parser.add_subparsers(dest="command")

    init_p = subparsers.add_parser("init", help="初始化知识库")
    init_p.add_argument("--name", required=True, help="知识库名称")

    add_p = subparsers.add_parser("add", help="添加文书")
    add_p.add_argument("file", help="文书文件路径")

    batch_p = subparsers.add_parser("batch", help="批量添加")
    batch_p.add_argument("directory", help="文书目录")

    search_p = subparsers.add_parser("search", help="检索")
    search_p.add_argument("--query", help="自然语言查询")
    search_p.add_argument("--case", help="案号")
    search_p.add_argument("--cause", help="案由")
    search_p.add_argument("--court", help="法院")
    search_p.add_argument("--party", help="当事人名称")
    search_p.add_argument("--law", help="适用法律（如 民法典第585条）")

    subparsers.add_parser("stats", help="统计")
    subparsers.add_parser("rebuild", help="重建索引")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    kb = JudgmentKnowledgeBase(args.kb)

    if args.command == "init":
        kb.init(args.name)
    elif args.command == "add":
        kb.add(args.file)
    elif args.command == "batch":
        kb.add_batch(args.directory)
    elif args.command == "search":
        results = kb.search(query=args.query, case_number=args.case,
                           cause=args.cause, court=args.court,
                           party=args.party, law=args.law)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.command == "stats":
        print(json.dumps(kb.stats(), ensure_ascii=False, indent=2))
    elif args.command == "rebuild":
        kb.rebuild_indices()


if __name__ == "__main__":
    main()
