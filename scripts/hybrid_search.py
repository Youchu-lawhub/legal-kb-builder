#!/usr/bin/env python3
"""
混合检索引擎（v2.1 - 三路融合 + Cross-Encoder 重排版）

v1 问题：
  - _extract_keywords() 仅硬编码 10 个法律术语，回退到单字分割
  - _calculate_match_score() 只是 count * length 权重（不及 TF-IDF）
  - 从不使用 _知识库索引.json 中的 topic_index
  - 声称"混合"但仅做关键词计数

v2 改进：
  - 三路检索：索引查找 + BM25 + 向量检索
  - Reciprocal Rank Fusion (RRF) 融合多路结果
  - jieba 分词 + 同义词扩展（取代硬编码术语）
  - 充分利用 _知识库索引.json 的 topic_index 和 structural_articles
  - 保留原有的精确法条检索和学者观点检索（仍工作正常）

v2.1 改进（2025 前沿补齐 P0）：
  - RRF 融合后接入 Cross-Encoder 重排（bge-reranker-v2-m3）
  - 流程：RRF 融合 → 取 Top-30 候选 → 读取文本 → Cross-Encoder 精排 → 取 Top-K
  - 重排为可选依赖，未安装 sentence-transformers 时自动降级为纯 RRF（取 fused[:5]）
  - 重排开关与 top_k 可通过构造参数控制
"""

import argparse
import contextlib
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shared_utils import chinese_to_number, normalize_article, number_to_chinese, load_synonym_expansion, vector_index_exists, file_signature

# 延迟导入检索模块（允许在无依赖环境中做精确检索）
_bm25_searcher = None
_embedding_manager = None
_reranker = None


def _get_bm25():
    global _bm25_searcher
    if _bm25_searcher is None:
        try:
            from bm25_searcher import BM25Searcher
            _bm25_searcher = BM25Searcher()
        except ImportError:
            pass
    return _bm25_searcher


def _get_embedder():
    global _embedding_manager
    if _embedding_manager is None:
        try:
            from embedding_manager import EmbeddingManager
            _embedding_manager = EmbeddingManager()
        except ImportError:
            pass
    return _embedding_manager


def _get_reranker():
    """获取 Cross-Encoder 重排序器单例（可选依赖，未安装时返回 None）"""
    global _reranker
    if _reranker is None:
        try:
            from reranker import get_reranker
            _reranker = get_reranker()
        except ImportError:
            pass
    return _reranker


# ─── jieba 分词（可选依赖） ───
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


# 停用词
STOPWORDS = {
    '的', '了', '是', '在', '和', '与', '或', '对', '把', '被',
    '让', '向', '从', '到', '给', '用', '以', '而', '但', '也',
    '都', '不', '没', '有', '这', '那', '就', '要', '会', '能',
    '可', '所', '其', '之', '上', '下', '中', '大', '小', '多',
    '少', '各', '每', '某', '本', '该', '此', '什么', '怎么',
    '如何', '为什么', '哪些', '哪个', '关于',
}


class HybridSearch:
    """混合检索引擎（v2.1 - 三路融合 + Cross-Encoder 重排版）"""

    def __init__(
        self,
        kb_path: Path,
        rerank_enabled: bool = True,
        rerank_top_k: int = 5,
        rerank_candidate_pool: int = 30,
    ):
        """
        Args:
            kb_path: 知识库根目录
            rerank_enabled: 是否在 RRF 融合后启用 Cross-Encoder 重排（默认 True）。
                            依赖未安装时自动降级为纯 RRF。
            rerank_top_k: 重排后输出的最终结果数（默认 5）
            rerank_candidate_pool: 重排候选池大小，从 RRF 结果取前 N 个做重排（默认 30）
        """
        self.kb_path = Path(kb_path)
        self.books_dir = self.kb_path / "books"
        self.indices_dir = self.kb_path / "indices"
        self._synonym_map = load_synonym_expansion()
        self._index_cache: Dict[str, Any] = {}  # 缓存已加载的 JSON 索引

        # 重排配置
        self._rerank_enabled = rerank_enabled
        self._rerank_top_k = rerank_top_k
        self._rerank_candidate_pool = rerank_candidate_pool
        self._query_rewrite_enabled = True
        self._query_rewrite_mode = "rule"
        self._graph_enabled = True
        self._rrf_k = 60
        self._fusion_weights = [1.0, 1.0, 1.0, 1.0]
        self._vector_config: Dict[str, Any] = {}
        self._embedder = None
        self._load_search_options()

    def _load_search_options(self) -> None:
        """读取 KB 根目录的可选检索配置；文件缺失时使用安全默认值。"""
        config_path = self.kb_path / "config.yaml"
        if not config_path.exists():
            return
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            search = config.get("search", {})
            rerank = search.get("rerank", {})
            rewrite = search.get("query_rewrite", {})
            graph = search.get("graph", {})
            fusion = search.get("fusion", {})
            vector = search.get("vector", {})
            self._rerank_enabled = rerank.get("enabled", self._rerank_enabled)
            self._rerank_top_k = rerank.get("top_k", self._rerank_top_k)
            self._rerank_candidate_pool = rerank.get("candidate_pool", self._rerank_candidate_pool)
            self._query_rewrite_enabled = rewrite.get("enabled", self._query_rewrite_enabled)
            self._query_rewrite_mode = rewrite.get("mode", self._query_rewrite_mode)
            self._graph_enabled = graph.get("enabled", self._graph_enabled)
            self._rrf_k = fusion.get("rrf_k", self._rrf_k)
            weights = fusion.get("weights", {})
            self._fusion_weights = [
                float(weights.get("index_lookup", 1.0)),
                float(weights.get("bm25", 1.0)),
                float(weights.get("vector", 1.0)),
                float(weights.get("graph", 1.0)),
            ]
            self._vector_config = vector if isinstance(vector, dict) else {}
        except Exception as exc:
            print(f"[HybridSearch] 忽略无效检索配置: {exc}", file=sys.stderr)

    # ═══════════════════════════════════════════
    # 公开检索接口
    # ═══════════════════════════════════════════

    def search_by_article(self, article: str, book: str = None) -> Dict[str, Any]:
        """按法条编号精确检索（保持原有逻辑）"""
        normalized = normalize_article(article)
        if not normalized:
            return {"error": f"无法解析法条编号: {article}"}

        results = {"query": article, "normalized": normalized, "results": []}

        books = [book] if book else self._list_books()
        for book_name in books:
            book_result = self._search_article_in_book(book_name, normalized)
            if book_result:
                results["results"].append(book_result)

        return results

    def search_by_query(self, query: str, book: str = None, use_ai: bool = True) -> Dict[str, Any]:
        """
        自然语言查询检索（v2 三路融合）

        策略：
        1. 检查是否包含法条编号 -> 精确检索
        2. 否则执行三路混合检索（索引 + BM25 + 向量）并 RRF 融合
        """
        original_query = query
        rewrite_info = None
        if self._query_rewrite_enabled:
            try:
                from query_rewriter import get_rewriter
                rewrite_info = get_rewriter(self._query_rewrite_mode).rewrite(query)
                query = rewrite_info.get("rewritten", query)
                expansions = rewrite_info.get("expansions", [])
                if expansions:
                    query = " ".join([query, *expansions])
            except Exception as exc:
                print(f"[HybridSearch] 查询改写失败，使用原查询: {exc}", file=sys.stderr)

        # 检查是否包含法条引用
        article_pattern = r'第([一二三四五六七八九十百千零\d]+)条'
        article_match = re.search(article_pattern, query)

        if article_match:
            article = f"第{article_match.group(1)}条"
            result = self.search_by_article(article, book)
            result["original_query"] = original_query
            if rewrite_info:
                result["query_rewrite"] = rewrite_info
            return result

        # 三路混合检索
        hybrid_results = self._hybrid_search(query, book)
        hybrid_results["original_query"] = original_query
        if rewrite_info:
            hybrid_results["query_rewrite"] = rewrite_info

        # AI 增强提示（在 QoderWork 中使用）
        if use_ai and hybrid_results.get("results"):
            return self._ai_enhance_results(query, hybrid_results)

        return hybrid_results

    def search_scholar_view(self, scholar: str, topic: str, book: str = None) -> Dict[str, Any]:
        """检索学者观点（保持原有逻辑）"""
        results = {
            "query": f"{scholar} 关于 {topic} 的观点",
            "scholar": scholar, "topic": topic, "results": []
        }

        books = [book] if book else self._list_books()
        for book_name in books:
            book_result = self._search_scholar_in_book(book_name, scholar, topic)
            if book_result and book_result.get("views"):
                results["results"].append(book_result)

        return results

    # ═══════════════════════════════════════════
    # 核心：三路混合检索 + RRF 融合
    # ═══════════════════════════════════════════

    def _hybrid_search(self, query: str, book: str = None) -> Dict[str, Any]:
        """
        执行三路混合检索并用 RRF 融合结果。

        Path A: 索引查找（_知识库索引.json 的 topic_index）
        Path B: BM25 关键词检索（jieba + rank_bm25）
        Path C: 向量语义检索（FAISS + bge）
        Path D: GraphRAG 法条关联图谱（多跳检索，v2.2 新增）
        """
        keywords = self._extract_keywords(query)

        results = {
            "query": query,
            "keywords": keywords,
            "search_paths_used": [],
            "results": []
        }

        books = [book] if book else self._list_books()

        for book_name in books:
            book_dir = self.books_dir / book_name
            if not book_dir.exists():
                continue

            # Path A: 索引查找
            index_results = self._index_lookup(book_name, query, keywords)

            # Path B: BM25
            bm25_results = self._bm25_search(book_dir, query)

            # Path C: 向量检索
            vector_results = self._vector_search(book_dir, query)

            # Path D: GraphRAG 图谱检索（v2.2 新增）
            graph_results = self._graph_search(book_dir, query) if self._graph_enabled else []

            # 记录使用了哪些路径
            paths_used = []
            if index_results:
                paths_used.append("index")
            if bm25_results:
                paths_used.append("bm25")
            if vector_results:
                paths_used.append("vector")
            if graph_results:
                paths_used.append("graph")

            if not paths_used:
                # 所有路径都没结果，降级到简单关键词匹配
                fallback = self._fallback_keyword_search(book_dir, keywords)
                if fallback:
                    paths_used.append("fallback")
                    bm25_results = fallback

            if not any([index_results, bm25_results, vector_results, graph_results]):
                continue

            # RRF 融合（四路）
            fused = self._reciprocal_rank_fusion(
                index_results or [],
                bm25_results or [],
                vector_results or [],
                graph_results or [],
                k=self._rrf_k,
                weights=self._fusion_weights,
            )

            if not fused:
                continue

            # Cross-Encoder 重排（v2.1 新增）
            # 流程：RRF 融合 → 取 Top-N 候选 → 读取文本 → 精排 → 取 Top-K
            final_ranked = self._rerank_candidates(query, book_dir, fused)

            # 读取实际文本片段
            top_matches = []
            for chunk_id, score in final_ranked:
                snippet = self._read_chunk_snippet(book_dir, chunk_id)
                top_matches.append({
                    "chunk_id": chunk_id,
                    "score": score,
                    "file": chunk_id.split(':')[0] if ':' in chunk_id else chunk_id,
                    "snippets": [snippet] if snippet else [],
                })

            # 加载书籍信息
            index_path = book_dir / "_book_index.json"
            book_info = {}
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as f:
                    idx = json.load(f)
                    book_info = idx.get("book_info", {})

            results["results"].append({
                "book": book_name,
                "book_info": book_info,
                "search_paths": paths_used,
                "match_count": len(fused),
                "top_matches": top_matches,
            })

        results["search_paths_used"] = list(set(
            p for r in results["results"] for p in r.get("search_paths", [])
        ))

        return results

    # ─── Path A: 索引查找 ───

    def _index_lookup(
        self, book_name: str, query: str, keywords: List[str]
    ) -> List[Tuple[str, float]]:
        """
        使用 _知识库索引.json 的 topic_index 进行精确查找。

        返回 [(chunk_id, score), ...] 其中 score 始终为 1.0
        """
        book_dir = self.books_dir / book_name

        # 优先加载目录/标题驱动的增强索引；合并器的 _知识库索引.json
        # 仅保存卷与来源信息，作为尚未生成增强索引时的降级来源。
        idx_path = book_dir / "_book_index.json"
        if not idx_path.exists():
            idx_path = book_dir / "_知识库索引.json"
        if not idx_path.exists():
            return []

        cache_key = str(idx_path)
        if cache_key not in self._index_cache:
            with open(idx_path, 'r', encoding='utf-8') as f:
                self._index_cache[cache_key] = json.load(f)

        index_data = self._index_cache[cache_key]

        results = []

        # 检查 topic_index
        topic_index = index_data.get("topic_index", {})
        for keyword in keywords:
            if keyword in topic_index:
                entries = topic_index[keyword]
                if isinstance(entries, list):
                    for entry in entries:
                        # entry 可能是条文号（int）或章节名（str）
                        chunk_id = f"topic:{keyword}:{entry}"
                        results.append((chunk_id, 1.0))

        # 检查 structural_articles（法典评注/司法解释类）
        structural = index_data.get("structural_articles", {})
        if structural:
            articles = structural.get("articles", {})
            for keyword in keywords:
                for art_key, art_info in articles.items():
                    if keyword in art_key or keyword in str(art_info.get("subject", "")):
                        if art_info.get("primary_file"):
                            chunk_id = f"{art_info['primary_file']}:line:{art_info.get('primary_line', 1)}"
                            results.append((chunk_id, 1.0))

        # 章节型书籍：优先使用 Markdown 标题定位。标题全匹配排在最前，
        # 其余按命中的关键词长度排序，避免“检索”等高频词压过目标章节。
        chapters = index_data.get("chapters", {}).get("entries", {})
        chapter_hits = []
        for title, location in chapters.items():
            if not isinstance(location, dict) or not location.get("file"):
                continue
            clean_title = location.get("title", title)
            if query in clean_title:
                score = 10_000
            else:
                score = sum(len(keyword) for keyword in keywords if keyword in clean_title)
            if score:
                chapter_hits.append((
                    f"{location['file']}:line:{location.get('line', 1)}",
                    float(score),
                ))
        chapter_hits.sort(key=lambda item: item[1], reverse=True)
        results.extend(chapter_hits)

        # 检查 volumes（所有类型）
        for vol in index_data.get("volumes", []):
            filename = vol.get("filename", "")
            for section in vol.get("sections", []):
                title = section.get("title", "")
                for keyword in keywords:
                    if keyword in title:
                        results.append((f"{filename}:0", 0.8))
                        break

        return results

    # ─── Path B: BM25 检索 ───

    def _bm25_search(
        self, book_dir: Path, query: str
    ) -> List[Tuple[str, float]]:
        """通过 BM25Searcher 进行关键词检索"""
        bm25 = _get_bm25()
        if bm25 is None:
            return []

        try:
            return bm25.search(book_dir, query, top_k=20)
        except Exception as e:
            print(f"[WARN] _bm25_search 执行异常，已跳过: {e}", file=sys.stderr)
            return []

    # ─── Path C: 向量检索 ───

    def _vector_search(
        self, book_dir: Path, query: str
    ) -> List[Tuple[str, float]]:
        """通过 EmbeddingManager 进行向量相似度检索"""
        try:
            if self._embedder is None:
                from embedding_manager import EmbeddingManager
                self._embedder = EmbeddingManager(
                    model_name=self._vector_config.get("model", "BAAI/bge-small-zh-v1.5"),
                    cache_dir=self._vector_config.get("model_cache") or None,
                )
            return self._embedder.search(book_dir, query, top_k=20)
        except Exception as e:
            print(f"[WARN] _vector_search 执行异常，已跳过: {e}", file=sys.stderr)
            return []

    # ─── Path D: GraphRAG 图谱检索（v2.2 新增） ───

    _graph_cache: Dict[str, Any] = {}  # 类级缓存 {kb_path_str: GraphRAG}

    def _graph_search(
        self, book_dir: Path, query: str
    ) -> List[Tuple[str, float]]:
        """
        通过 GraphRAG 做法条关联图谱多跳检索。

        将图谱查询结果转换为 (chunk_id, score) 格式，
        与其他三路结果一起参与 RRF 融合。

        图谱无索引时返回空列表（自动降级为三路融合）。
        """
        graph = self._get_graph(book_dir)
        if graph is None:
            return []

        try:
            results = graph.search(query, max_depth=2, top_k=10)
        except Exception as e:
            print(f"[WARN] _graph_search 执行异常，已跳过: {e}", file=sys.stderr)
            return []

        # 将图谱节点映射回建立图谱时记录的源文件，确保图谱结果可以
        # 返回正文片段而不是只有一个没有来源的实体标签。
        graph_results = []
        for r in results:
            node_id = r.get('node_id', '')
            score = 1.0 / (r['depth'] + 1) + min(r.get('mentions', 0), 100) * 0.001
            source_files = {
                edge.get('source_file', '') for edge in graph._graph_data.get('edges', [])
                if node_id in (edge.get('source'), edge.get('target')) and edge.get('source_file')
            }
            for source_file in source_files:
                graph_results.append((f"{source_file}:0", score))

        return graph_results

    @classmethod
    def _get_graph(cls, book_dir: Path) -> Optional[Any]:
        """获取或加载 GraphRAG 实例（带缓存）"""
        key = str(book_dir)
        signature = file_signature(book_dir / "_graph.json")
        cached = cls._graph_cache.get(key)
        if cached and cached[0] == signature:
            return cached[1]

        try:
            from graph_rag import GraphRAG
            graph = GraphRAG()
            if graph.load(book_dir):
                cls._graph_cache[key] = (signature, graph)
                return graph
        except ImportError:
            pass
        except Exception:
            pass

        cls._graph_cache[key] = None  # 标记无图谱
        return None

    # ─── 降级：简单关键词匹配 ───

    def _fallback_keyword_search(
        self, book_dir: Path, keywords: List[str]
    ) -> List[Tuple[str, float]]:
        """
        当 BM25 和向量检索都不可用时的降级方案。
        基于简单的关键词出现频率（比 v1 稍好，使用 jieba 分词）。
        """
        if not keywords:
            return []

        results = []
        for md_file in book_dir.glob("*.md"):
            if md_file.name.startswith("_"):
                continue

            content = md_file.read_text(encoding='utf-8')
            content_lower = content.lower()

            score = 0.0
            for kw in keywords:
                count = content_lower.count(kw.lower())
                weight = len(kw)  # 长关键词权重更高
                score += count * weight

            if score > 0:
                results.append((f"{md_file.name}:0", score / len(keywords)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:20]

    # ─── RRF 融合 ───

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: List[Tuple[str, float]],
        k: int = 60,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion (RRF)。

        将多个排序结果列表融合为一个统一排序。
        RRF_score(d) = sum(1 / (k + rank_i(d))) 对每个结果列表 i

        Args:
            *result_lists: 多个 [(chunk_id, score)] 列表
            k: RRF 常数（默认 60，标准值）

        Returns:
            [(chunk_id, rrf_score)] 按融合分数降序排列
        """
        scores: Dict[str, float] = defaultdict(float)

        for list_index, result_list in enumerate(result_lists):
            if not result_list:
                continue
            weight = weights[list_index] if weights and list_index < len(weights) else 1.0
            for rank, (chunk_id, _) in enumerate(result_list):
                scores[chunk_id] += weight / (k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    # ─── Cross-Encoder 重排（v2.1） ───

    def _rerank_candidates(
        self,
        query: str,
        book_dir: Path,
        fused: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        对 RRF 融合结果做 Cross-Encoder 重排。

        策略：
          1. 取 RRF 前 N（candidate_pool）个候选
          2. 读取每个候选的真实文本（用于 Cross-Encoder 打分）
          3. Cross-Encoder 精排，取前 K（rerank_top_k）个
          4. 依赖不可用时降级：直接返回 fused[:rerank_top_k]（纯 RRF）

        Args:
            query: 用户查询
            book_dir: 书籍目录（用于读取候选文本）
            fused: RRF 融合结果 [(chunk_id, rrf_score), ...]

        Returns:
            [(chunk_id, score), ...] 最多 rerank_top_k 条
            - 启用重排时 score 为 Cross-Encoder 分数
            - 降级时 score 为 RRF 分数
        """
        # 取候选池
        pool = fused[:self._rerank_candidate_pool]

        # 未启用重排或候选数不足 -> 直接返回前 K
        if not self._rerank_enabled or len(pool) <= self._rerank_top_k:
            return pool[:self._rerank_top_k]

        reranker = _get_reranker()
        if reranker is None or not reranker.is_available():
            # 降级：纯 RRF
            return pool[:self._rerank_top_k]

        # 读取候选文本（Cross-Encoder 需要 query-doc pair）
        candidate_texts = []
        for chunk_id, _ in pool:
            text = self._read_chunk_text(book_dir, chunk_id)
            candidate_texts.append(text if text else "")

        # 过滤空文本（保留原始索引映射）
        valid_pairs = []  # [(pool_idx, text)]
        for i, text in enumerate(candidate_texts):
            if text:
                valid_pairs.append((i, text))

        if len(valid_pairs) <= self._rerank_top_k:
            return pool[:self._rerank_top_k]

        # Cross-Encoder 打分
        texts = [t for _, t in valid_pairs]
        try:
            ranked = reranker.rerank(query, texts, top_k=self._rerank_top_k)
        except Exception as e:
            print(f"[Rerank] 重排异常，降级为纯 RRF: {e}")
            return pool[:self._rerank_top_k]

        # 映射回 chunk_id
        result = []
        for valid_idx, ce_score in ranked:
            pool_idx = valid_pairs[valid_idx][0]
            chunk_id = pool[pool_idx][0]
            result.append((chunk_id, ce_score))

        return result

    def _read_chunk_text(
        self, book_dir: Path, chunk_id: str, max_chars: int = 800
    ) -> str:
        """
        读取 chunk_id 对应的完整文本（供 Cross-Encoder 打分）。

        v2.2: parent_child 策略下优先返回父块文本（更完整上下文，重排更准）。
        比 _read_chunk_snippet 读取更长（800 字符），保证重排精度。
        """
        if ':' not in chunk_id:
            return ""

        parts = chunk_id.split(':')
        filename = parts[0]

        # topic:xxx:yyy 格式（来自索引查找，无对应文件文本）
        if filename == 'topic':
            return f"[索引匹配: {':'.join(parts[1:])}]"

        # 优先尝试通过 embedding_manager 获取父块文本（parent_child 策略）
        embedder = self._embedder or _get_embedder()
        if embedder is not None:
            try:
                parent_text = embedder.get_parent_text(book_dir, chunk_id)
                if parent_text:
                    return parent_text[:max_chars] if len(parent_text) > max_chars else parent_text
            except Exception:
                pass  # 降级到文件读取

        file_path = book_dir / filename
        if not file_path.exists():
            return ""

        try:
            start_char = int(parts[1])
        except (ValueError, IndexError):
            start_char = 0

        content = file_path.read_text(encoding='utf-8')
        end = min(start_char + max_chars, len(content))
        start = max(0, start_char)
        return content[start:end]

    # ─── 分词与同义词扩展 ───

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词（v2：使用 jieba + 同义词扩展）。
        """
        if HAS_JIEBA:
            words = list(jieba.cut(query, cut_all=False))
        else:
            # 降级：按标点分割
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', query)

        # 过滤停用词，保留有意义的词
        keywords = []
        for word in words:
            word = word.strip()
            if word and word not in STOPWORDS and len(word) >= 2:
                keywords.append(word)

        # 同义词扩展
        expanded = list(keywords)
        seen = set(keywords)
        for kw in keywords:
            synonyms = self._synonym_map.get(kw, [])
            for syn in synonyms:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

        return list(set(expanded))

    # ─── 结果读取 ───

    def _read_chunk_snippet(self, book_dir: Path, chunk_id: str, size: int = 200) -> str:
        """
        读取 chunk_id 对应的文本片段。

        v2.2: 若 _chunks.json 标记为 parent_child 策略，优先返回父块文本
        （完整上下文），否则降级为从文件读取 size 字符。
        """
        if ':' not in chunk_id:
            return ""

        parts = chunk_id.split(':')
        filename = parts[0]

        # topic:xxx:yyy 格式（来自索引查找）
        if filename == 'topic':
            return f"[索引匹配: {':'.join(parts[1:])}]"

        # 优先尝试通过 embedding_manager 获取父块文本（parent_child 策略）
        embedder = self._embedder or _get_embedder()
        if embedder is not None:
            try:
                parent_text = embedder.get_parent_text(book_dir, chunk_id)
                if parent_text:
                    # 父块可能较长，截取 size 字符作为 snippet
                    return parent_text[:size] if len(parent_text) > size else parent_text
            except Exception:
                pass  # 降级到文件读取

        file_path = book_dir / filename
        if not file_path.exists():
            return ""

        content = file_path.read_text(encoding='utf-8')
        if len(parts) >= 3 and parts[1] == 'line':
            try:
                line_no = max(1, int(parts[2]))
                lines = content.splitlines(keepends=True)
                start_char = sum(len(line) for line in lines[:line_no - 1])
            except ValueError:
                start_char = 0
        else:
            try:
                start_char = int(parts[1])
            except (ValueError, IndexError):
                start_char = 0
        end = min(start_char + size, len(content))
        start = max(0, start_char)
        return content[start:end]

    # ═══════════════════════════════════════════
    # 保持原有的精确检索方法（不变）
    # ═══════════════════════════════════════════

    def _search_article_in_book(self, book_name: str, article: str) -> Optional[Dict[str, Any]]:
        """在单本书中检索法条"""
        book_dir = self.books_dir / book_name
        index_path = book_dir / "_book_index.json"

        if not index_path.exists():
            # 由 merge_md.py 构建的轻量书籍库只有 _知识库索引.json。
            # 优先命中标题行，找不到时才回退到正文引用，且明确标出来源。
            merged_index_path = book_dir / "_知识库索引.json"
            if not merged_index_path.exists():
                return None
            try:
                with open(merged_index_path, 'r', encoding='utf-8') as f:
                    merged = json.load(f)
            except (OSError, json.JSONDecodeError):
                return None

            number = chinese_to_number(article)
            candidates = {article}
            if number is not None:
                candidates.add(f"第{number}条")
                candidates.add(f"第{number_to_chinese(number)}条")
            heading_locations, mention_locations = [], []
            for md_file in sorted(book_dir.glob("*.md")):
                lines = md_file.read_text(encoding='utf-8').splitlines()
                for line_no, line in enumerate(lines, 1):
                    if not any(candidate in line for candidate in candidates):
                        continue
                    start = max(0, line_no - 1)
                    location = {
                        "file": md_file.name,
                        "line": line_no,
                        "context": "\n".join(lines[start:start + 12]),
                    }
                    if line.lstrip().startswith('#') or line.strip() in candidates:
                        heading_locations.append(location)
                    else:
                        mention_locations.append(location)
            locations = heading_locations or mention_locations
            if not locations:
                return None
            contents = [
                {"location": loc, "content": self._read_location(book_dir, loc)}
                for loc in locations
            ]
            return {
                "book": book_name,
                "book_info": {"name": merged.get("name", book_name)},
                "article": article,
                "occurrences": len(contents),
                "contents": [item for item in contents if item["content"]],
                "index_source": "merge_index_fallback",
            }

        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        article_index = index.get("article_index", {})
        locations = article_index.get(article, [])
        if not locations:
            return None

        contents = []
        for loc in locations:
            content = self._read_location(book_dir, loc)
            if content:
                contents.append({"location": loc, "content": content})

        return {
            "book": book_name,
            "book_info": index.get("book_info", {}),
            "article": article,
            "occurrences": len(locations),
            "contents": contents
        }

    def _search_scholar_in_book(self, book_name: str, scholar: str, topic: str) -> Optional[Dict[str, Any]]:
        """在单本书中检索学者观点"""
        book_dir = self.books_dir / book_name
        index_path = book_dir / "_book_index.json"

        if not index_path.exists():
            return None

        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        scholar_index = index.get("scholar_index", {})
        views = scholar_index.get(scholar, [])

        filtered_views = [v for v in views if self._is_topic_related(v.get("view", ""), topic)]
        if not filtered_views:
            return None

        full_views = []
        for view in filtered_views:
            content = self._read_location(book_dir, view)
            if content:
                full_views.append({
                    "view_summary": view.get("view", ""),
                    "location": view,
                    "full_content": content
                })

        return {
            "book": book_name,
            "book_info": index.get("book_info", {}),
            "scholar": scholar,
            "topic": topic,
            "view_count": len(full_views),
            "views": full_views
        }

    def _read_location(self, book_dir: Path, location: Dict[str, Any]) -> str:
        """读取指定位置的内容"""
        file_path = book_dir / location.get("file", "")
        if not file_path.exists():
            return ""
        context = location.get("context", "")
        if context:
            return context
        line_no = location.get("line")
        if isinstance(line_no, int) and line_no > 0:
            lines = file_path.read_text(encoding='utf-8').splitlines()
            return '\n'.join(lines[line_no - 1:line_no + 11])
        return ""

    def _is_topic_related(self, text: str, topic: str) -> bool:
        """判断文本是否与主题相关"""
        if topic in text:
            return True
        similarity = SequenceMatcher(None, text, topic).ratio()
        return similarity > 0.3

    def _ai_enhance_results(self, query: str, keyword_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        为 QoderWork AI 准备增强上下文。
        """
        context_parts = []
        for book_result in keyword_results.get("results", [])[:3]:
            book_name = book_result.get("book", "")
            for match in book_result.get("top_matches", [])[:2]:
                snippets = match.get("snippets", [])
                if snippets:
                    context_parts.append(f"[{book_name}]\n{snippets[0]}")

        context = "\n\n".join(context_parts)

        keyword_results["ai_enhanced"] = True
        keyword_results["context_for_ai"] = context
        keyword_results["suggested_prompt"] = (
            f"基于以下法律知识库检索结果，回答用户问题：\n\n"
            f"用户问题：{query}\n\n"
            f"检索到的相关内容：\n{context}\n\n"
            f"请：\n1. 总结相关法律规定和学说观点\n"
            f"2. 指出不同学者观点的差异（如有）\n"
            f"3. 给出结论性意见"
        )

        return keyword_results

    def _list_books(self) -> List[str]:
        """列出所有书籍"""
        if not self.books_dir.exists():
            return []
        return [
            item.name for item in self.books_dir.iterdir()
            if item.is_dir() and not item.name.startswith("_")
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法律知识库四路融合检索")
    parser.add_argument("kb_path", nargs="?", help="知识库根目录（兼容旧位置参数）")
    parser.add_argument("query_pos", nargs="?", help="查询语句（兼容旧位置参数）")
    parser.add_argument("--kb", help="知识库根目录")
    parser.add_argument("--query", help="自然语言查询")
    parser.add_argument("--article", help="法条精确检索")
    parser.add_argument("--book", help="限定 books/ 下的书籍目录名")
    parser.add_argument("--top-k", type=int, default=5, help="最多返回结果数")
    parser.add_argument("--no-ai", action="store_true", help="不生成下游回答提示")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（默认）")
    args = parser.parse_args()

    kb_path = args.kb or args.kb_path
    query = args.query or args.query_pos
    if not kb_path or not (query or args.article):
        parser.error("请提供 --kb/--query，或兼容的位置参数 <kb_path> <query>")

    # 机器可读模式必须只在 stdout 输出 JSON；底层可选依赖的诊断信息
    # 转到 stderr，供 consultation_agent / MCP 稳定解析。
    output_redirect = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with output_redirect:
        searcher = HybridSearch(Path(kb_path), rerank_top_k=args.top_k)
        if args.article:
            result = searcher.search_by_article(args.article, args.book)
        else:
            result = searcher.search_by_query(query, args.book, use_ai=not args.no_ai)
    print(json.dumps(result, ensure_ascii=False, indent=2))
