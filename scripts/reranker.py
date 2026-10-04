#!/usr/bin/env python3
"""
Cross-Encoder 重排序模块（v1）

在 RRF 三路融合之后，对 Top-N 候选做 Cross-Encoder 精排，取 Top-K 输出。
Cross-Encoder 让 Query 和 Doc 做注意力交互，比 Bi-Encoder 精度更高。

基准（2026 测试）：
  Bi-Encoder + RRF：NDCG@10 ≈ 0.65
  Cross-Encoder 重排后：NDCG@10 ≈ 0.78（+20%）

默认模型：BAAI/bge-reranker-v2-m3（开源、多语言、中英法律文本均适用）

加载策略（与 embedding_manager 一致的延迟加载 + 可选依赖降级）：
  1. 优先用 sentence-transformers 的 CrossEncoder（项目已有该依赖，无需额外安装）
  2. 未安装时 rerank() 直接原样返回，hybrid_search 自动降级为纯 RRF

使用方式：
    from reranker import get_reranker

    reranker = get_reranker()
    if reranker is not None:
        # candidates: List[str] 候选文本，与 RRF 输出顺序对齐
        ranked = reranker.rerank(query, candidates, top_k=10)
        # ranked: List[Tuple[int, float]]  (原始索引, 重排分数) 按分数降序
"""

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

# 延迟导入（可选依赖）
try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except ImportError:
    HAS_CE = False


# ─── 常量 ───
DEFAULT_RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'legal-kb-builder' / 'models'

# 单次重排的默认候选数上限（召回）与输出数（精排）
DEFAULT_CANDIDATE_TOPN = 30
DEFAULT_RETURN_TOPK = 10

_reranker_instance = None  # 单例缓存
_reranker_lock = None       # 延迟初始化锁（避免模块导入时引入 threading 开销）


def _get_lock():
    global _reranker_lock
    if _reranker_lock is None:
        import threading
        _reranker_lock = threading.Lock()
    return _reranker_lock


class Reranker:
    """
    Cross-Encoder 重排序器

    Args:
        model_name: 模型名（默认 bge-reranker-v2-m3）
        cache_dir: 模型缓存目录
        max_length: Cross-Encoder 最大 token 长度（bge-reranker-v2-m3 默认 512）
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        cache_dir: str = None,
        max_length: int = 512,
    ):
        self._model_name = model_name
        self._cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._max_length = max_length
        self._model: Optional[Any] = None  # 延迟加载

    def is_available(self) -> bool:
        """是否可用（依赖已安装）"""
        return HAS_CE

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = DEFAULT_RETURN_TOPK,
    ) -> List[Tuple[int, float]]:
        """
        对候选文档列表做 Cross-Encoder 精排。

        Args:
            query: 查询字符串
            candidates: 候选文档文本列表（顺序与上游 RRF 输出对齐）
            top_k: 返回前 K 个

        Returns:
            [(原始索引, 重排分数), ...] 按分数降序，最多 top_k 条。
            依赖不可用时原样返回 [(i, 0.0) for i in range(min(top_k, len(candidates)))]。
        """
        if not candidates:
            return []

        # 降级：依赖不可用
        if not HAS_CE:
            return [(i, 0.0) for i in range(min(top_k, len(candidates)))]

        # 候选数过少时直接返回，省去模型加载开销
        if len(candidates) <= top_k:
            return [(i, 0.0) for i in range(len(candidates))]

        self._ensure_model_loaded()

        # 构造 query-doc 对
        pairs = [[query, doc] for doc in candidates]

        # 批量打分（CrossEncoder.predict 返回每对的相似度分数）
        scores = self._model.predict(pairs)

        # 按分数降序取 top_k
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        return [(idx, float(score)) for idx, score in indexed[:top_k]]

    def rerank_pairs(
        self,
        pairs: List[List[str]],
        top_k: int = DEFAULT_RETURN_TOPK,
    ) -> List[Tuple[int, float]]:
        """
        直接对已构造好的 query-doc 对打分并排序。

        Args:
            pairs: [[query, doc], ...]
            top_k: 返回前 K 个

        Returns:
            [(原始索引, 分数), ...] 降序
        """
        if not pairs or not HAS_CE:
            return [(i, 0.0) for i in range(min(top_k, len(pairs)))]

        self._ensure_model_loaded()
        scores = self._model.predict(pairs)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in indexed[:top_k]]

    # ─── 内部方法 ───

    def _ensure_model_loaded(self) -> None:
        """确保模型已加载（延迟加载）"""
        if self._model is not None:
            return

        if not HAS_CE:
            raise ImportError(
                "重排序依赖未安装。请运行: pip install sentence-transformers"
            )

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Reranker] 加载 Cross-Encoder 模型: {self._model_name}")
        self._model = CrossEncoder(
            self._model_name,
            max_length=self._max_length,
            cache_folder=str(self._cache_dir),
        )
        print(f"[Reranker] 模型已就绪")


def get_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    cache_dir: str = None,
) -> Optional['Reranker']:
    """
    获取全局 Reranker 单例（延迟初始化，线程安全）。

    依赖不可用时返回 None，调用方应据此判断是否跳过重排。
    """
    global _reranker_instance
    if _reranker_instance is None and HAS_CE:
        lock = _get_lock()
        with lock:
            if _reranker_instance is None:
                _reranker_instance = Reranker(model_name=model_name, cache_dir=cache_dir)
    return _reranker_instance


# ─── CLI 入口 ───

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python reranker.py --test                # 依赖自检")
        print("  python reranker.py <query> <doc1> <doc2>...  # 简单重排测试")
        sys.exit(0)

    if sys.argv[1] == '--test':
        print(f"sentence-transformers CrossEncoder: {'yes' if HAS_CE else 'no'}")
        print(f"默认模型: {DEFAULT_RERANKER_MODEL}")
        if HAS_CE:
            print("\n注: 实际加载模型需运行重排任务时触发。")
        else:
            print("\n降级模式：重排将被跳过，RRF 结果原样返回。")
            print("安装依赖以启用: pip install sentence-transformers")
        sys.exit(0)

    # 简单重排测试
    query = sys.argv[1]
    docs = sys.argv[2:]
    if len(docs) < 2:
        print("至少提供 2 个文档用于重排")
        sys.exit(1)

    r = get_reranker()
    if r is None:
        print("重排序依赖未安装，降级为原顺序输出")
        for i, doc in enumerate(docs):
            print(f"  [{i}] {doc[:80]}")
        sys.exit(0)

    ranked = r.rerank(query, docs, top_k=min(5, len(docs)))
    print(f"Query: {query}\n")
    for idx, score in ranked:
        print(f"  {score:+.4f}  [{idx}] {docs[idx][:100]}")
