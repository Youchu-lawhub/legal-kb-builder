#!/usr/bin/env python3
"""
法律知识库 RAG 评估脚本

评估流程：
  1. 读取 eval/legal_qa_eval.jsonl 标注集
  2. 对每条 query 调用 hybrid_search 检索
  3. 计算检索质量指标 + 生成质量指标

两套指标：

  A. 基础检索指标（无外部依赖，始终可用）：
     - keyword_hit_rate@k / keyword_recall@k: 关键词诊断（不代表相关性）
     - hit_rate@k / mrr@k / recall@k: 仅在提供 relevant_ids 标注时计算

  B. RAGAS 指标（需 pip install ragas，可选）：
     - faithfulness:        答案是否完全基于检索上下文
     - answer_relevance:    答案是否回答了问题
     - context_precision:   相关文档是否排在前面
     - context_recall:      答案所需信息是否都在上下文
     - answer_correctness:  答案与标准答案的匹配度

用法：
    # 基础检索评估（无需 LLM）
    python eval/eval_rag.py --kb-path ~/legal_kb/books/民法典评注

    # 指定 K 值
    python eval/eval_rag.py --kb-path ~/legal_kb/books/民法典评注 --top-k 5

    # 完整 RAGAS 评估（需安装 ragas，并提供待评估系统实际生成的答案）
    python eval/eval_rag.py --kb-path ~/legal_kb --full --answers-file answers.jsonl

    # 对比两次检索（参数变化前后）
    python eval/eval_rag.py --kb-path ~/kb --label "rerank_enabled"

输出：
    控制台指标摘要 + eval/results/<timestamp>_<label>.json 详细结果
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

EVAL_DIR = _SCRIPT_DIR
EVAL_SET = EVAL_DIR / "legal_qa_eval.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


# ─── 标注集加载 ───

def load_eval_set(path: Path = EVAL_SET) -> List[Dict[str, Any]]:
    """加载 JSONL 格式的评估集"""
    if not path.exists():
        print(f"评估集不存在: {path}")
        return []
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_generated_answers(path: Path) -> Dict[str, str]:
    """读取系统实际回答，JSONL 每行必须含 ``query`` 和 ``answer``。"""
    if not path.exists():
        raise FileNotFoundError(f"答案文件不存在: {path}")
    answers = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            query, answer = item.get("query"), item.get("answer")
            if not isinstance(query, str) or not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"答案文件第 {line_no} 行需要非空 query 与 answer")
            answers[query] = answer
    return answers


# ─── 基础检索指标（无依赖） ───

def compute_retrieval_metrics(query, retrieved_items, relevant_keywords, top_k=5, relevant_ids=None):
    """Return standard ranking metrics for qrels; keywords are diagnostics only."""
    if top_k < 1:
        raise ValueError("top_k 必须 >= 1")
    top_items = retrieved_items[:top_k]
    keyword_hit = any(any(keyword.lower() in item["text"].lower()
                          for keyword in relevant_keywords)
                      for item in top_items) if relevant_keywords else False
    combined = " ".join(item["text"] for item in top_items).lower()
    keyword_recall = (sum(keyword.lower() in combined for keyword in relevant_keywords) /
                      len(relevant_keywords)) if relevant_keywords else 0.0
    if relevant_ids is None:
        return {"keyword_hit_rate@k": float(keyword_hit),
                "keyword_recall@k": float(keyword_recall),
                "keyword_rank_metrics_available": 0.0}
    relevant = set(relevant_ids)
    ranks = [rank for rank, item in enumerate(top_items, 1) if item["id"] in relevant]
    return {"hit_rate@k": float(bool(ranks)),
            "mrr@k": 1.0 / min(ranks) if ranks else 0.0,
            "recall@k": len({item["id"] for item in top_items} & relevant) / len(relevant) if relevant else 0.0,
            "keyword_hit_rate@k": float(keyword_hit),
            "keyword_recall@k": float(keyword_recall),
            "keyword_rank_metrics_available": 1.0}


# ─── RAGAS 指标（可选依赖） ───

def try_ragas_eval(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, float]:
    """
    尝试用 RAGAS 计算 5 大指标。

    依赖不可用或计算失败时显式报错，避免将未完成的评估当作成功。
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall,
            answer_correctness,
        )
        from datasets import Dataset
    except ImportError:
        raise RuntimeError(
            "完整 RAGAS 评估依赖缺失；请安装 ragas 与 datasets 后重试"
        ) from None

    try:
        data = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        })
        result = evaluate(
            data,
            metrics=[
                faithfulness,
                answer_relevance,
                context_precision,
                context_recall,
                answer_correctness,
            ],
        )
        return {k: float(v) for k, v in result.items()}
    except Exception as e:
        print(f"  [RAGAS] 评估失败: {e}")
        raise RuntimeError(f"RAGAS 完整评估失败: {e}") from e


# ─── 主评估流程 ───

def run_evaluation(
    kb_path: str,
    top_k: int = 5,
    full: bool = False,
    label: str = "",
    answers_file: str = "",
) -> Dict[str, Any]:
    """
    对评估集执行检索评估。

    Args:
        kb_path: 知识库路径
        top_k: 检索 Top-K
        full: 是否启用 RAGAS 完整评估；必须配合 answers_file
        label: 结果标签（用于对比实验）
        answers_file: 系统实际生成答案的 JSONL 文件

    Returns:
        评估结果汇总
    """
    samples = load_eval_set()
    if not samples:
        raise ValueError(f"评估集为空或不存在: {EVAL_SET}")
    if top_k < 1:
        raise ValueError("top_k 必须 >= 1")

    # 延迟导入检索引擎
    try:
        from hybrid_search import HybridSearch
    except ImportError as e:
        raise RuntimeError(f"无法导入 HybridSearch: {e}") from e

    if full and not answers_file:
        raise ValueError("--full 需要 --answers-file；RAGAS 不能把检索提示词当作模型答案")
    generated_answers = load_generated_answers(Path(answers_file)) if full else {}
    searcher = HybridSearch(Path(kb_path))

    results = []
    for i, sample in enumerate(samples, 1):
        query = sample["query"]
        relevant_keywords = sample.get("relevant_keywords", [])
        ground_truth = sample.get("ground_truth_answer", "")

        print(f"\n[{i}/{len(samples)}] {query}")

        # 执行检索
        try:
            search_result = searcher.search_by_query(query, use_ai=False)
        except Exception as e:
            raise RuntimeError(f"检索失败，题目 {i} 未完成评估: {e}") from e
        if not isinstance(search_result, dict) or not isinstance(search_result.get("results"), list):
            raise RuntimeError(f"检索器对题目 {i} 返回了无效结果，评估未完成")

        # 提取检索片段
        # Merge results across books into a true global ranking.
        passages = []
        for book_result in search_result.get("results", []):
            for match in book_result.get("top_matches", []):
                for snippet in match.get("snippets", []):
                    passages.append({"id": f"{book_result.get('book','')}::{match.get('chunk_id','')}",
                                     "text": snippet, "score": float(match.get("score", 0.0))})
        passages.sort(key=lambda item: item["score"], reverse=True)
        snippets = [item["text"] for item in passages]

        metrics = compute_retrieval_metrics(
            query, passages, relevant_keywords, top_k=top_k,
            relevant_ids=sample.get("relevant_ids"),
        )
        if "mrr@k" in metrics:
            print(f"  qrels hit@{top_k}={metrics['hit_rate@k']:.0f} "
                  f"mrr@{top_k}={metrics['mrr@k']:.2f} "
                  f"recall@{top_k}={metrics['recall@k']:.2f}")
        else:
            print(f"  keyword diagnostics only: hit={metrics['keyword_hit_rate@k']:.0f} "
                  f"keyword coverage={metrics['keyword_recall@k']:.2f}; no qrels")

        # RAGAS 指标（可选）
        ragas_metrics = {}
        if full:
            answer = generated_answers.get(query)
            if not answer:
                raise ValueError(f"答案文件缺少 query 的回答: {query}")
            ragas_metrics = try_ragas_eval(
                query, answer, snippets[:top_k], ground_truth
            )
            print(f"  RAGAS: " + " ".join(
                f"{k}={v:.2f}" for k, v in ragas_metrics.items()
            ))

        results.append({
            "query": query,
            "category": sample.get("category", ""),
            "retrieved_count": len(snippets),
            "evaluation_kind": "document_qrels" if sample.get("relevant_ids") else "keyword_diagnostic_only",
            "metrics": {**metrics, **ragas_metrics},
        })

    # 汇总
    summary = summarize(results)
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    # 保存结果
    save_result(summary, results, label, kb_path)

    return {"summary": summary, "details": results}


def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """汇总所有样本的指标平均值"""
    if not results:
        return {}
    metric_keys = set()
    for r in results:
        metric_keys.update(r["metrics"].keys())

    summary = {}
    for key in metric_keys:
        values = [r["metrics"][key] for r in results if key in r["metrics"]]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def save_result(
    summary: Dict[str, float],
    details: List[Dict[str, Any]],
    label: str,
    kb_path: str,
) -> Path:
    """保存评估结果到文件"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_suffix = f"_{label}" if label else ""
    out_path = RESULTS_DIR / f"{ts}{label_suffix}.json"

    output = {
        "timestamp": ts,
        "kb_path": kb_path,
        "label": label,
        "summary": summary,
        "sample_count": len(details),
        "details": details,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    return out_path


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="法律知识库 RAG 评估脚本"
    )
    parser.add_argument("--kb-path", required=True, help="知识库根目录（含 books/）")
    parser.add_argument("--top-k", type=int, default=5, help="检索 Top-K")
    parser.add_argument("--full", action="store_true", help="启用 RAGAS 完整评估")
    parser.add_argument("--answers-file", default="", help="系统实际答案 JSONL（--full 必填）")
    parser.add_argument("--label", default="", help="结果标签（用于对比实验）")
    args = parser.parse_args()

    if args.full and not args.answers_file:
        parser.error("--full 必须同时提供 --answers-file（系统实际生成的答案 JSONL）")
    try:
        run_evaluation(
            kb_path=args.kb_path,
            top_k=args.top_k,
            full=args.full,
            label=args.label,
            answers_file=args.answers_file,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
