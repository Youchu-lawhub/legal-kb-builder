#!/usr/bin/env python3
"""
问答集知识库主入口（流水线 D — qa_kb）

适用于：FAQ / 业务咨询问答 / 培训问答 / 客服话术 / 法律咨询问答集 等问答形态语料。

核心检索策略："问题匹配为主、答案召回为辅"
  - 路径 A：问题 BM25 匹配（用户提问 ↔ 库内问题关键词）
  - 路径 B：问题向量匹配（用户提问 ↔ 库内问题语义）
  - 路径 C：答案 BM25 召回（用户提问 ↔ 库内答案关键词）
  - 路径 D：答案向量召回（用户提问 ↔ 库内答案语义）
  - 路径 E：分类/标签精确匹配
  - RRF 融合 → 排序结果

提供 init / add / batch / search / stats / rebuild 命令。

存储结构：
    qa_kb/
    ├── config.yaml
    ├── qa_pairs/                 # 每个问答对一个 JSON
    │   ├── qa_0001.json
    │   └── ...
    ├── indices/
    │   ├── metadata.json         # 结构化元数据索引
    │   ├── q_index/              # 问题专用索引（核心：用户提问 ↔ 库内问题）
    │   │   ├── _bm25_corpus.json
    │   │   ├── _vectors.faiss
    │   │   └── _chunks.json
    │   └── a_index/              # 答案召回索引（辅助：用户提问 ↔ 库内答案）
    │       ├── _bm25_corpus.json
    │       ├── _vectors.faiss
    │       └── _chunks.json
    └── cache/
"""

import argparse
import json
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from qa_parser import QAParser, QAPair
from shared_utils import file_lock, atomic_write_json, active_record_index, build_record_index_bundle, metadata_revision, file_signature


class QAKnowledgeBase:
    """问答集知识库"""

    def __init__(self, kb_path: str = None):
        self.kb_path = Path(kb_path) if kb_path else Path.cwd() / "qa_kb"
        self.parser = QAParser()
        self._meta: Optional[Dict] = None
        self.build_warnings = []
        self.build_warnings = []

    # ─── 初始化 ───

    def init(self, name: str) -> None:
        """初始化知识库目录结构"""
        for d in ["qa_pairs", "indices", "cache"]:
            (self.kb_path / d).mkdir(parents=True, exist_ok=True)

        config = {
            "name": name,
            "kb_type": "qa_kb",
            "created_at": datetime.now().isoformat(),
            "qa_count": 0,
        }
        self._save_config(config)

        metadata = {"qa_pairs": [], "stats": {}, "categories": {}, "tags": {}}
        if not (self.kb_path / "indices" / "metadata.json").exists():
            self._save_metadata(metadata)

        print(f"问答知识库 '{name}' 初始化成功: {self.kb_path}")

    # ─── 添加问答对 ───

    def add_file(self, file_path: str, default_category: str = "",
                 default_source: str = "", rebuild: bool = True) -> int:
        """
        从文件解析并添加问答对。

        Args:
            file_path: 问答集文件路径（md/txt/json）
            default_category: 默认分类（当问答未自带分类时使用）
            default_source: 默认来源

        Returns:
            添加的问答对数量
        """
        path = Path(file_path)
        if not path.exists():
            print(f"文件不存在: {file_path}")
            return 0

        pairs = self.parser.parse_file(str(path))
        if not pairs:
            print(f"未从 {path.name} 中解析出问答对")
            return 0

        count = 0
        for pair in pairs:
            # 应用默认值
            if not pair.category and default_category:
                pair.category = default_category
            if not pair.source and default_source:
                pair.source = default_source
            if self._add_pair(pair, source_file=path.name):
                count += 1

        if rebuild and count:
            self.rebuild_indices()
        print(f"从 {path.name} 新增 {count} 个问答对")
        return count

    def add_pair(self, question: str, answer: str, category: str = "",
                 tags: List[str] = None, source: str = "") -> None:
        """手动添加单个问答对"""
        pair = QAPair(
            question=question,
            answer=answer,
            category=category,
            tags=tags or [],
            source=source,
            raw_format="manual",
        )
        if self._add_pair(pair, source_file="manual"):
            self.rebuild_indices()

    def add_batch(self, directory: str, default_category: str = "") -> int:
        """批量导入目录下所有问答文件"""
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"目录不存在: {directory}")
            return 0

        files = (list(dir_path.glob("*.md")) + list(dir_path.glob("*.txt"))
                 + list(dir_path.glob("*.json")))
        print(f"找到 {len(files)} 个文件")

        total = 0
        for f in files:
            total += self.add_file(str(f), default_category=default_category, rebuild=False)
            print()

        if total > 0:
            self.rebuild_indices()
        return total

    def _add_pair(self, pair: QAPair, source_file: str = "") -> bool:
        """按内容哈希幂等追加；记录先原子写入，再原子发布元数据清单。"""
        record = pair.to_dict()
        identity = {key: record.get(key) for key in ("question", "answer", "category", "tags", "source")}
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()[:20]
        qa_id = f"qa_{digest}"
        with file_lock(self.kb_path / ".write.lock"):
            self._meta = None
            metadata = self._load_metadata()
            if any(item.get("id") == qa_id for item in metadata.get("qa_pairs", [])):
                return False
            record.update({"id": qa_id, "source_file": source_file,
                           "added_at": datetime.now().isoformat()})
            atomic_write_json(self.kb_path / "qa_pairs" / f"{qa_id}.json", record)
            entry = {"id": qa_id, "question": pair.question, "category": pair.category,
                     "tags": pair.tags, "source": pair.source, "file": f"{qa_id}.json"}
            metadata.setdefault("qa_pairs", []).append(entry)
            metadata["stats"] = {"total": len(metadata["qa_pairs"])}
            cats, tags = {}, {}
            for item in metadata["qa_pairs"]:
                if item.get("category"):
                    cats[item["category"]] = cats.get(item["category"], 0) + 1
                for tag in item.get("tags", []):
                    tags[tag] = tags.get(tag, 0) + 1
            metadata["categories"], metadata["tags"] = cats, tags
            self._save_metadata(metadata)
            return True

    # ─── 重建索引 ───

    def rebuild_indices(self) -> None:
        """构建不可变索引版本，所有查询通过版本清单读取。"""
        with file_lock(self.kb_path / ".write.lock"):
            self._meta = None
            metadata = self._load_metadata()
            source_groups = {"q": [], "a": []}
            for entry in metadata.get("qa_pairs", []):
                path = self.kb_path / "qa_pairs" / entry.get("file", "")
                if not path.is_file():
                    raise FileNotFoundError(f"问答记录缺失: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                qa_id = data["id"]
                source_groups["q"].append((qa_id, data["question"]))
                answer = data.get("answer", "")
                tags = data.get("tags", [])
                source_groups["a"].append((qa_id, answer + "\n" + " ".join(tags)))
            self.build_warnings = build_record_index_bundle(self.kb_path, source_groups, metadata)
        print(f"索引版本已发布 ({len(metadata.get('qa_pairs', []))} 条)")

    def _active_indices(self, metadata):
        return active_record_index(self.kb_path, metadata) or (self.kb_path / "indices")

    # ─── 检索 ───

    def search(
        self,
        query: str,
        category: str = None,
        tag: str = None,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        检索问答对。

        策略："问题匹配为主、答案召回为辅"，RRF 融合多路结果。

        Args:
            query: 用户提问
            category: 限定分类（可选）
            tag: 限定标签（可选）
            mode: 检索索引范围——hybrid(双索引) | question(仅问题索引) | answer(仅答案索引)
                  bm25 与 semantic 等同 hybrid（保留向后兼容）
            top_k: 返回前 K 条

        Returns:
            [{"id","question","answer","category","tags","source","score","matched_via"}, ...]
        """
        metadata = self._load_metadata()
        qa_entries = metadata.get("qa_pairs", [])

        if not qa_entries:
            return []

        # 建立 id → entry 映射
        by_id = {e["id"]: e for e in qa_entries}

        # 分类/标签过滤
        if category or tag:
            filtered = []
            for e in qa_entries:
                if category and category not in (e.get("category") or ""):
                    continue
                if tag and tag not in (e.get("tags") or []):
                    continue
                filtered.append(e)
            qa_entries = filtered
            if not qa_entries:
                return []

        indices_dir = self._active_indices(metadata)

        # ── 各路检索结果：{qa_id: score} ──
        q_bm25_hits: Dict[str, float] = {}
        q_vec_hits: Dict[str, float] = {}
        a_bm25_hits: Dict[str, float] = {}
        a_vec_hits: Dict[str, float] = {}

        def _search_index(prefix: str, q: str, k: int) -> Dict[str, float]:
            """搜索一个索引（BM25 + 向量），返回 {qa_id: max_score}"""
            idx_dir = indices_dir / f"{prefix}_index"
            hits: Dict[str, float] = {}

            if (idx_dir / "_bm25_corpus.json").exists():
                try:
                    from bm25_searcher import BM25Searcher
                    bm25 = BM25Searcher()
                    for chunk_id, score in bm25.search(idx_dir, q, top_k=k):
                        stem = Path(chunk_id.split("#", 1)[0].split(":", 1)[0]).stem
                        hits[stem] = max(hits.get(stem, 0.0), float(score))
                except (ImportError, Exception) as e:
                    print(f"[WARN] {prefix}_index BM25 检索异常，已跳过: {e}", file=sys.stderr)
                    pass

            from shared_utils import vector_index_exists
            if vector_index_exists(idx_dir):
                try:
                    from embedding_manager import EmbeddingManager
                    em = EmbeddingManager()
                    for chunk_id, score in em.search(idx_dir, q, top_k=k):
                        stem = Path(chunk_id.split("#", 1)[0].split(":", 1)[0]).stem
                        hits[stem] = max(hits.get(stem, 0.0), float(score))
                except (ImportError, Exception) as e:
                    print(f"[WARN] {prefix}_index 向量检索异常，已跳过: {e}", file=sys.stderr)
                    pass

            return hits

        # 搜索问题索引
        if mode in ("hybrid", "question", "bm25", "semantic"):
            q_hits = _search_index("q", query, top_k * 3)
            for k, v in q_hits.items():
                q_bm25_hits[k] = v  # 问题匹配分

        # 搜索答案索引
        if mode in ("hybrid", "answer", "bm25", "semantic"):
            a_hits = _search_index("a", query, top_k * 3)
            for k, v in a_hits.items():
                a_bm25_hits[k] = v  # 答案召回分

        # ── RRF 融合（问题匹配权重更高）──
        # 权重来自 qa-fields.yaml fusion.weights
        fusion_cfg = self._load_field_config().get("fusion", {})
        weights = fusion_cfg.get("weights", {
            "question_semantic": 1.5,
            "question_bm25": 1.2,
            "answer_semantic": 0.8,
            "answer_bm25": 0.6,
        })

        def _to_rank(hits: Dict[str, float]) -> Dict[str, int]:
            ordered = sorted(hits.items(), key=lambda x: x[1], reverse=True)
            return {k: i + 1 for i, (k, _) in enumerate(ordered)}

        rrf_k = fusion_cfg.get("rrf_k", 60)
        fused: Dict[str, float] = {}

        # 问题匹配（合并 BM25 + 向量为一路，取问题权重）
        q_combined = {}
        for k, v in q_bm25_hits.items():
            q_combined[k] = q_combined.get(k, 0) + v
        w_q = weights.get("question_semantic", 1.5) + weights.get("question_bm25", 1.2)
        for k, rank in _to_rank(q_combined).items():
            fused[k] = fused.get(k, 0.0) + w_q / (rrf_k + rank)

        # 答案召回
        w_a = weights.get("answer_semantic", 0.8) + weights.get("answer_bm25", 0.6)
        for k, rank in _to_rank(a_bm25_hits).items():
            fused[k] = fused.get(k, 0.0) + w_a / (rrf_k + rank)

        # ── 兜底：grep ──
        if not fused:
            for e in qa_entries:
                qa_file = self.kb_path / "qa_pairs" / e.get("file", "")
                if not qa_file.exists():
                    continue
                with open(qa_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if query in data.get("question", "") or query in data.get("answer", ""):
                    fused[e["id"]] = 0.3

        # ── 组装结果 ──
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for qa_id, score in ordered[:top_k]:
            entry = by_id.get(qa_id)
            if not entry:
                continue
            # 读取完整问答对
            qa_file = self.kb_path / "qa_pairs" / entry.get("file", f"{qa_id}.json")
            full_data = {}
            if qa_file.exists():
                with open(qa_file, "r", encoding="utf-8") as f:
                    full_data = json.load(f)

            matched_via = []
            if qa_id in q_combined:
                matched_via.append("question")
            if qa_id in a_bm25_hits:
                matched_via.append("answer")

            results.append({
                "id": qa_id,
                "question": full_data.get("question", entry.get("question", "")),
                "answer": full_data.get("answer", ""),
                "category": full_data.get("category", entry.get("category", "")),
                "tags": full_data.get("tags", entry.get("tags", [])),
                "source": full_data.get("source", ""),
                "score": round(score, 4),
                "matched_via": matched_via,
            })

        return results

    def recommend_similar(self, question: str, top_k: int = 5) -> List[Dict]:
        """推荐相似问题（只走问题索引）"""
        return self.search(question, mode="question", top_k=top_k)

    # ─── 统计 ───

    def stats(self) -> Dict[str, Any]:
        """知识库统计"""
        metadata = self._load_metadata()
        if "error" in metadata:
            return metadata

        stats = metadata.get("stats", {})
        manifest = self.kb_path / "indices" / "_index_manifest.json"
        try:
            current_manifest = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        except (OSError, json.JSONDecodeError):
            current_manifest = {}
        current = current_manifest.get("metadata_revision") == metadata_revision(metadata)
        return {
            "total_qa": stats.get("total", 0),
            "categories": metadata.get("categories", {}),
            "top_tags": dict(
                sorted(metadata.get("tags", {}).items(), key=lambda x: x[1], reverse=True)[:20]
            ),
            "has_indices": current and bool(active_record_index(self.kb_path, metadata)),
            "index_warnings": current_manifest.get("warnings", []) if current else ["索引与当前语料不一致"],
        }

    # ─── 内部工具 ───

    def _load_config(self) -> Dict:
        cfg_path = self.kb_path / "config.yaml"
        if not cfg_path.exists():
            return {}
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _save_config(self, config: Dict) -> None:
        import yaml
        with open(self.kb_path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

    def _load_metadata(self) -> Dict:
        meta_path = self.kb_path / "indices" / "metadata.json"
        if not meta_path.exists():
            return {"qa_pairs": [], "stats": {}, "categories": {}, "tags": {}}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
                return self._meta
        except json.JSONDecodeError as exc:
            raise ValueError(f"问答元数据损坏: {meta_path}: {exc}") from exc
        except FileNotFoundError:
            return {"qa_pairs": [], "stats": {}, "categories": {}, "tags": {}}

    def _save_metadata(self, metadata: Dict) -> None:
        meta_path = self.kb_path / "indices" / "metadata.json"
        atomic_write_json(meta_path, metadata)
        self._meta = metadata

    def _load_field_config(self) -> Dict:
        """加载 qa-fields.yaml 配置"""
        from shared_utils import load_yaml_config
        return load_yaml_config(
            str(Path(__file__).parent.parent / "assets" / "qa-fields.yaml")
        )


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(description="问答集知识库（流水线 D）")
    parser.add_argument("--kb", default=None, help="知识库路径")
    subparsers = parser.add_subparsers(dest="command")

    init_p = subparsers.add_parser("init", help="初始化知识库")
    init_p.add_argument("--name", required=True, help="知识库名称")

    add_p = subparsers.add_parser("add", help="从文件导入问答对")
    add_p.add_argument("file", help="问答集文件路径")
    add_p.add_argument("--category", default="", help="默认分类")

    addpair_p = subparsers.add_parser("add-pair", help="手动添加单个问答对")
    addpair_p.add_argument("--question", required=True, help="问题")
    addpair_p.add_argument("--answer", required=True, help="答案")
    addpair_p.add_argument("--category", default="", help="分类")
    addpair_p.add_argument("--tags", default="", help="标签（逗号分隔）")
    addpair_p.add_argument("--source", default="", help="来源")

    batch_p = subparsers.add_parser("batch", help="批量导入目录")
    batch_p.add_argument("directory", help="问答文件目录")
    batch_p.add_argument("--category", default="", help="默认分类")

    search_p = subparsers.add_parser("search", help="检索问答")
    search_p.add_argument("--query", help="自然语言提问")
    search_p.add_argument("--category", help="限定分类")
    search_p.add_argument("--tag", help="限定标签")
    search_p.add_argument("--mode", default="hybrid",
                          choices=["hybrid", "question", "answer", "bm25", "semantic"])
    search_p.add_argument("--top-k", type=int, default=10)

    rec_p = subparsers.add_parser("recommend", help="推荐相似问题")
    rec_p.add_argument("question", help="问题文本")
    rec_p.add_argument("--top-k", type=int, default=5)

    subparsers.add_parser("stats", help="统计")
    subparsers.add_parser("rebuild", help="重建索引")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    kb = QAKnowledgeBase(args.kb)

    if args.command == "init":
        kb.init(args.name)
    elif args.command == "add":
        kb.add_file(args.file, default_category=args.category)
    elif args.command == "add-pair":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        kb.add_pair(args.question, args.answer, args.category, tags, args.source)
        print("已添加 1 个问答对")
    elif args.command == "batch":
        kb.add_batch(args.directory, default_category=args.category)
    elif args.command == "search":
        results = kb.search(args.query, category=args.category, tag=args.tag,
                            mode=args.mode, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.command == "recommend":
        results = kb.recommend_similar(args.question, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.command == "stats":
        print(json.dumps(kb.stats(), ensure_ascii=False, indent=2))
    elif args.command == "rebuild":
        kb.rebuild_indices()


if __name__ == "__main__":
    main()
