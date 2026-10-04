#!/usr/bin/env python3
"""
知识集群持久化存储

实现 Sirchmunk 的 Knowledge Cluster 概念：将检索结果跨会话持久化，
支持相似查询复用、热度跟踪和自动清理。

支持两种后端：
  - JSON（默认）：简单、易调试、适合中小规模
  - SQLite：更健壮、支持并发、适合大规模

使用方式：
    from knowledge_clusters import KnowledgeClusterStore

    store = KnowledgeClusterStore(Path("~/.cache/clusters"), backend='json')

    # 查找相似查询的缓存
    cached = store.find_similar("善意取得的构成要件")
    if cached:
        return cached.results

    # 存储新结果
    store.add_cluster(
        query="善意取得的构成要件",
        results=[...],
        source_files=["file1.md", "file2.md"],
        confidence=0.85,
    )
"""

import json
import math
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


@dataclass
class Cluster:
    """知识集群"""
    cluster_id: str = ""
    query: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = ""
    last_accessed: str = ""
    hit_count: int = 0
    tags: List[str] = field(default_factory=list)


class KnowledgeClusterStore:
    """
    知识集群持久化存储

    Args:
        store_path: 存储目录
        backend: 'json' 或 'sqlite'
    """

    def __init__(self, store_path: Path, backend: str = 'json'):
        self._store_path = Path(store_path)
        self._backend = backend
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._clusters: List[Cluster] = []
        self._load()

    # ─── 集群操作 ───

    def find_similar(self, query: str, threshold: float = 0.85) -> Optional[Cluster]:
        """
        查找与给定查询相似的已有集群。

        使用 jieba 分词 + Jaccard 相似度进行快速匹配。

        Args:
            query: 查询字符串
            threshold: 相似度阈值（0-1）

        Returns:
            最匹配的 Cluster，不存在时返回 None
        """
        if not self._clusters:
            return None

        query_tokens = self._tokenize(query)
        best_cluster = None
        best_score = 0.0

        for cluster in self._clusters:
            cluster_tokens = self._tokenize(cluster.query)
            score = self._jaccard_similarity(query_tokens, cluster_tokens)

            if score > best_score and score >= threshold:
                best_score = score
                best_cluster = cluster

        if best_cluster:
            best_cluster.last_accessed = datetime.now().isoformat()
            best_cluster.hit_count += 1
            self._save()

        return best_cluster

    def add_cluster(
        self,
        query: str,
        results: List[Dict[str, Any]],
        source_files: List[str],
        confidence: float,
        tags: List[str] = None,
    ) -> Cluster:
        """
        创建新的知识集群。

        Args:
            query: 查询字符串
            results: 检索结果
            source_files: 源文件列表
            confidence: 置信度
            tags: 标签（可选，自动生成）
        """
        now = datetime.now().isoformat()

        cluster = Cluster(
            cluster_id=str(uuid.uuid4())[:8],
            query=query,
            results=results,
            source_files=source_files,
            confidence=confidence,
            created_at=now,
            last_accessed=now,
            hit_count=0,
            tags=tags or self._auto_tag(query),
        )

        self._clusters.append(cluster)
        self._save()
        return cluster

    def update_cluster(self, cluster_id: str, new_results: List[Dict[str, Any]]) -> None:
        """合并新结果到已有集群"""
        for cluster in self._clusters:
            if cluster.cluster_id == cluster_id:
                # 合并结果（去重）
                existing_texts = {r.get('text', r.get('snippet', ''))[:100] for r in cluster.results}
                for r in new_results:
                    key = r.get('text', r.get('snippet', ''))[:100]
                    if key not in existing_texts:
                        cluster.results.append(r)
                        existing_texts.add(key)

                cluster.last_accessed = datetime.now().isoformat()
                self._save()
                return

    def get_hot_clusters(self, top_k: int = 10) -> List[Cluster]:
        """
        返回最热门的集群。

        热度 = hit_count * exp(-days_since_last_access / 30)
        """
        now = datetime.now()
        scored = []

        for cluster in self._clusters:
            try:
                last = datetime.fromisoformat(cluster.last_accessed)
                days_ago = (now - last).days
            except (ValueError, TypeError):
                days_ago = 90

            recency = math.exp(-days_ago / 30.0)
            hotness = cluster.hit_count * recency
            scored.append((hotness, cluster))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def prune_cold_clusters(self, max_age_days: int = 90, min_hits: int = 1) -> int:
        """
        移除冷集群（过期且未被频繁访问的）。

        Returns:
            移除的数量
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        original_count = len(self._clusters)

        self._clusters = [
            c for c in self._clusters
            if not self._is_cold(c, cutoff, min_hits)
        ]

        pruned = original_count - len(self._clusters)
        if pruned > 0:
            self._save()
        return pruned

    def export_stats(self) -> Dict[str, Any]:
        """导出集群统计"""
        if not self._clusters:
            return {"total_clusters": 0}

        total_hits = sum(c.hit_count for c in self._clusters)
        avg_confidence = sum(c.confidence for c in self._clusters) / len(self._clusters)

        # 热门标签
        tag_counts: Dict[str, int] = {}
        for c in self._clusters:
            for tag in c.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1 + c.hit_count

        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_clusters": len(self._clusters),
            "total_hits": total_hits,
            "avg_confidence": round(avg_confidence, 2),
            "avg_hits_per_cluster": round(total_hits / len(self._clusters), 1),
            "top_tags": [{"tag": t, "score": s} for t, s in top_tags],
        }

    # ─── 存储后端 ───

    def _load(self) -> None:
        if self._backend == 'sqlite':
            self._load_sqlite()
        else:
            self._load_json()

    def _save(self) -> None:
        if self._backend == 'sqlite':
            self._save_sqlite()
        else:
            self._save_json()

    def _load_json(self) -> None:
        path = self._store_path / 'clusters.json'
        if not path.exists():
            self._clusters = []
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._clusters = [Cluster(**c) for c in data.get('clusters', [])]
        except (json.JSONDecodeError, TypeError):
            self._clusters = []

    def _save_json(self) -> None:
        path = self._store_path / 'clusters.json'
        tmp_path = path.with_suffix('.json.tmp')

        data = {
            'version': 'v1',
            'updated_at': datetime.now().isoformat(),
            'cluster_count': len(self._clusters),
            'clusters': [asdict(c) for c in self._clusters],
        }

        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 原子性重命名
        os.replace(str(tmp_path), str(path))

    def _load_sqlite(self) -> None:
        db_path = self._store_path / 'clusters.db'
        if not db_path.exists():
            self._init_sqlite(db_path)
            self._clusters = []
            return

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM clusters").fetchall()
            self._clusters = []
            for row in rows:
                self._clusters.append(Cluster(
                    cluster_id=row['id'],
                    query=row['query'],
                    results=json.loads(row['results_json']),
                    source_files=json.loads(row['source_files_json']),
                    confidence=row['confidence'],
                    created_at=row['created_at'],
                    last_accessed=row['last_accessed'],
                    hit_count=row['hit_count'],
                    tags=json.loads(row['tags_json']),
                ))
        finally:
            conn.close()

    def _save_sqlite(self) -> None:
        db_path = self._store_path / 'clusters.db'
        if not db_path.exists():
            self._init_sqlite(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DELETE FROM clusters")
            for c in self._clusters:
                conn.execute(
                    """INSERT INTO clusters
                    (id, query, results_json, source_files_json,
                     confidence, created_at, last_accessed, hit_count, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (c.cluster_id, c.query, json.dumps(c.results, ensure_ascii=False),
                     json.dumps(c.source_files, ensure_ascii=False),
                     c.confidence, c.created_at, c.last_accessed,
                     c.hit_count, json.dumps(c.tags, ensure_ascii=False))
                )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _init_sqlite(db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                results_json TEXT,
                source_files_json TEXT,
                confidence REAL,
                created_at TEXT,
                last_accessed TEXT,
                hit_count INTEGER DEFAULT 0,
                tags_json TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ─── 工具方法 ───

    @staticmethod
    def _tokenize(text: str) -> set:
        if HAS_JIEBA:
            return set(w for w in jieba.cut(text) if len(w) >= 2)
        return set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', text))

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    @staticmethod
    def _is_cold(cluster: Cluster, cutoff: datetime, min_hits: int) -> bool:
        try:
            last = datetime.fromisoformat(cluster.last_accessed)
        except (ValueError, TypeError):
            return True
        return last < cutoff and cluster.hit_count < min_hits

    @staticmethod
    def _auto_tag(query: str) -> List[str]:
        if HAS_JIEBA:
            words = [w for w in jieba.cut(query) if len(w) >= 2]
            return words[:5]
        return re.findall(r'[\u4e00-\u9fff]{2,4}', query)[:5]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        store = KnowledgeClusterStore(Path("/tmp/_test_clusters"), backend='json')

        # 添加集群
        c1 = store.add_cluster(
            query="善意取得的构成要件",
            results=[{"text": "善意取得要求...", "file": "test.md"}],
            source_files=["test.md"],
            confidence=0.9,
        )
        print(f"Added cluster: {c1.cluster_id}")

        # 查找相似
        found = store.find_similar("善意取得构成要件包括哪些")
        print(f"Similar found: {found is not None}")

        # 统计
        stats = store.export_stats()
        print(f"Stats: {json.dumps(stats, ensure_ascii=False)}")

        # 清理测试文件
        import shutil
        shutil.rmtree("/tmp/_test_clusters")
        print("All tests completed.")
    else:
        print("用法: python knowledge_clusters.py --test")
