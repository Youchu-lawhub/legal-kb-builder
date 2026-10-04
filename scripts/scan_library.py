#!/usr/bin/env python3
"""
文献库快速扫描工具
用于在 Agentic 检索前快速了解文献库结构

使用方式（由 QoderWork 调用）：
    python scan_library.py <文献库目录>

输出：
    文献库结构报告（JSON格式），包含：
    - 文件列表及大小
    - 每个文件的标题和章节结构
    - 关键词频率统计
    - 建议的检索分组
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter


def scan_file_structure(library_dir: str) -> list[dict]:
    """扫描目录下所有 Markdown 文件的基本信息"""
    files = []
    lib_path = Path(library_dir)

    if not lib_path.exists():
        print(f"错误：目录 {library_dir} 不存在", file=sys.stderr)
        sys.exit(1)

    for md_file in sorted(lib_path.rglob("*.md")):
        stat = md_file.stat()
        line_count = 0
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
        except Exception:
            pass

        files.append({
            "path": str(md_file),
            "name": md_file.name,
            "relative_path": str(md_file.relative_to(lib_path)),
            "size_bytes": stat.st_size,
            "size_readable": _human_size(stat.st_size),
            "line_count": line_count,
        })

    return files


def extract_headings(filepath: str, max_lines: int = 200) -> list[dict]:
    """提取文件前 N 行中的标题结构"""
    headings = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    headings.append({
                        "level": level,
                        "title": title,
                        "line": i + 1,
                    })
    except Exception as e:
        headings.append({"error": str(e)})

    return headings


def count_legal_keywords(filepath: str) -> dict:
    """统计法律关键词出现频率"""
    # 常见法律关键词
    keywords = [
        "第.*?条", "合同", "侵权", "损害赔偿", "违约",
        "善意", "过错", "责任", "权利", "义务",
        "诉讼时效", "举证", "抗辩", "请求权", "构成要件",
        "法律效果", "但书", "例外", "推定", "证明责任",
        "判决", "裁定", "最高人民法院", "司法解释",
    ]

    counts = Counter()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            for kw in keywords:
                if ".*?" in kw or "." in kw:
                    matches = re.findall(kw, content)
                    if matches:
                        counts[kw] = len(matches)
                else:
                    count = content.count(kw)
                    if count > 0:
                        counts[kw] = count
    except Exception:
        pass

    return dict(counts.most_common(20))


def suggest_search_groups(files: list[dict]) -> list[dict]:
    """根据文件名和结构建议检索分组"""
    groups = {}

    for f in files:
        name = f["name"].lower()
        # 按编/篇/卷分组
        for pattern in [r'第.编', r'第.篇', r'第.卷', r'第.章',
                        r'part\s*\d', r'chapter\s*\d']:
            match = re.search(pattern, f["name"])
            if match:
                prefix = f["name"][:match.start()].strip("_- ")
                if prefix:
                    groups.setdefault(prefix, []).append(f["relative_path"])
                    break
        else:
            # 无法分组的归入"其他"
            groups.setdefault("未分组", []).append(f["relative_path"])

    return [{"group": k, "files": v, "count": len(v)} for k, v in groups.items()]


def generate_report(library_dir: str) -> dict:
    """生成完整的文献库扫描报告"""
    files = scan_file_structure(library_dir)

    if not files:
        return {
            "status": "empty",
            "message": f"目录 {library_dir} 下没有找到 .md 文件",
            "suggestion": "请先将 PDF 转换为 Markdown 文件放入该目录",
        }

    # 汇总信息
    total_size = sum(f["size_bytes"] for f in files)
    total_lines = sum(f["line_count"] for f in files)

    # 提取每个文件的标题
    for f in files:
        f["headings"] = extract_headings(f["path"], max_lines=100)
        f["keywords"] = count_legal_keywords(f["path"])

    # 搜索分组建议
    search_groups = suggest_search_groups(files)

    # v2: 检查知识集群缓存
    cluster_info = _check_cluster_store(library_dir)

    report = {
        "status": "ok",
        "library_path": str(Path(library_dir).resolve()),
        "summary": {
            "total_files": len(files),
            "total_size": _human_size(total_size),
            "total_lines": total_lines,
            "avg_lines_per_file": total_lines // max(len(files), 1),
        },
        "files": [{
            "name": f["name"],
            "relative_path": f["relative_path"],
            "size": f["size_readable"],
            "lines": f["line_count"],
            "top_headings": [h for h in f["headings"] if h.get("level", 99) <= 2][:5],
            "top_keywords": dict(list(f["keywords"].items())[:5]),
        } for f in files],
        "search_groups": search_groups,
        "cluster_info": cluster_info,
        "recommendations": _generate_recommendations(files, total_lines, cluster_info),
    }

    return report


def _generate_recommendations(files: list[dict], total_lines: int,
                               cluster_info: dict = None) -> list[str]:
    """生成检索建议"""
    recs = []

    if cluster_info and cluster_info.get("has_clusters"):
        stats = cluster_info.get("stats", {})
        hit_rate = stats.get("cache_hit_rate", 0)
        recs.append(
            f"已有 {stats.get('total_clusters', 0)} 个知识集群缓存"
            f"（命中率 {hit_rate:.0%}），FAST 模式可能足够"
        )

    if total_lines > 50000:
        recs.append("文献库较大（超过5万行），建议使用 DEEP 模式或 Task 工具并行搜索")

    large_files = [f for f in files if f["line_count"] > 3000]
    if large_files:
        names = ", ".join(f["name"] for f in large_files[:3])
        recs.append(f"以下文件较大，搜索时建议先 Grep 定位再 Read 精确段落：{names}")

    if len(files) > 10:
        recs.append("文件数量较多，建议先根据问题主题筛选相关文件再深入搜索")

    if not recs:
        recs.append("文献库规模适中，可直接进行全库搜索")

    return recs


def _check_cluster_store(library_dir: str) -> dict:
    """检查是否存在知识集群缓存"""
    cluster_path = Path(library_dir) / ".agentic_cache" / "_clusters"
    if not cluster_path.exists():
        return {"has_clusters": False}

    try:
        from knowledge_clusters import KnowledgeClusterStore
        store = KnowledgeClusterStore(cluster_path)
        stats = store.export_stats()
        hot = store.get_hot_clusters(5)
        return {
            "has_clusters": True,
            "stats": stats,
            "hot_topics": [{"query": c.query, "hits": c.hit_count} for c in hot],
        }
    except Exception:
        return {"has_clusters": False}


def _human_size(size_bytes: int) -> str:
    """将字节数转为人类可读格式"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scan_library.py <文献库目录路径>")
        print("示例: python scan_library.py ~/legal_docs/民法典评注/")
        sys.exit(1)

    library_dir = sys.argv[1]
    report = generate_report(library_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
