#!/usr/bin/env python3
"""
法条关联图谱模块（GraphRAG for Legal）

从法律文本中抽取带法律/版本身份的条文和案例，记录局部共现候选，
构建知识图谱，支持多跳查询，作为第四路检索融入 RRF 融合。

核心能力：
  1. 实体抽取：法条号（第X条）、法律概念、案例案号
  2. 关系记录：显式法律限定与局部共现；不推断适用、引用或解释关系
  3. 图谱构建：用 networkx 存储（可选依赖，未安装时降级为简单字典）
  4. 多跳查询：返回可定位的关联候选，供人工核验
  5. 社区检测：Louvain 聚类（可选，不生成社区报告或法律结论）

存储格式：
  - _graph.json: 图谱序列化（节点+边）
  - _graph_communities.json: 社区摘要（可选，需 networkx）

使用方式：
    from graph_rag import GraphRAG

    graph = GraphRAG()
    graph.build_from_kb(kb_path)          # 从知识库构建图谱
    results = graph.search("第311条")      # 多跳查询
    # -> [{"node": "...", "relation": "...", "depth": 1}, ...]
"""

import json
import hashlib
from collections import deque
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 可选依赖：networkx（用于图算法和社区检测）
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ─── 常量 ───

# 法条号正则（匹配"第X条"、"第X条第Y款"、"第X条之Y"）
ARTICLE_PATTERN = re.compile(
    r'第([一二三四五六七八九十百千万零\d]+)条'
    r'(?:之([一二三四五六七八九十\d]+))?'
    r'(?:第([一二三四五六七八九十\d]+)款)?'
    r'(?:第([一二三四五六七八九十\d]+)项)?'
)

# 案例案号正则（匹配"(2023)京01民终123号"等）
CASE_NUMBER_PATTERN = re.compile(
    r'[\(（](\d{4})[\)）]'
    r'([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]'
    r'[0-9]{2,4})'
    r'(民|刑|行|执|商|知|破)?'
    r'(初|终|再|抗|监)?'
    r'(\d+)号'
)

# 法律名称正则（匹配常见法律名）
LAW_NAME_PATTERN = re.compile(
    r'(中华人民共和国)?(民法典|刑法|民事诉讼法|刑事诉讼法|行政诉讼法|'
    r'合同法|物权法|侵权责任法|公司法|婚姻法|继承法|担保法|'
    r'劳动法|劳动合同法|商标法|专利法|著作权法|电子商务法|'
    r'个人信息保护法|数据安全法|网络安全法)'
)

# 中文数字转阿拉伯（统一使用 shared_utils 实现）
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from shared_utils import chinese_to_number as _chinese_to_number, atomic_write_json


def chinese_to_num(cn: str) -> int:
    """中文数字转阿拉伯数字（兼容包装：空/无法解析时返回 0）"""
    result = _chinese_to_number(cn)
    return result if result is not None else 0


class GraphRAG:
    """
    法条关联图谱检索器

    图谱结构：
      节点类型: article(法条), concept(法律概念), case(案例), law(法律名称)
      边类型: references(引用), applies(适用), explains(解释), mentions(提及)
    """

    def __init__(self):
        self._edge_keys = set()
        self._graph_data: Dict[str, Any] = {
            'nodes': {},   # {node_id: {type, label, ...}}
            'edges': [],   # [{source, target, relation, ...}]
        }
        self._nx_graph = None  # networkx 图（可选）

    # ─── 构建图谱 ───

    def build_from_kb(self, kb_path: Path, jurisdiction="未指定", law_versions=None):
        """构建有来源的轻量共现图。共现不推断适用、引用或解释关系。

        未指定版本的条文按来源文档隔离；显式 law_versions 可统一版本身份。
        未限定法律的条文按具体出现位置隔离，不能自动绑定文件中其他法律。
        """
        root = Path(kb_path)
        law_versions = law_versions or {}
        self._graph_data = {"schema_version": 2, "nodes": {}, "edges": []}
        self._edge_keys = set()
        files = sorted(f for f in root.glob("*.md") if not f.name.startswith("_"))
        for path in files:
            offset = 0
            document = hashlib.sha256(path.read_bytes()).hexdigest()
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(True), 1):
                entities = self._extract_entities(line)
                ids = []
                for article in entities["articles"]:
                    law = article.get("law")
                    version = article.get("version") or law_versions.get(law) or f"source:{path.name}:{document}"
                    identity = [jurisdiction, law or "未限定法律", version,
                                article["num"], article.get("suffix"), article.get("clause"), article.get("item")]
                    if not law:
                        identity.append(offset + article["start"])
                    nid = "article:" + json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
                    origin = {"file": path.name, "line": line_no,
                              "start_char": offset + article["start"],
                              "end_char": offset + article["end"]}
                    node = self._graph_data["nodes"].setdefault(nid, {
                        "type": "article", "label": (law or "未限定法律") + article["full"],
                        "article_num": article["num"], "suffix": article.get("suffix"),
                        "clause": article.get("clause"), "item": article.get("item"),
                        "law": law, "version": version, "jurisdiction": jurisdiction,
                        "mentions": 0, "sources": [],
                    })
                    node["mentions"] += 1
                    node["sources"].append(origin)
                    ids.append(nid)
                    if law:
                        law_id = "law:" + json.dumps([jurisdiction, law, version], ensure_ascii=False)
                        self._graph_data["nodes"].setdefault(law_id, {
                            "type": "law", "label": law, "law": law, "version": version,
                            "mentions": 1, "sources": [origin],
                        })
                        self._add_edge(nid, law_id, "qualified_in_text", path.name, line_no)
                for case in entities["cases"]:
                    nid = f"case:{case['full']}"
                    origin = {"file": path.name, "line": line_no, "start_char": offset + case["start"]}
                    node = self._graph_data["nodes"].setdefault(nid, {
                        "type": "case", "label": case["full"], "mentions": 0, "sources": [],
                    })
                    node["mentions"] += 1
                    node["sources"].append(origin)
                    ids.append(nid)
                ids = list(dict.fromkeys(ids))
                # 有界局部共现；长 OCR 行不构造全连接图。
                if len(line) <= 2000 and len(ids) <= 20:
                    for i, first in enumerate(ids):
                        for second in ids[i + 1:]:
                            self._add_edge(first, second, "co_occurs", path.name, line_no)
                offset += len(line)
        if HAS_NX:
            self._build_nx_graph()
        stats = {"nodes": self.node_count, "edges": self.edge_count, "files": len(files)}
        print(f"[GraphRAG] 共现图构建完成: {stats}")
        return stats

    def save(self, kb_path: Path) -> Path:
        """保存图谱到知识库目录"""
        kb_path = Path(kb_path)
        graph_path = kb_path / '_graph.json'
        atomic_write_json(graph_path, self._graph_data)
        print(f"[GraphRAG] 图谱已保存: {graph_path}")
        return graph_path

    def load(self, kb_path: Path) -> bool:
        """从知识库目录加载图谱"""
        kb_path = Path(kb_path)
        graph_path = kb_path / '_graph.json'
        if not graph_path.exists():
            return False
        with open(graph_path, 'r', encoding='utf-8') as f:
            self._graph_data = json.load(f)
        if self._graph_data.get("schema_version") != 2:
            raise ValueError("旧图谱缺少法律/版本身份，请重新 build，不能安全迁移")
        if HAS_NX:
            self._build_nx_graph()
        print(f"[GraphRAG] 图谱已加载: {len(self._graph_data['nodes'])} 节点")
        return True

    # ─── 查询 ───

    def search(
        self,
        query: str,
        max_depth: int = 2,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        图谱多跳查询：从查询中抽取法条/概念，做多跳邻居检索。

        Args:
            query: 查询字符串（如"第311条"或"善意取得"）
            max_depth: 最大跳数（默认 2）
            top_k: 返回前 K 个结果

        Returns:
            [{"node_id", "label", "type", "relation", "depth", "path"}, ...]
        """
        if top_k < 1 or max_depth < 0:
            raise ValueError("top_k 必须为正数，max_depth 不能为负数")
        seeds = self._extract_seeds(query)
        if not seeds:
            return []
        query_laws = set(self._extract_entities(query)["laws"])
        adjacency = {}
        for edge in self._graph_data["edges"]:
            for first, second in [(edge["source"], edge["target"]), (edge["target"], edge["source"])]:
                adjacency.setdefault(first, []).append((second, edge))
        visited, results = set(), []
        queue = deque((seed, 0, []) for seed in seeds)
        while queue:
            nid, depth, path = queue.popleft()
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            node = self._graph_data["nodes"].get(nid, {})
            if query_laws and node.get("law") and node["law"] not in query_laws:
                continue
            results.append({"node_id": nid, "label": node.get("label", ""),
                            "type": node.get("type", ""), "mentions": node.get("mentions", 0),
                            "depth": depth, "path": path, "sources": node.get("sources", []),
                            "relation": "matched" if not path else path[-1]["relation"]})
            if depth == max_depth or (depth > 0 and node.get("type") == "law"):
                continue
            for neighbor, edge in adjacency.get(nid, []):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1, path + [{
                        "node": nid, "relation": edge["relation"],
                        "file": edge["source_file"], "line": edge.get("line"),
                    }]))
        results.sort(key=lambda item: (item["depth"], -item["mentions"], item["node_id"]))
        return results[:top_k]

    def get_related_articles(self, article: str) -> List[Dict]:
        """获取与指定法条相关的所有节点"""
        return self.search(article, max_depth=2, top_k=20)

    def get_article_cases(self, article: str) -> List[Dict]:
        """获取与某法条局部共现的案例候选；不表示该案例适用了条文。"""
        results = self.search(article, max_depth=1, top_k=20)
        return [r for r in results if r.get('type') == 'case']

    # ─── 社区检测（可选，需 networkx） ───

    def detect_communities(self) -> Dict[str, List[str]]:
        """
        社区检测：发现法条主题聚类。

        需 networkx。用 Louvain 算法（nx.community.louvain_communities）。
        若不可用降级为连通分量。
        """
        if not HAS_NX or self._nx_graph is None:
            return {}

        try:
            # 尝试 Louvain（networkx >= 3.0）
            communities = nx.community.louvain_communities(self._nx_graph)
        except (AttributeError, Exception):
            # 降级：连通分量
            communities = list(nx.connected_components(self._nx_graph))

        result = {}
        for i, comm in enumerate(communities):
            nodes = list(comm)
            if len(nodes) < 2:
                continue
            result[f"community_{i}"] = {
                'size': len(nodes),
                'nodes': nodes[:20],  # 只取前20个做摘要
                'labels': [
                    self._graph_data['nodes'].get(n, {}).get('label', n)
                    for n in nodes[:10]
                ],
            }
        return result

    # ─── 内部方法 ───

    def _extract_entities(self, text: str):
        articles, cases = [], []
        for match in ARTICLE_PATTERN.finditer(text):
            # 只接受同一语句中紧邻条号的法律名，可含显式版本。
            before = text[:match.start()]
            qualified = re.search(
                r"(?:《)?(" + LAW_NAME_PATTERN.pattern + r")(?:》)?"
                r"(?:[（(]([^()（）\n]{1,40})[）)])?\s*$", before)
            law = version = None
            if qualified:
                names = list(LAW_NAME_PATTERN.finditer(qualified.group(1)))
                law = names[-1].group(2) if names else None
                version = qualified.group(4)  # 外层组1 + 内层两组 + 版本组
            articles.append({"full": match.group(0), "num": chinese_to_num(match.group(1)),
                             "suffix": chinese_to_num(match.group(2)) if match.group(2) else None,
                             "clause": chinese_to_num(match.group(3)) if match.group(3) else None,
                             "item": chinese_to_num(match.group(4)) if match.group(4) else None,
                             "law": law, "version": version,
                             "start": match.start(), "end": match.end()})
        for match in CASE_NUMBER_PATTERN.finditer(text):
            cases.append({"full": match.group(0).replace("（", "(").replace("）", ")"),
                          "year": match.group(1), "court": match.group(2),
                          "type": match.group(3) or "", "stage": match.group(4) or "",
                          "number": match.group(5), "start": match.start()})
        laws = list(dict.fromkeys(match.group(2) for match in LAW_NAME_PATTERN.finditer(text)))
        return {"articles": articles, "cases": cases, "laws": laws}

    def _extract_seeds(self, query: str):
        entities = self._extract_entities(query)
        seeds = []
        for article in entities["articles"]:
            for nid, node in self._graph_data["nodes"].items():
                if node.get("type") != "article" or node.get("article_num") != article["num"]:
                    continue
                if article["law"] and node.get("law") != article["law"]:
                    continue
                if article["version"] and node.get("version") != article["version"]:
                    continue
                if any(article.get(key) is not None and node.get(key) != article[key]
                       for key in ("suffix", "clause", "item")):
                    continue
                seeds.append(nid)
        for case in entities["cases"]:
            nid = f"case:{case['full']}"
            if nid in self._graph_data["nodes"]:
                seeds.append(nid)
        if not entities["articles"]:
            seeds.extend(nid for nid, node in self._graph_data["nodes"].items()
                         if node.get("type") == "law" and node.get("law") in entities["laws"])
        return list(dict.fromkeys(seeds))

    def _add_edge(self, source, target, relation, source_file="", line=None):
        if source == target:
            return
        key = (min(source, target), max(source, target), relation, source_file, line)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self._graph_data["edges"].append({"source": source, "target": target,
                                        "relation": relation, "source_file": source_file,
                                        "line": line})

    def _build_nx_graph(self) -> None:
        """构建 networkx 图"""
        if not HAS_NX:
            return
        self._nx_graph = nx.Graph()
        for nid in self._graph_data['nodes']:
            self._nx_graph.add_node(nid)
        for edge in self._graph_data['edges']:
            self._nx_graph.add_edge(edge['source'], edge['target'])

    @property
    def node_count(self) -> int:
        return len(self._graph_data['nodes'])

    @property
    def edge_count(self) -> int:
        return len(self._graph_data['edges'])


# ─── CLI ───

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python graph_rag.py build <kb_path>          # 构建图谱")
        print("  python graph_rag.py search <kb_path> <query> # 多跳查询")
        print("  python graph_rag.py communities <kb_path>    # 社区检测")
        print("  python graph_rag.py --test                   # 自测")
        sys.exit(0)

    if sys.argv[1] == '--test':
        print(f"networkx: {'yes' if HAS_NX else 'no'}")
        # 测试实体抽取
        sample = "依据民法典第311条规定，(2023)京01民终123号案例中..."
        g = GraphRAG()
        entities = g._extract_entities(sample)
        print(f"法条: {[a['full'] for a in entities['articles']]}")
        print(f"案例: {[c['full'] for c in entities['cases']]}")
        print(f"法律: {entities['laws']}")
        # 测试中文数字转换
        print(f"'三百一十一' -> {chinese_to_num('三百一十一')}")
        print(f"'十' -> {chinese_to_num('十')}")
        print(f"'二十五' -> {chinese_to_num('二十五')}")
        print("自测完成")
        sys.exit(0)

    command = sys.argv[1]
    if command == 'build' and len(sys.argv) >= 3:
        g = GraphRAG()
        stats = g.build_from_kb(Path(sys.argv[2]))
        g.save(Path(sys.argv[2]))
        print(f"节点: {stats['nodes']}, 边: {stats['edges']}, 文件: {stats['files']}")

    elif command == 'search' and len(sys.argv) >= 4:
        g = GraphRAG()
        if not g.load(Path(sys.argv[2])):
            print("图谱不存在，请先运行 build")
            sys.exit(1)
        results = g.search(sys.argv[3])
        for r in results:
            print(f"  [d{r['depth']}] {r['type']}: {r['label']} (mentions={r['mentions']})")

    elif command == 'communities' and len(sys.argv) >= 3:
        g = GraphRAG()
        if not g.load(Path(sys.argv[2])):
            print("图谱不存在，请先运行 build")
            sys.exit(1)
        communities = g.detect_communities()
        print(f"发现 {len(communities)} 个社区:")
        for cid, info in communities.items():
            print(f"\n  {cid} (size={info['size']}):")
            for label in info['labels']:
                print(f"    - {label}")

    else:
        print("未知命令")
