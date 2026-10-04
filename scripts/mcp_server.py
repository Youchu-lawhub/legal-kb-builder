#!/usr/bin/env python3
"""
legal-kb-builder MCP Server
────────────────────────────
把 legal-kb-builder 的三路融合检索（BM25 + 向量 + 结构索引）以 MCP 工具的形式暴露给
任何支持 Model Context Protocol 的客户端（Claude Desktop / Cursor / Cline / Continue / …）。

## 环境要求

    pip install mcp  # 官方 MCP Python SDK（>=1.0）

配套依赖仍来自 legal-kb-builder/requirements.txt。

## 启动方式

    # 单库
    KB_PATH=/path/to/your/kb  python3 scripts/mcp_server.py

    # 或用 --kb 覆盖
    python3 scripts/mcp_server.py --kb /path/to/your/kb

    # 多库聚合（第一个作为默认）
    KB_PATHS="/path/to/kb1:/path/to/kb2"  python3 scripts/mcp_server.py

## 暴露的工具

1. search_by_article  —— 按法条编号精确检索（对应 legal_kb search --article）
2. search_by_query    —— 自然语言 / 关键词检索（对应 legal_kb search --query）
3. search_by_scholar  —— 学者观点检索（对应 legal_kb search --scholar --topic）
4. list_books         —— 列出所有已入库书籍及元信息
5. get_book_index     —— 读取某本书的 _知识库索引.json（供上层做二次分析）

## Claude Desktop 配置示例

```json
{
  "mcpServers": {
    "legal-kb": {
      "command": "python3",
      "args": ["/absolute/path/to/legal-kb-builder/scripts/mcp_server.py"],
      "env": { "KB_PATH": "/absolute/path/to/your/kb" }
    }
  }
}
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# 让 scripts/ 目录内的其他模块可被直接 import
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ─── MCP SDK ───
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.stderr.write(
        "❌ 需要安装 mcp SDK：pip install mcp\n"
        "   参考：https://modelcontextprotocol.io/quickstart/server\n"
    )
    sys.exit(2)

# ─── legal-kb-builder 内部模块 ───
try:
    from legal_kb import LegalKnowledgeBase
except ImportError as e:
    sys.stderr.write(f"❌ 无法导入 legal_kb 模块：{e}\n")
    sys.exit(2)


# ─── 参数解析 ───

def _parse_kb_paths() -> list[Path]:
    """
    读取知识库路径。优先级：
    1. --kb / --kb-paths 命令行参数
    2. KB_PATH 环境变量（单库）
    3. KB_PATHS 环境变量（冒号分隔，多库）
    4. 当前工作目录下的 legal_kb/
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kb", dest="kb", help="单个知识库路径")
    parser.add_argument("--kb-paths", dest="kb_paths",
                        help="多个知识库路径，冒号分隔（第一个为默认）")
    args, _ = parser.parse_known_args()

    if args.kb_paths:
        raw = args.kb_paths
    elif args.kb:
        raw = args.kb
    elif os.environ.get("KB_PATHS"):
        raw = os.environ["KB_PATHS"]
    elif os.environ.get("KB_PATH"):
        raw = os.environ["KB_PATH"]
    else:
        raw = str(Path.cwd() / "legal_kb")

    paths = [Path(p).expanduser().resolve() for p in raw.split(":") if p.strip()]
    if not paths:
        sys.stderr.write("❌ 未指定任何知识库路径\n")
        sys.exit(2)

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            sys.stderr.write(f"⚠️  知识库路径不存在：{p}\n")

    return paths


KB_PATHS = _parse_kb_paths()
KB_REGISTRY: dict[str, LegalKnowledgeBase] = {}
for _p in KB_PATHS:
    if _p.exists():
        try:
            KB_REGISTRY[_p.name] = LegalKnowledgeBase(str(_p))
        except Exception as e:
            sys.stderr.write(f"⚠️  加载知识库 {_p} 失败：{e}\n")

if not KB_REGISTRY:
    sys.stderr.write("❌ 没有可用的知识库\n")
    sys.exit(2)

DEFAULT_KB_NAME = next(iter(KB_REGISTRY))


# ─── 问答知识库加载（可选） ───
# 通过 QA_KB_PATH 环境变量或 --qa-kb 参数指定
QA_KB_PATH: Path | None = None
_qa_kb_instance: Any = None

_qa_parser = argparse.ArgumentParser(add_help=False)
_qa_parser.add_argument("--qa-kb", dest="qa_kb", help="问答知识库路径")
_qa_args, _ = _qa_parser.parse_known_args()

if _qa_args.qa_kb:
    QA_KB_PATH = Path(_qa_args.qa_kb)
elif os.environ.get("QA_KB_PATH"):
    QA_KB_PATH = Path(os.environ["QA_KB_PATH"])

# 延迟加载 QA KB（首次使用时才 import）
def _get_qa_kb() -> Any:
    global _qa_kb_instance
    if _qa_kb_instance is None and QA_KB_PATH and QA_KB_PATH.exists():
        try:
            from qa_kb import QAKnowledgeBase
            _qa_kb_instance = QAKnowledgeBase(str(QA_KB_PATH))
        except Exception as e:
            sys.stderr.write(f"⚠️  加载问答知识库 {QA_KB_PATH} 失败：{e}\n")
    return _qa_kb_instance


# ─── 咨询智能体（可选） ───
# 跨库咨询：意图分类 → KB 路由 → 多源检索 → 结构化回答
_consultation_agent: Any = None

def _get_consultation_agent() -> Any:
    global _consultation_agent
    if _consultation_agent is None:
        try:
            from consultation_agent import ConsultationAgent
            kb_paths: dict[str, str] = {}
            # book_kb
            if KB_REGISTRY:
                first_kb = next(iter(KB_REGISTRY.values()))
                kb_paths["book_kb"] = str(first_kb.kb_path)
            # qa_kb
            if QA_KB_PATH:
                kb_paths["qa_kb"] = str(QA_KB_PATH)
            # case_kb（从环境变量）
            case_path = os.environ.get("CASE_KB_PATH")
            if case_path:
                kb_paths["case_kb"] = case_path
            _consultation_agent = ConsultationAgent(kb_paths=kb_paths)
        except Exception as e:
            sys.stderr.write(f"⚠️  咨询智能体初始化失败：{e}\n")
    return _consultation_agent


def _resolve_kb(kb_name: str | None) -> LegalKnowledgeBase:
    if kb_name is None or kb_name == "":
        return KB_REGISTRY[DEFAULT_KB_NAME]
    if kb_name in KB_REGISTRY:
        return KB_REGISTRY[kb_name]
    raise ValueError(
        f"未知知识库 '{kb_name}'。可用：{', '.join(KB_REGISTRY.keys())}"
    )


# ─── MCP 服务 ───

mcp = FastMCP("legal-kb-builder")


@mcp.tool()
def list_kbs() -> dict[str, Any]:
    """
    列出当前 MCP server 挂载的所有知识库及其根路径。
    多库场景下，其他检索工具的 kb 参数需要用返回的 name。
    """
    return {
        "default": DEFAULT_KB_NAME,
        "kbs": [
            {"name": name, "path": str(inst.kb_path)}
            for name, inst in KB_REGISTRY.items()
        ],
    }


@mcp.tool()
def list_books(kb: str = "") -> dict[str, Any]:
    """
    列出指定知识库中已入库的所有书籍。

    Args:
        kb: 知识库名（可选，缺省用默认库）。用 list_kbs 查看可用值。
    """
    inst = _resolve_kb(kb)
    main_index_path = inst.kb_path / "indices" / "main_index.json"
    if not main_index_path.exists():
        return {"kb": inst.kb_path.name, "books": []}
    with main_index_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    return {
        "kb": inst.kb_path.name,
        "books": [
            {
                "name": meta.get("name", ""),
                "author": meta.get("author"),
                "book_type": meta.get("type", meta.get("book_type")),
                "year": meta.get("year"),
                "added_at": meta.get("added_at"),
            }
            for meta in idx.get("books", [])
            if isinstance(meta, dict)
        ],
    }


@mcp.tool()
def search_by_article(article: str, book: str = "", kb: str = "") -> dict[str, Any]:
    """
    按法条编号精确检索。

    Args:
        article: 法条编号，如 "第1165条" / "第一千一百六十五条" / "第1条"。
        book: 限定书籍名（可选）。留空则跨全库检索。
        kb: 知识库名（可选，缺省用默认库）。
    """
    inst = _resolve_kb(kb)
    result = inst.hybrid_search.search_by_article(article, book or None)
    return result or {"error": "no result"}


@mcp.tool()
def search_by_query(query: str, book: str = "", kb: str = "") -> dict[str, Any]:
    """
    自然语言 / 关键词检索。走 BM25 + 向量 + 结构索引三路融合。

    Args:
        query: 检索问题，如 "格式条款的效力认定" / "表见代理构成要件"。
        book: 限定书籍名（可选）。
        kb: 知识库名（可选，缺省用默认库）。
    """
    inst = _resolve_kb(kb)
    result = inst.hybrid_search.search_by_query(query, book or None)
    return result or {"error": "no result"}


@mcp.tool()
def search_by_scholar(scholar: str, topic: str, book: str = "", kb: str = "") -> dict[str, Any]:
    """
    检索特定学者对某个法律问题的论述。

    Args:
        scholar: 学者姓名，如 "王泽鉴"。
        topic: 法律问题或概念，如 "无权处分"。
        book: 限定书籍名（可选）。
        kb: 知识库名（可选，缺省用默认库）。
    """
    inst = _resolve_kb(kb)
    result = inst.hybrid_search.search_scholar_view(scholar, topic, book or None)
    return result or {"error": "no result"}


@mcp.tool()
def get_book_index(book: str, kb: str = "") -> dict[str, Any]:
    """
    读取某本书的 _知识库索引.json，返回结构条目、topic_index、citation_graph 等。
    用于上层做定制化的二次分析（例如按主题聚合、绘制引用图谱）。

    Args:
        book: 书籍名。
        kb: 知识库名（可选，缺省用默认库）。
    """
    inst = _resolve_kb(kb)
    book_index_path = inst.kb_path / "books" / book / "_book_index.json"
    kb_index_path = inst.kb_path / "books" / book / "_知识库索引.json"
    index_path = book_index_path if book_index_path.exists() else kb_index_path
    if not index_path.exists():
        return {"error": f"未找到书籍索引：{book}"}
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
#  问答知识库工具（流水线 D — qa_kb）
# ═══════════════════════════════════════════════════

@mcp.tool()
def qa_search(query: str, category: str = "", tag: str = "",
              mode: str = "hybrid", top_k: int = 10) -> dict[str, Any]:
    """
    问答知识库检索（流水线 D）。

    策略："问题匹配为主、答案召回为辅"，RRF 融合。
    适用于 FAQ / 业务咨询问答 / 培训问答 等问答形态语料。

    Args:
        query: 用户提问。
        category: 限定分类（可选，如 "合同法"）。
        tag: 限定标签（可选）。
        mode: hybrid（默认，问题+答案）| question（仅问题匹配）| answer（仅答案召回）。
        top_k: 返回前 K 条（默认 10）。
    """
    qa_kb = _get_qa_kb()
    if qa_kb is None:
        return {"error": "问答知识库未加载。请通过 QA_KB_PATH 环境变量或 --qa-kb 参数指定。"}
    results = qa_kb.search(
        query, category=category or None, tag=tag or None,
        mode=mode, top_k=top_k,
    )
    return {"query": query, "count": len(results), "results": results}


@mcp.tool()
def qa_stats() -> dict[str, Any]:
    """问答知识库统计信息。"""
    qa_kb = _get_qa_kb()
    if qa_kb is None:
        return {"error": "问答知识库未加载。"}
    return qa_kb.stats()


@mcp.tool()
def qa_recommend(question: str, top_k: int = 5) -> dict[str, Any]:
    """
    推荐相似问题。只走问题索引，用于"用户输入时联想推荐"场景。

    Args:
        question: 用户正在输入的问题。
        top_k: 返回前 K 个相似问题。
    """
    qa_kb = _get_qa_kb()
    if qa_kb is None:
        return {"error": "问答知识库未加载。"}
    results = qa_kb.recommend_similar(question, top_k=top_k)
    return {"question": question, "recommendations": results}


# ═══════════════════════════════════════════════════
#  跨库咨询工具（Consultation Agent）
# ═══════════════════════════════════════════════════

@mcp.tool()
def consult(question: str, reset_context: bool = False) -> dict[str, Any]:
    """
    跨库业务咨询。

    自动完成：意图分类 → KB 路由 → 多源检索 → 证据融合 → 结构化回答。
    支持多轮对话（同一 MCP 会话内保留上下文）。

    Args:
        question: 用户的业务咨询问题。
        reset_context: 是否重置对话上下文（开始新话题时设为 true）。
    """
    agent = _get_consultation_agent()
    if agent is None:
        return {"error": "咨询智能体初始化失败，请检查知识库配置。"}
    if reset_context:
        agent.reset_dialogue()
    result = agent.ask(question)
    return result.to_dict()


# ─── 入口 ───

if __name__ == "__main__":
    # FastMCP 默认走 stdio，与 Claude Desktop / Cursor / Continue 等 stdio 客户端兼容。
    mcp.run()
