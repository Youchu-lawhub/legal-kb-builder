#!/usr/bin/env python3
"""
legal-kb-builder HTTP API 服务

把法律知识库构建工厂的检索与咨询能力以 HTTP 接口暴露，支撑：
  - 前端问答平台（通过 /ask 通用咨询接口）
  - 钉钉机器人后端（通过 /webhook/dingtalk）
  - 飞书机器人后端（通过 /webhook/feishu）
  - 任意 HTTP 客户端

## 环境要求

    pip install fastapi uvicorn pydantic

## 启动方式

    # 基本启动（自动发现知识库）
    python3 scripts/api_server.py

    # 指定知识库
    python3 scripts/api_server.py --qa-kb /path/to/qa_kb --case-kb /path/to/case_kb

    # 自定义端口
    LEGAL_KB_API_TOKEN=<random-secret> python3 scripts/api_server.py --port 8080 --host 0.0.0.0

All routes require Authorization: Bearer <token>; use TLS for network access.
Set LEGAL_KB_CORS_ORIGINS to a comma-separated list of allowed browser origins.
Third-party webhooks should pass through an authenticated reverse proxy that injects the bearer token.

## 接口概览

    POST /ask                 — 通用业务咨询（意图分类 → KB 路由 → 多源检索 → 结构化回答）
    POST /qa/ask              — 问答库直接检索
    POST /case/search         — 裁判文书检索
    GET  /kb/list             — 列出已加载的知识库
    GET  /kb/{kb_type}/stats  — 知识库统计
    GET  /health              — 健康检查
    POST /webhook/dingtalk    — 钉钉机器人 webhook 适配
    POST /webhook/feishu      — 飞书机器人 webhook 适配

## 钉钉机器人配置

钉钉自定义机器人 → 安全设置 → 关键词或加签 →
出方向（Outgoing）：POST 到 http://your-server:8000/webhook/dingtalk

## 飞书机器人配置

飞书开放平台 → 事件订阅 → 请求地址：http://your-server:8000/webhook/feishu
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import hmac
import re
import threading

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ─── FastAPI 依赖检查 ───
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    from pydantic import BaseModel, Field
except ImportError:
    sys.stderr.write(
        "❌ 需要安装 FastAPI：pip install fastapi uvicorn pydantic\n"
    )
    sys.exit(2)

try:
    import uvicorn
except ImportError:
    sys.stderr.write("❌ 需要安装 uvicorn：pip install uvicorn\n")
    sys.exit(2)

from consultation_agent import ConsultationAgent


# ─── 请求/响应模型 ───

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class QASearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    category: str = ""
    tag: str = ""
    mode: str = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)


class CaseSearchRequest(BaseModel):
    query: str = Field(default="", max_length=8000)
    case_number: str = ""
    cause: str = ""
    court: str = ""
    top_k: int = Field(default=10, ge=1, le=50)


# ─── 全局状态 ───

_agent: Optional[ConsultationAgent] = None
_kb_paths: Dict[str, str] = {}
_sessions: Dict[str, tuple] = {}  # session_id -> (agent, last_activity, request lock)
_MAX_SESSIONS = 500
_SESSION_TTL_SECONDS = 1800
_sessions_guard = threading.RLock()


def _load_kb_paths_from_env():
    """从环境变量加载知识库路径（uvicorn 字符串导入时使用）"""
    global _kb_paths
    env_map = {
        "QA_KB_PATH": "qa_kb",
        "CASE_KB_PATH": "case_kb",
        "BOOK_KB_PATH": "book_kb",
        "AGENTIC_LIBRARY_PATH": "agentic_library",
    }
    for env_key, kb_type in env_map.items():
        val = os.environ.get(env_key)
        if val and kb_type not in _kb_paths:
            _kb_paths[kb_type] = val


def _get_session(session_id: str):
    """获取会话并限制空闲与并发状态。"""
    now = time.monotonic()
    with _sessions_guard:
        for key, (_, touched, _) in list(_sessions.items()):
            if now - touched > _SESSION_TTL_SECONDS:
                _sessions.pop(key, None)
        if session_id not in _sessions:
            if len(_sessions) >= _MAX_SESSIONS:
                raise HTTPException(status_code=429, detail="会话数已达上限")
            _sessions[session_id] = (ConsultationAgent(kb_paths=_kb_paths), now, threading.RLock())
        agent, _, request_lock = _sessions[session_id]
        _sessions[session_id] = (agent, now, request_lock)
        return agent, request_lock


def _get_agent(session_id: str = "default") -> ConsultationAgent:
    return _get_session(session_id)[0]


def _ask_session(agent, request_lock, question):
    with request_lock:
        return agent.ask(question)


# 模块加载时从环境变量初始化（uvicorn 字符串导入场景）
_load_kb_paths_from_env()


# ─── Lifespan ───

@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动时加载知识库路径（环境变量 → 自动发现）"""
    global _kb_paths
    # 先从环境变量加载（main() 已将命令行参数写入环境变量）
    _load_kb_paths_from_env()
    # --discover 可补充显式配置；未提供任何路径时也自动发现。
    if not _kb_paths or os.environ.get("LEGAL_KB_DISCOVER") == "1":
        try:
            agent = ConsultationAgent(kb_paths={})
            discovered = agent.discover_kbs()
            for k, v in discovered.items():
                _kb_paths.setdefault(k, v)
            if _kb_paths:
                print(f"✅ 自动发现知识库: {list(_kb_paths.keys())}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  自动发现知识库失败: {e}", file=sys.stderr)
    if _kb_paths:
        print(f"✅ 已加载知识库: {list(_kb_paths.keys())}", file=sys.stderr)
    yield


# ─── FastAPI 应用 ───

app = FastAPI(
    title="legal-kb-builder API",
    description="法律知识库构建工厂 — 业务咨询智能体 HTTP 服务",
    version="2.0.0",
    lifespan=_lifespan,
)

# CORS — 开发默认值，生产环境请将 allow_origins 限制为具体域名
# 注意：allow_origins=["*"] 与 allow_credentials=True 在浏览器中互斥，
# 生产部署时二选一：要么指定 origin + credentials，要么 wildcard + 无 credentials。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def authenticate(request: Request, call_next):
    # Environment token is required for every route, including docs and webhooks.
    # This bearer guard is an application boundary; deploy behind TLS as well.
    expected = os.environ.get("LEGAL_KB_API_TOKEN", "")
    content_length = request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > 1_048_576:
        return JSONResponse(status_code=413, content={"detail": "请求体超过 1 MiB"})
    supplied = request.headers.get("authorization", "")
    token = supplied[7:] if supplied.startswith("Bearer ") else ""
    if not expected:
        return JSONResponse(status_code=503, content={"detail": "服务未配置 LEGAL_KB_API_TOKEN"})
    if not token or not hmac.compare_digest(token, expected):
        return JSONResponse(status_code=401, content={"detail": "需要有效 Bearer token"})
    return await call_next(request)


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": time.time(),
    }


@app.get("/kb/list")
async def list_kbs():
    """列出已加载的知识库"""
    return {
        "kbs": [
            {"type": k, "configured": True}
            for k, v in _kb_paths.items()
        ],
    }


@app.get("/kb/{kb_type}/stats")
async def kb_stats(kb_type: str):
    """获取指定知识库的统计信息"""
    kb_path = _kb_paths.get(kb_type)
    if not kb_path:
        raise HTTPException(status_code=404, detail=f"知识库类型 '{kb_type}' 未加载")

    try:
        if kb_type == "qa_kb":
            from qa_kb import QAKnowledgeBase
            kb = QAKnowledgeBase(kb_path)
            return kb.stats()
        elif kb_type == "case_kb":
            from judgment_kb import JudgmentKnowledgeBase
            kb = JudgmentKnowledgeBase(kb_path)
            return kb.stats()
        elif kb_type == "book_kb":
            # book_kb 通过子进程调用 hybrid_search 获取统计
            return {"kb_type": "book_kb", "path": kb_path}
        else:
            return {"error": f"不支持的知识库类型: {kb_type}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    """
    通用业务咨询接口。

    意图分类 → KB 路由 → 多源检索 → 证据融合 → 结构化回答。
    支持多轮对话（通过 session_id 维护上下文）。
    """
    session_id = req.session_id or uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", session_id):
        raise HTTPException(status_code=422, detail="session_id 只允许字母、数字及 ._:-")
    agent, request_lock = _get_session(session_id)
    result = await run_in_threadpool(_ask_session, agent, request_lock, req.question)
    response = result.to_dict()
    response["session_id"] = session_id
    return response


@app.post("/qa/ask")
async def qa_search(req: QASearchRequest):
    """问答库直接检索（不走意图分类，直接查问答库）"""
    kb_path = _kb_paths.get("qa_kb")
    if not kb_path:
        raise HTTPException(status_code=404, detail="问答知识库未加载")

    try:
        from qa_kb import QAKnowledgeBase
        kb = QAKnowledgeBase(kb_path)
        results = await run_in_threadpool(kb.search, req.query,
            req.category or None, req.tag or None, req.mode, req.top_k)
        return {"query": req.query, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/case/search")
async def case_search(req: CaseSearchRequest):
    """裁判文书检索"""
    kb_path = _kb_paths.get("case_kb")
    if not kb_path:
        raise HTTPException(status_code=404, detail="裁判文书知识库未加载")

    try:
        from judgment_kb import JudgmentKnowledgeBase
        kb = JudgmentKnowledgeBase(kb_path)
        results = await run_in_threadpool(kb.search, req.query or None,
            req.case_number or None, req.cause or None, req.court or None,
            top_k=req.top_k)
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 钉钉 Webhook 适配 ───

@app.post("/webhook/dingtalk")
async def webhook_dingtalk(request: Request):
    """
    钉钉机器人 Outgoing webhook 适配。

    钉钉 Outgoing 消息格式：
        {
          "msgtype": "text",
          "text": {"content": "用户消息"},
          "senderId": "...",
          "conversationId": "...",
          "senderNick": "用户昵称"
        }

    返回格式（钉钉要求）：
        {"msgtype": "text", "text": {"content": "回复内容"}}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # 提取用户消息
    content = ""
    if isinstance(body, dict):
        content = (
            body.get("text", {}).get("content", "")
            or body.get("content", "")
            or body.get("msgtype_content", "")
        )

    content = content.strip()
    if not content:
        return {"msgtype": "text", "text": {"content": "请输入您的问题。"}}

    # 会话 ID（用 conversationId 或 senderId）
    session_id = body.get("conversationId", body.get("senderId", "dingtalk_default"))

    # 调用咨询智能体
    agent, request_lock = _get_session(f"dingtalk_{session_id}")
    result = await run_in_threadpool(_ask_session, agent, request_lock, content)

    # 钉钉要求纯文本回复
    reply = result.answer
    if result.suggestions:
        reply += "\n\n💡 建议：\n" + "\n".join(f"• {s}" for s in result.suggestions)

    return {"msgtype": "text", "text": {"content": reply}}


# ─── 飞书 Webhook 适配 ───

@app.post("/webhook/feishu")
async def webhook_feishu(request: Request):
    """
    飞书机器人事件回调适配。

    支持两种场景：
    1. URL 验证（飞书首次配置时发送 challenge）：
       {"challenge": "xxx", "type": "url_verification"}
       → 返回 {"challenge": "xxx"}

    2. 消息事件（im.message.receive_v1）：
       {"event": {"message": {"content": "{\"text\":\"问题\"}"}, ...}}
       → 返回飞书消息卡片或纯文本
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # URL 验证
    if body.get("type") == "url_verification" or "challenge" in body:
        return {"challenge": body.get("challenge", "")}

    # 消息事件
    event = body.get("event", {})
    message = event.get("message", {})
    # 飞书消息 content 是 JSON 字符串
    raw_content = message.get("content", "{}")
    try:
        content_obj = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
    except (json.JSONDecodeError, TypeError):
        content_obj = {}

    content = ""
    if isinstance(content_obj, dict):
        content = content_obj.get("text", content_obj.get("content", ""))

    content = content.strip()
    if not content:
        return JSONResponse(content={"msg": "empty content"}, status_code=200)

    # 会话 ID
    chat_id = message.get("chat_id", event.get("sender", {}).get("sender_id", {}).get("open_id", "feishu_default"))

    # 调用咨询智能体
    agent, request_lock = _get_session(f"feishu_{chat_id}")
    result = await run_in_threadpool(_ask_session, agent, request_lock, content)

    # 飞书消息卡片格式
    reply_text = result.answer
    if result.suggestions:
        reply_text += "\n\n💡 建议：\n" + "\n".join(f"• {s}" for s in result.suggestions)

    return {
        "msg_type": "text",
        "content": {"text": reply_text},
    }


# ─── 会话管理 ───

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清除指定会话的上下文"""
    with _sessions_guard:
        if session_id in _sessions:
            _, _, lock = _sessions[session_id]
            with lock:
                del _sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"会话 '{session_id}' 不存在")


@app.get("/sessions")
async def list_sessions():
    """列出所有活跃会话"""
    return {
        "active_sessions": list(_sessions.keys()),
        "count": len(_sessions),
    }


# ─── 启动 ───

def _init_kb_paths(args):
    """将命令行参数写入环境变量（uvicorn 字符串导入时从环境变量读取）"""
    if args.qa_kb:
        os.environ["QA_KB_PATH"] = args.qa_kb
    if args.case_kb:
        os.environ["CASE_KB_PATH"] = args.case_kb
    if args.book_kb:
        os.environ["BOOK_KB_PATH"] = args.book_kb
    if args.agentic_library:
        os.environ["AGENTIC_LIBRARY_PATH"] = args.agentic_library
    if args.discover:
        os.environ["LEGAL_KB_DISCOVER"] = "1"


def main():
    parser = argparse.ArgumentParser(description="legal-kb-builder HTTP API 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--qa-kb", help="问答知识库路径")
    parser.add_argument("--case-kb", help="裁判文书知识库路径")
    parser.add_argument("--book-kb", help="书籍知识库路径")
    parser.add_argument("--agentic-library", help="Agentic 检索库路径")
    parser.add_argument("--discover", action="store_true", help="自动发现知识库")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")

    args = parser.parse_args()

    if not os.environ.get("LEGAL_KB_API_TOKEN"):
        parser.error("必须设置 LEGAL_KB_API_TOKEN；网络部署还必须使用 TLS")
    _init_kb_paths(args)

    print(f"🚀 启动 API 服务: http://{args.host}:{args.port}", file=sys.stderr)
    print(f"   API 文档: http://{args.host}:{args.port}/docs", file=sys.stderr)

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
