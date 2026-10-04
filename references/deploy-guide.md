# 云端部署指南

把本地知识库智能体托管到云端平台，获取公网可远程访问的智能体 API。

## 两条路径

| 平台 | 特点 | 公网 API | 适合场景 |
|------|------|---------|---------|
| **Qoder Cloud Agents** | 全托管，Agent 在云端 sandbox 运行 | `https://api.qoder.com/api/v1/cloud` | 想快速获得公网 API，无需自备服务器 |
| **阿里云百炼** | 通过百炼 CLI 创建智能体应用 | 百炼应用 API | 已有百炼账号，想用百炼的模型生态 |

---

## 前置条件

### 通用

- 本地知识库已构建完成（`kb_factory.py build` 产出 `_知识库索引.json` + md 文件）
- 检索脚本已验证可用（`hybrid_search.py` 能返回结果）

### Qoder Cloud Agents

```bash
# 1. 注册 Qoder 账号：https://qoder.com
# 2. 获取个人访问令牌（PAT）：
#    Qoder 控制台 → 设置 → 个人访问令牌 → 创建令牌
# 3. 设置环境变量
export QODER_PAT="your-personal-access-token"

# 4. 验证
curl -s "https://api.qoder.com/api/v1/cloud/agents?limit=1" \
  -H "Authorization: Bearer $QODER_PAT"
# 应返回 {"data":[],"first_id":null,...}
```

### 阿里云百炼

```bash
# 1. 安装百炼 CLI（需 Node.js ≥ 22.12）
npm install -g bailian-cli
npx skills add modelstudioai/cli --all -g

# 2. 获取 API Key：
#    https://bailian.console.aliyun.com/cn-beijing/?tab=app#/api-key

# 3. 认证（二选一）
bl auth login --console         # 浏览器登录（推荐）
bl auth login --api-key sk-xxx  # API Key 登录

# 4. 验证
bl auth status --output json
bl text chat --message "ping" --non-interactive --output json
```

---

## 路径一：Qoder Cloud Agents

### 快速部署

```bash
# 交互式部署（推荐）
python3 scripts/deploy_agent.py

# 或直接指定
python3 scripts/deploy_agent.py --platform qoder --kb-path ~/kbs/民法典评注
```

### 部署流程

```
本地知识库 ──→ Qoder Cloud Agent
                  │
                  ├── 1. 获取/创建 Environment（运行环境）
                  ├── 2. 上传知识库文件到 Files
                  │     • kb_dir/*.md（知识库正文）
                  │     • kb_dir/_知识库索引.json（索引）
                  │     • scripts/hybrid_search.py（检索脚本）
                  │     • scripts/embedding_manager.py
                  │     • scripts/bm25_searcher.py
                  │     • scripts/reranker.py
                  │     • scripts/graph_rag.py
                  │     • scripts/query_rewriter.py
                  │     • scripts/shared_utils.py
                  │     • assets/keyword-expansion.yaml
                  ├── 3. 创建 Agent（含 system prompt + tools）
                  └── 4. 输出公网 API endpoint
```

### 部署后调用

部署完成后，你会获得 `agent_id` 和 `environment_id`。通过以下方式调用：

```bash
# 1. 创建 Session
SESSION_ID=$(curl -s -X POST https://api.qoder.com/api/v1/cloud/sessions \
  -H "Authorization: Bearer $QODER_PAT" \
  -H "Content-Type: application/json" \
  -d "{\"agent\": \"$AGENT_ID\", \"environment_id\": \"$ENV_ID\"}" | jq -r '.id')

# 2. 发送咨询消息
curl -s -X POST "https://api.qoder.com/api/v1/cloud/sessions/$SESSION_ID/events" \
  -H "Authorization: Bearer $QODER_PAT" \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"type": "user.message", "content": [{"type": "text", "text": "善意取得的构成要件有哪些？"}]}
    ]
  }'

# 3. 实时接收响应（SSE 流）
curl -s -N "https://api.qoder.com/api/v1/cloud/sessions/$SESSION_ID/events/stream" \
  -H "Authorization: Bearer $QODER_PAT"
```

### 对外集成

拿到公网 API 后，你可以：

- **网站/小程序/App**：后端调用 Qoder API，前端展示 SSE 事件流
- **MCP 服务**：把 Qoder API 封装为 MCP 工具
- **CLI 工具**：写一个 CLI 脚本调用 Qoder API
- **钉钉/飞书机器人**：webhook 接收消息 → 调用 Qoder API → 返回结果

### 注意事项

- Qoder Agent 在云端 sandbox 运行，每次 Session 独立
- 知识库文件上传后在 Agent 环境中可访问
- 向量检索依赖（torch ~2GB）在 sandbox 中安装较慢，建议先用 BM25+Grep 模式
- Session 最长运行 26 小时，适合长程任务

---

## 路径二：阿里云百炼

### 三种知识库接入模式

| 模式 | 原理 | 优势 | 限制 |
|------|------|------|------|
| `upload` | 上传 md 文件到百炼内置知识库 | 百炼自己做检索，无需部署服务 | 丢失你的四路混合检索能力 |
| `mcp` | 百炼智能体通过 MCP 调用你的 mcp_server | 保留完整检索能力 | 需要公网服务器部署 mcp_server |
| `api` | 百炼智能体通过 HTTP 调用你的 api_server | 保留完整检索能力 | 需要公网服务器部署 api_server |

### 模式选择建议

- **没有公网服务器** → `upload` 模式（接受百炼内置检索）
- **有公网服务器，想保留混合检索** → `mcp` 模式（推荐）
- **已有 api_server 运行中** → `api` 模式

### 快速部署

```bash
# 交互式部署
python3 scripts/deploy_agent.py

# 或直接指定
python3 scripts/deploy_agent.py --platform bailian --kb-path ~/kbs/民法典评注
```

### MCP 模式部署流程（推荐）

```
步骤 1：在公网服务器部署 mcp_server.py
─────────────────────────────────────────
# 在你的公网服务器上
git clone <your-repo> /opt/legal-kb-builder
cd /opt/legal-kb-builder
pip install -r requirements.txt -r requirements-vector.txt -r requirements-mcp.txt

# 启动 MCP 服务（stdio 模式，由客户端通过 stdin/stdout 通信）
# MCP 不支持 --port，它是 stdio 协议，通常由 MCP 客户端（如 Claude Desktop）直接启动
python3 scripts/mcp_server.py \
  --kb /opt/kbs/民法典评注 \
  --qa-kb /opt/kbs/faq_kb

步骤 2：配置 MCP 客户端或百炼智能体
──────────────────────────────────
# mcp_server.py 使用 stdio 协议，不支持直接 HTTP 连接。
# 远程使用有两种方式：
#   方式 A：通过 SSH 隧道，让本地 MCP 客户端启动远程 mcp_server.py（stdio）
#     示例（Claude Desktop mcp_config.json）：
#     { "command": "ssh", "args": ["user@server", "python3 /opt/legal-kb-builder/scripts/mcp_server.py --kb /opt/kbs/民法典评注 --qa-kb /opt/kbs/faq_kb"] }
#
#   方式 B：使用 api_server.py（HTTP API），适合百炼/前端等远程调用
#     python3 scripts/api_server.py --book-kb /opt/kbs/民法典评注 --qa-kb /opt/kbs/faq_kb --host 0.0.0.0 --port 8000
#
# 智能体 system prompt:
#   你是一个法律知识库咨询智能体。
#   用户提问时，通过 MCP 工具检索本地知识库。
#   工具：consult / qa_search / search_by_article / search_by_query

步骤 3：发布应用获取 API
──────────────────────
# 在百炼控制台「发布应用」
# 发布后在「应用调用」页面获取：
#   - App ID
#   - API Key
#   - API endpoint: https://dashscope.aliyuncs.com/api/v1/apps/{app_id}/completion
```

### 部署后调用

```bash
# 百炼应用 API 调用
curl -X POST "https://dashscope.aliyuncs.com/api/v1/apps/$APP_ID/completion" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "善意取得的构成要件有哪些？"
    },
    "parameters": {},
    "debug": {}
  }'
```

### Upload 模式（无服务器）

```bash
# 用百炼 CLI 上传知识库文件
bl kb upload --file ~/kbs/民法典评注/第一部分_总则.md
bl kb upload --file ~/kbs/民法典评注/第二部分_物权.md
# ... 上传所有 md 文件

# 在百炼控制台创建知识库应用，绑定上传的文档
# https://bailian.console.aliyun.com/cn-beijing/?tab=app
```

---

## 配置文件

部署配置在 `assets/deploy-backends.yaml`（从 `deploy-backends.example.yaml` 复制）：

```bash
cp assets/deploy-backends.example.yaml assets/deploy-backends.yaml
$EDITOR assets/deploy-backends.yaml
```

关键配置项：

```yaml
platforms:
  qoder:
    enabled: true
    pat: ""  # 或用环境变量 QODER_PAT
    agent:
      name: "legal-kb-agent"
      model: "ultimate"
      system: |
        你是一个法律知识库咨询智能体...
  
  bailian:
    enabled: true
    api_key: ""  # 或用 bl auth login
    app:
      name: "legal-kb-agent"
      model: "qwen-plus"
      kb_mode: "mcp"  # upload / mcp / api
```

---

## 部署助手命令

```bash
# 检查前置条件（两个平台都检查）
python3 scripts/deploy_agent.py --check

# 生成配置文件
python3 scripts/deploy_agent.py --generate-config

# 交互式部署
python3 scripts/deploy_agent.py

# 指定平台部署
python3 scripts/deploy_agent.py --platform qoder --kb-path ~/kbs/民法典评注
python3 scripts/deploy_agent.py --platform bailian --kb-path ~/kbs/民法典评注

# 演练（不实际调用 API）
python3 scripts/deploy_agent.py --platform qoder --kb-path ~/kbs/民法典评注 --dry-run
```

---

## 两条路径对比

| 维度 | Qoder Cloud Agents | 阿里云百炼 |
|------|-------------------|-----------|
| **公网服务器** | 不需要（全托管） | mcp/api 模式需要 |
| **检索能力** | 在 sandbox 运行你的脚本 | upload 模式用百炼内置；mcp/api 用你的 |
| **部署速度** | 快（上传文件 + 创建 Agent） | 中（upload 快，mcp/api 需先部署服务） |
| **模型** | ultimate（Qoder 平台模型） | qwen-max/plus/turbo（可选） |
| **API 格式** | RESTful + SSE 事件流 | RESTful（同步/流式） |
| **计费** | 按 Session 运行时长 | 按 Token 用量 |
| **适合场景** | 快速获得公网 API | 深度集成百炼生态 |

---

## 常见问题

**Q: 部署到 Qoder 后，向量检索能用吗？**

A: 可以，但需要安装 torch（~2GB），sandbox 中首次安装较慢。建议先用 BM25+Grep 模式（`requirements.txt` 里的依赖），验证可用后再加向量检索。

**Q: 百炼 mcp 模式需要什么？**

A: 需要一台公网服务器（VPS/云服务器），部署 `mcp_server.py` 并暴露端口。百炼智能体通过 MCP 协议调用你的检索能力。

**Q: 部署后 API 调用有延迟吗？**

A: Qoder：Agent 在云端 sandbox 运行，首次 Session 启动有冷启动延迟（~10秒），之后响应快。百炼 mcp 模式：延迟 = 百炼模型推理 + MCP 网络往返 + 本地检索时间。

**Q: 知识库更新后怎么办？**

A: Qoder：重新上传文件到 Files，创建新 Session。百炼 upload 模式：重新上传文档到百炼知识库。百炼 mcp/api 模式：更新本地知识库即可（服务实时检索）。
