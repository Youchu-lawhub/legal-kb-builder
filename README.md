# legal-kb-builder

**本地法律知识库建设一体化工厂。**

用于本地整理、索引和检索法律材料；对外 API 需要显式配置身份验证并经受控网络部署。

- 🔀 **双层路由**：先判"是不是 md（不是就交给你配置的解析后端）"，再判"该走书籍 KB / 裁判文书 KB / 问答集 KB / Agentic 实时检索 哪条流水线"。
- 🧩 **解析后端可插拔**：MinerU、阿里云百炼、合合信息 TextIn、PaddleOCR、LibreOffice、pandoc、任意自写脚本——你自己在 `assets/parser-backends.yaml` 里配。
- 🔎 **三路融合检索**：BM25（`jieba` + `rank_bm25`）+ 向量（`bge-small-zh-v1.5` + FAISS）+ 结构索引，用 RRF 融合。
- 💬 **问答集流水线**：支持 FAQ / 业务咨询问答 / 培训问答等问答形态语料，"问题匹配为主、答案召回为辅"双索引检索。
- 🤖 **业务咨询智能体**：跨库编排（意图分类→KB路由→多源检索→结构化回答），支持多轮对话。
- 🌐 **可选服务接口**：HTTP API（FastAPI）与 MCP；API 所有路由要求 `LEGAL_KB_API_TOKEN`。Webhooks 应经验证请求的反向代理转发。
- ⚖️ **面向法律语义**：区分"书的目录条文"与"正文引用条文"，避免把 69 条司法解释建成 489 条索引这类常见事故。
- 📦 **本地优先**：核心工作流以文件和 CLI 运行；向量、OCR、API 与 MCP 按需安装，可选解析服务由调用方配置。

## 快速开始

```bash
# 1) 依赖
python3 -m pip install -r requirements.txt

# 2) 配置解析后端（至少启用一个）
cp assets/parser-backends.example.yaml assets/parser-backends.yaml
# 打开 parser-backends.yaml，把你要用的后端 enabled 改为 true 并填 token/cmd

# 3) 一键构建（工厂入口自动路由 + 构建）
python3 scripts/kb_factory.py build ~/legal_materials/合同法FAQ.pdf \
    --output ~/kbs/faq_kb --name "合同法FAQ"

# 4) 本地命令行咨询
python3 scripts/consultation_agent.py --qa-kb ~/kbs/faq_kb ask "格式条款无效？"
# 5) 可选 API：先生成强 token；只经 TLS / 受控反向代理暴露；webhook 由代理做签名验证
LEGAL_KB_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" python3 scripts/api_server.py --qa-kb ~/kbs/faq_kb
# 6) 可选 MCP（本机 stdio）
python3 scripts/mcp_server.py  # 配合 KB_PATH / QA_KB_PATH 环境变量
```

## 四条流水线 + 咨询智能体

| 类型 | 适用场景 | 主脚本 |
|---|---|---|
| **book_kb** | 法典评注 / 司法解释理解与适用 / 学术专著 / 教科书 / 案例汇编 / 法规汇编 / 实务指引 | `scripts/legal_kb.py` |
| **case_kb** | 法院裁判文书（判决书 / 裁定书 / 调解书 / 决定书） | `scripts/judgment_kb.py` |
| **qa_kb** | FAQ / 业务咨询问答 / 培训问答 / 客服话术 / 法律咨询问答集 | `scripts/qa_kb.py` |
| **agentic** | 探索性学术资料（论文 / 比较法 / 报告 / 单本随查） | `scripts/monte_carlo_sampler.py` |
| **consult** | 跨库业务咨询（意图分类→KB路由→多源检索→结构化回答） | `scripts/consultation_agent.py` |

细节请见 [SKILL.md](SKILL.md)。

## 目录

```
legal-kb-builder/
├── SKILL.md                  # 详细使用说明（AI Agent 也可直接读作 skill 定义）
├── README.md                 # 本文件
├── requirements.txt
├── scripts/                  # 全部可执行脚本
│   ├── kb_factory.py         # 工厂主入口（一键路由 + 构建）
│   ├── consultation_agent.py # 业务咨询智能体
│   ├── api_server.py         # HTTP API 服务（钉钉/飞书 webhook）
│   ├── mcp_server.py         # MCP 服务
│   └── ...                   # 四条流水线 + 检索引擎
├── assets/                   # 配置：解析后端 / 路由规则 / 词表 / 正则 / 问答 / 咨询
└── references/               # 深度文档（检索模式 / 索引模板 / 配置字段 / 端到端样例）
```

## 设计原则

1. **目录即锚**：书籍类知识库的一切都以书的真实目录为唯一依据。
2. **正文引用 ≠ 结构条目**：一本 69 条的司法解释书里，正文出现的"民法典第 584 条"是引用，不是本书条目。
3. **只纳入主文**：排除前言、序言、后记、致谢、脚注、出版信息。
4. **问题匹配为主**：问答集检索以"用户提问 ↔ 库内问题"的语义对齐为核心，答案召回为辅。
5. **本地优先，按需依赖**：基础索引可离线运行；向量、OCR 和服务依赖分开安装，并在构建结果中报告缺失能力。

## 许可

采用限制性许可 **CC BY-NC-ND 4.0 + 6 附加条款**；因限制商业使用和分发改编，本项目不是 OSI 定义的开源软件。署名／非商业／禁改编 + 禁 AI 训练 / 禁上架收费平台 / 企业与行政机关识别为商用 / 学术引用格式 / 禁背书 / 权利保留。完整条款见 [LICENSE](LICENSE)。商业授权、企业部署授权、AI 训练豁免请联系作者：游初 &lt;994559732@qq.com&gt;。
