---
name: legal-kb-builder
version: 2.0.0
description: 构建、修复或评估本地法律知识库。适用于把法律书籍、裁判文书、问答集或已解析的 Markdown 变为可检索资产，并按资料类型选择书籍、裁判、问答或临时检索流水线；不用于一般法律问答或未经授权的对外部署。
metadata:
  clawdbot:
    emoji: ⚖️
    requires:
      anyBins:
        - python3
---

# 本地法律知识库建设

把用户提供的法律材料构建为可验证、可本地检索的知识库。默认交付本地目录和检索结果；只有用户明确要求时才启动 API、MCP 或云端部署。

## 开始前：只确认会改变方案的缺失信息

1. 输入是已解析的 `.md/.txt`，还是 PDF、DOCX、图片等需要解析的文件？非 Markdown 输入须先检查 `assets/parser-backends.yaml` 是否存在已启用的解析后端。没有可用后端时，说明阻塞原因并请用户选择/配置后端；不要把文件误当作已解析文本。
2. 用户要的是常驻、可重复检索的库，还是一次性的文献随查？前者使用 `book_kb`、`case_kb` 或 `qa_kb`；后者使用 `agentic`。
3. 用户是否明确要求对外服务或部署？未明确时止于本地构建与验证。

材料类型不明确时，先运行路由，不要仅凭文件扩展名猜测：

```bash
python3 scripts/format_detector.py <input>
python3 scripts/material_router.py <markdown-file-or-dir>
```

`material_router.py` 的建议是默认路由而非不可改变的结论。若用户已明确要“常驻书籍库”，应按书籍库流程构建，即使样本内容较短而被暂时建议为 `agentic`。

## 输出契约

每次构建以来源文件为单位记录状态；失败必须以非零状态结束，成功和依赖降级要分开报告。相同内容的 FAQ 可重复导入而不重复计数。向量索引按版本清单发布。所有书籍库都使用同一根目录契约；不要把书籍正文直接写在根目录：

```text
<kb-root>/
├── config.yaml
├── indices/main_index.json
└── books/
    └── <book-name>/
        ├── *.md
        ├── _知识库索引.json
        ├── _目录.md
        └── _bm25_corpus.json / _vectors.faiss  # 按已安装依赖生成
```

`hybrid_search.py`、`consultation_agent.py`、MCP 和 API 都把 `<kb-root>` 作为书籍库路径。裁判与问答库分别是独立根目录，不要嵌入 `books/`。

## 依赖和解析

最小依赖支持路由、清洗、BM25 与 PDF 拆分：

```bash
python3 -m pip install -r requirements.txt
```

只在用户需要相应能力时安装可选依赖：`requirements-vector.txt`（向量/重排）、`requirements-graph.txt`（GraphRAG）、`requirements-api.txt`、`requirements-mcp.txt`、`requirements-eval.txt`。使用 `scripts/install_deps.py --list` 或 `--check` 了解当前环境；不要为了普通构建安装全部依赖。

非 Markdown 材料的解析后端配置与页数限制见[配置参考](references/config-reference.md)。大型 PDF 优先保留书签边界并使用 `split_pdf.py`；细节见[端到端样例](references/examples.md)。

## 书籍与评注库（`book_kb`）

适用于法典评注、司法解释、专著、教材、案例汇编、法规汇编和实务指引。首先确认目录的真实组织单元：条文、章节、案例、法规或专题。正文引用的“第 X 条”不是目录结构，不可据此声称书籍包含该条目的评注。

已解析的 Markdown 可直接由工厂构建：

```bash
python3 scripts/kb_factory.py build <input.md-or-dir> \
  --output <kb-root> --name "<book-name>" --skip-parse --json
```

需要保留人工审核步骤时，使用手工链路。`merge_md.py` 既接受单个 `.md` 文件，也接受包含多个 Markdown 的目录：

```bash
KB=<kb-root>
BOOK=<book-name>
python3 scripts/legal_kb.py --kb "$KB" init --name "法律知识库"  # 仅首次
python3 scripts/merge_md.py <md-file-or-dir> \
  -o "$KB/books/$BOOK" --name "$BOOK" --legal --clean
python3 scripts/legal_kb.py --kb "$KB" build-search-index --book "$BOOK"
```

`_知识库索引.json` 是合并器生成的初稿。对条文驱动的材料，需人工核验目录条目、数量和抽样位置；使用七种书籍类型的索引策略时读取[检索指南](references/retrieval-guide.md)。来源文件信息在 JSON 索引中保存，不应出现在正文开头的分隔符或注释中。

检索时使用标准 CLI；`--json` 模式只向 stdout 输出 JSON，适合下游程序调用：

```bash
python3 scripts/hybrid_search.py --kb <kb-root> \
  --query "格式条款的效力认定" --top-k 5 --json
python3 scripts/hybrid_search.py --kb <kb-root> --article "第1165条" --json
```

检索按可用索引融合结构、BM25、向量和局部共现候选；缺少可选依赖时，结果会注明降级状态。共现图只提供带来源的位置候选，不表示条文适用、引用或解释关系。若同一条号可能属于多个法律或版本，先限定法律名称与版本并核验证据。`config.yaml` 中的 `search.rerank`、`search.query_rewrite` 与 `search.graph` 可覆盖相应开关。

## 裁判文书库（`case_kb`）

适用于判决书、裁定书、调解书。先转为 Markdown/文本后再导入；不要把 PDF 直接交给导入器。入库会提取案号、法院、当事人、案由、事实、说理、裁判结果与法条。案号检索会规范化全/半角括号，但仍保留原案号展示。

```bash
python3 scripts/judgment_kb.py --kb <case-kb-root> init --name "合同纠纷案例库"
python3 scripts/judgment_kb.py --kb <case-kb-root> batch <judgments-md-dir>
python3 scripts/judgment_kb.py --kb <case-kb-root> search --case "(2023)京0108民初12345号"
python3 scripts/judgment_kb.py --kb <case-kb-root> search --query "违约金调整标准"
```

导入后至少核验一份样本的 `court`、案号、当事人和 `court_reasoning`；若这些首部字段缺失，先检查解析文本的换行是否被破坏，再调整规则。

## 问答库（`qa_kb`）

适用于 FAQ、编号问答、Markdown 标题、表格、JSON 与连续问答。先用少量样本检查解析出的问答数量与边界，再批量导入：

```bash
python3 scripts/qa_kb.py --kb <qa-kb-root> init --name "合同法 FAQ"
python3 scripts/qa_kb.py --kb <qa-kb-root> add <faq.md> --category "合同法"
python3 scripts/qa_kb.py --kb <qa-kb-root> rebuild
python3 scripts/qa_kb.py --kb <qa-kb-root> search --query "格式条款无效"
```

问答格式与字段说明见 `assets/qa-patterns.yaml`、`assets/qa-fields.yaml`；必要时先用 `scripts/qa_parser.py` 单独检查。

## 临时 Agentic 检索（`agentic`）

用户只需要一次性探索、尚不值得建立常驻索引时，先将材料转为 Markdown 缓存，再执行 FAST/DEEP 检索。不要把它称为已建成的书籍知识库：

```bash
python3 scripts/monte_carlo_sampler.py \
  --library <markdown-library> --query "<question>" --mode fast --json
```

复杂检索策略、法条结构与多跳场景按需阅读[检索模式](references/search-patterns.md)。

## 咨询、评估与验收

跨库咨询只在对应库确实存在且已验证检索后使用：

```bash
python3 scripts/consultation_agent.py \
  --book-kb <kb-root> --case-kb <case-kb-root> --qa-kb <qa-kb-root> \
  ask "<question>" --json
```

交付前至少验证：构建命令退出成功、目标目录符合本页的输出契约、代表性查询返回有来源的片段、并核验一项资料类型特有字段（书籍目录锚点 / 裁判法院字段 / 问答对数）。保留依赖降级信息，避免把“未安装”误报成“索引已建成”。

基础评估只衡量检索，使用：

```bash
python3 eval/eval_rag.py --kb-path <kb-root> --top-k 5 --label "baseline"
```

内置关键词命中只作诊断，不能作为法律准确率或相关文档召回的证据。要报告 hit/MRR/recall，评估条目必须有 `relevant_ids`，对应全库唯一的段落 ID。`--full` 的 RAGAS 评估必须提供待评估系统**实际生成**的答案文件（JSONL，每行含 `query` 与 `answer`）；不能用检索提示词或标准答案代替模型回答。

## 可选服务与部署

仅在用户明确要求时启动。HTTP 服务默认监听 `127.0.0.1`，所有路由要求 `LEGAL_KB_API_TOKEN`。网络访问必须使用 TLS；将浏览器来源配置到 `LEGAL_KB_CORS_ORIGINS`。Webhook 应放在能验证平台签名、时间戳与重放的反向代理后，并由代理注入 Bearer token。当前服务按单一共享 token 设计，不作为多租户身份系统。部署细节按需读取[部署指南](references/deploy-guide.md)。

```bash
python3 scripts/api_server.py --book-kb <kb-root> --host 127.0.0.1 --port 8000
python3 scripts/mcp_server.py --kb <kb-root>
```

## 参考资料路由

- [端到端样例](references/examples.md)：完整的书籍、裁判、问答和咨询流程。
- [配置参考](references/config-reference.md)：解析、拆分、合并和 YAML 参数。
- [检索指南](references/retrieval-guide.md)：书籍类型、目录锚定与检索结果解释。
- [检索模式](references/search-patterns.md)：Agentic 的复杂检索路径。
- [部署指南](references/deploy-guide.md)：用户已明确要求云端托管时再读取。
