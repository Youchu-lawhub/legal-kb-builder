# 配置参考

## split_pdf.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | （必填） | 输入PDF文件路径 |
| `-o, --output` | `splits` | 输出目录 |
| `--mode` | `bookmark` | 拆分模式：bookmark / manual / auto |
| `--level` | `1` | 书签层级深度（0=仅顶级） |
| `--config` | 无 | 手动模式YAML配置路径 |
| `--chunk-size` | `150` | 自动模式每段页数 |
| `--max-pages` | `200` | 单文件最大页数（MinerU当前限制，可通过 parser-backends.yaml 的 max_pages 覆盖） |
| `--analyze` | false | 仅分析PDF结构不拆分 |

## merge_md.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_path` | （必填） | 单个 Markdown 文件或 MD 文件所在目录 |
| `-o, --output` | `knowledge_base` | 输出知识库目录 |
| `--name` | （必填） | 知识库名称 |
| `--manifest` | 无 | split_manifest.json路径 |
| `--max-size` | `500KB` | 单卷文件最大大小 |
| `--legal` | false | 启用法律文本模式 |
| `--clean` | false | 启用 OCR/法律文本清洗；保留行边界以供结构字段抽取 |

## split_manifest.json 格式

拆分脚本自动生成的清单文件，记录拆分结果和MinerU上传计划：

```json
{
  "source": "/path/to/original.pdf",
  "source_name": "original",
  "total_pages": 1200,
  "total_size_mb": 150.5,
  "split_mode": "bookmark",
  "files_count": 5,
  "mineru_batches": 1,
  "mineru_batch_plan": [
    {
      "batch": 1,
      "files": ["/path/to/001.pdf", "/path/to/002.pdf"],
      "count": 2
    }
  ],
  "files": [
    {
      "index": 1,
      "filename": "001_第一编总则.pdf",
      "filepath": "/absolute/path/001_第一编总则.pdf",
      "title": "第一编 总则",
      "start_page": 1,
      "end_page": 280,
      "pages": 280,
      "size_bytes": 35000000,
      "size_mb": 33.4
    }
  ]
}
```

## 解析后端页数限制（2026-07 更新）

各后端单文件页数上限（超过时 parser_adapter 自动分片）：

| 后端 | 默认上限 | 可配置 | 说明 |
|------|---------|--------|------|
| MinerU | **200 页** | parser-backends.yaml → max_pages | 2026-07 从 600 降至 200 |
| 阿里云百炼 | 200 页 | 同上 | |
| 合合信息 TextIn | 200 页 | 同上 | |
| PaddleOCR | 500 页 | 同上 | 本地部署，限制较宽松 |
| LibreOffice | 无限制 | — | 本地工具 |

> 若后端限制有变动，在 `parser-backends.yaml` 中对应后端下加 `max_pages: 新值` 即可，无需改代码。
> `parser_adapter.py` 检测到 PDF 页数超过 max_pages 时，自动调用 `split_pdf.py` 拆分 → 逐片解析 → 合并。
> 若某片仍失败，自动缩小分片到 max_pages/2 重试，最多降级 2 次。

## YAML配置模板字段

用于手动模式的配置文件：

| 字段 | 类型 | 说明 |
|------|------|------|
| `book.title` | string | 书名（信息参考） |
| `book.author` | string | 作者（信息参考） |
| `book.total_pages` | int | 总页数（信息参考） |
| `segments[].title` | string | 分段标题 |
| `segments[].start_page` | int | 起始页（自然页码，从1开始） |
| `segments[].end_page` | int | 结束页（自然页码） |

## 检索配置（config-template.yaml）

### 向量检索（search.vector）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `model` | `BAAI/bge-small-zh-v1.5` | 嵌入模型，可切换为 `bge-large-zh-v1.5` 或 `bge-m3`。**模型变更后需重建索引**（维度可能变化） |
| `model_cache` | `~/.cache/legal-kb-builder/models/` | 模型缓存目录 |
| `chunk_size` | `512` | 分块字符数 |
| `chunk_overlap` | `50` | 块间重叠字符数 |

可用模型（`python embedding_manager.py --models` 查看）：

| 模型 | 维度 | 特点 |
|------|------|------|
| `BAAI/bge-small-zh-v1.5` | 384 | 轻量快速，适合低配环境（默认） |
| `BAAI/bge-large-zh-v1.5` | 1024 | 精度更高，MTEB 中文榜领先 small |
| `BAAI/bge-m3` | 1024 | 多语言+多粒度(Dense/Sparse/ColBERT)，2025 前沿 |

> 切换模型后需重建索引：`python embedding_manager.py build <kb_path> --model BAAI/bge-m3`

### 融合配置（search.fusion）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `method` | `rrf` | 融合算法（Reciprocal Rank Fusion） |
| `rrf_k` | `60` | RRF 常数（标准值） |
| `weights.*` | `1.0` | 各路检索权重（index/bm25/vector） |

### 重排序配置（search.rerank，v2.1 新增）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用 Cross-Encoder 重排 |
| `model` | `BAAI/bge-reranker-v2-m3` | 重排模型（通过 sentence-transformers 加载） |
| `candidate_pool` | `30` | 从 RRF 结果取前 N 个做重排 |
| `top_k` | `5` | 重排后输出的最终结果数 |

> 重排为可选依赖，复用 sentence-transformers。未安装时自动降级为纯 RRF。

### GraphRAG 配置（search.graph，v2.2 新增）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用法条关联图谱 |
| `max_depth` | `2` | 多跳查询最大深度 |
| `top_k` | `10` | 返回结果数 |

> 图谱为第四路检索，融入 RRF。图谱不存在时自动降级为三路融合。构建图谱：`python graph_rag.py build <kb_path>`

### 查询改写配置（search.query_rewrite，v2.2 新增）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用查询改写 |
| `mode` | `rule` | 模式：`rule`(规则，无依赖) 或 `llm`(LLM，需 API key) |

> 规则模式：口语→书面语规范化 + 同义词扩展 + 指代消解。
> LLM 模式：语义改写 + 复杂问题拆解为子查询（需设置 `LLM_API_KEY` 环境变量）。

## RAG 评估（eval/）

| 命令 | 说明 |
|------|------|
| `python eval/eval_rag.py --kb-path <kb>` | 基础检索评估（hit_rate/mrr/recall，无依赖） |
| `python eval/eval_rag.py --kb-path <kb> --full --answers-file answers.jsonl` | 完整 RAGAS 评估；答案文件须含系统实际生成的 `query`/`answer` |
| `python eval/eval_rag.py --kb-path <kb> --label <name>` | 带标签，用于对比实验 |

详见 `eval/README.md`。
