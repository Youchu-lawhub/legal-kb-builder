# RAG 评估框架

对法律知识库的检索与生成质量做量化评估，支撑迭代优化（换 embedding 模型、调 chunk_size、开关 rerank 等参数前后对比）。

## 快速开始

```bash
# 基础检索评估（无需 LLM，无额外依赖）
python eval/eval_rag.py --kb-path ~/legal_kb

# 指定 K 值
python eval/eval_rag.py --kb-path ~/legal_kb --top-k 5

# 完整 RAGAS 评估（需安装 ragas）
pip install ragas datasets
python eval/eval_rag.py --kb-path ~/legal_kb --full --answers-file answers.jsonl

# 带标签，用于对比实验
python eval/eval_rag.py --kb-path ~/kb --label "rerank_enabled"
python eval/eval_rag.py --kb-path ~/kb --label "rerank_disabled"
```

`answers.jsonl` 必须来自待评估系统的实际回答，每行形如 `{"query":"…","answer":"…"}`。不要以检索提示词、上下文或标准答案替代 `answer`，否则 RAGAS 的生成质量指标没有意义。

## 两套指标

### A. 基础检索指标（始终可用，无依赖）

| 指标 | 说明 | 目标 |
|------|------|------|
| `keyword_hit_rate@k` | Top-K 是否命中任一标注关键词（仅诊断） | 无准确率承诺 |
| `keyword_recall@k` | Top-K 命中标注关键词比例（仅诊断） | 无准确率承诺 |
| `hit_rate@k`, `mrr@k`, `recall@k` | 以人工标注的 `relevant_ids` 计算段落级指标 | 仅对有 qrels 的题目输出 |

`relevant_keywords` 命中与覆盖只作为诊断值；关键词同现不证明段落相关。标准 hit/MRR/recall 需要每个问题标注 `relevant_ids`，且这些 ID 必须与实际全库片段 ID 对应。当前内置数据没有 qrels，因此当前运行只报告关键词诊断。

### B. RAGAS 指标（可选，需 `pip install ragas`）

| 指标 | 说明 | 目标 |
|------|------|------|
| `faithfulness` | 答案是否完全基于检索上下文（防幻觉） | → 1.0 |
| `answer_relevance` | 答案是否回答了问题 | → 1.0 |
| `context_precision` | 相关文档是否排在前面 | → 1.0 |
| `context_recall` | 答案所需信息是否都在上下文 | → 1.0 |
| `answer_correctness` | 答案与标准答案的匹配度 | → 1.0 |

RAGAS 评估需要 LLM（默认调用 OpenAI，可配置其他），用于深度评估生成质量。

## 评估集格式

`eval/legal_qa_eval.jsonl`，每行一个 JSON：

```json
{
  "query": "善意取得的构成要件有哪些",
  "ground_truth_answer": "善意取得需满足：受让时善意、以合理价格受让、已交付或登记...",
  "relevant_keywords": ["善意取得", "善意", "合理价格", "交付", "登记", "第311条"],
  "category": "物权"
}
```

当前内置 20 条覆盖民法典各编（物权、合同、侵权、继承、婚姻家庭、担保、总则）的标注样本。
**要做检索质量验收，需先对材料版本与段落 ID 作人工相关性标注，分层覆盖题型和无答案问题。** 增加关键词条目可用于诊断，但不会替代 qrels。

## 构建 qrels（`relevant_ids`）

标准 `hit_rate@k / mrr@k / recall@k` 需要每条样本带 `relevant_ids`，且其中的 ID 必须与目标知识库实际产出的段落 ID 完全一致，否则指标恒为 0。ID 格式为 `{书籍名}::{chunk_id}`（`chunk_id` 由检索结果给出，形如 `文件名:行号` 或 `主题:关键词:条目`）。标注流程：

1. 用与正式评估相同的参数先跑一遍检索，拿到候选段落 ID。为避免“用被评估系统自己的输出当标准答案”，建议对每题放宽 `--top-k`（如 20）再人工判断相关性：
   ```bash
   python3 scripts/hybrid_search.py --kb <kb-root> --query "善意取得的构成要件有哪些" --top-k 20 --json
   ```
2. 从 JSON 输出中挑出真正回答该问题的段落，记录其 `book` 与 `chunk_id`，拼成 `{book}::{chunk_id}` 列表。
3. 写回评估集对应行的 `relevant_ids` 字段：
   ```json
   {"query": "善意取得的构成要件有哪些", "...": "...",
    "relevant_ids": ["王泽鉴民法总则::005_善意取得.md:line:45"]}
   ```
4. 未标注的样本会被自动归为 `keyword_diagnostic_only`，只出诊断值、不计入标准指标——这是允许的，可按题型分批补标。

qrels 与知识库版本绑定：换书、重新分块或调整 `chunk_size` 后段落 ID 会变，需相应更新 `relevant_ids`。

## 迭代优化闭环

```
1. 基线评估：python eval/eval_rag.py --kb-path ~/kb --label "baseline"
2. 调整参数（如切换 bge-m3 模型、开启 rerank、调 chunk_size）
3. 重新评估：python eval/eval_rag.py --kb-path ~/kb --label "bge_m3_rerank"
4. 对比 eval/results/ 下两次结果的 summary
5. 指标提升则采纳，下降则回退
```

### 短板诊断

| 指标低 | 可能原因 | 优化方向 |
|--------|---------|---------|
| `recall@k` 低 | 召回不足 | 增大 top_k / 换更强 embedding / 优化分块 |
| `hit_rate@k` 低 | 精排不足 | 开启 rerank / 调 RRF 权重 |
| `faithfulness` 低 | 幻觉 | 优化 prompt / 限制仅引用检索内容 |
| `context_recall` 低 | 检索缺失 | 扩充知识库 / 换 embedding 模型 |

## 结果存储

每次评估结果保存到 `eval/results/<timestamp>_<label>.json`，含：
- `summary`: 各指标平均值
- `details`: 每条样本的逐项指标
- `sample_count`: 样本数

对比实验时直接 diff 两个结果文件的 `summary` 即可。
