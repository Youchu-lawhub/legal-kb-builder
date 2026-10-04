# 使用样例

以下样例演示如何用 legal-kb-builder 走完从"一个 PDF"到"可检索知识库"的全流程。
所有命令均可复制粘贴，只需把路径改成你自己的。

## 样例 1 · 构建一本"合同编通则司法解释理解与适用"知识库（`book_kb`）

假设你有一个 800 页的 PDF：`~/legal_books/合同编通则司法解释理解与适用.pdf`。

```bash
cd /path/to/legal-kb-builder

# 0. 一次性准备：装依赖 + 配置解析后端
python3 -m pip install -r requirements.txt
cp assets/parser-backends.example.yaml assets/parser-backends.yaml
# 打开 parser-backends.yaml，把 aliyun-bailian（或 mineru / hehe-textin）
# 的 enabled 改为 true，并把 cmd 换成你自己的调用命令。

# 1. 格式检测（应输出 NEED-PARSE / pdf）
python3 scripts/format_detector.py ~/legal_books/合同编通则司法解释理解与适用.pdf

# 2. 拆分 PDF（150 页/卷 + 3 页重叠）
python3 scripts/split_pdf.py \
    ~/legal_books/合同编通则司法解释理解与适用.pdf \
    -o /tmp/合同通则_splits/ \
    --mode bookmark

# 3. 逐个解析成 md
for pdf in /tmp/合同通则_splits/*.pdf; do
    python3 scripts/parser_adapter.py "$pdf" -o /tmp/合同通则_md/
done

# 4. 质量校验
python3 scripts/quality_checker.py /tmp/合同通则_md/

# 5. 合并入库 + 边界去重 + 清洗
python3 scripts/merge_md.py /tmp/合同通则_md/ \
    -o ~/kb/合同通则/books/合同通则司法解释理解与适用/ \
    --name "合同编通则司法解释理解与适用" \
    --manifest /tmp/合同通则_splits/split_manifest.json \
    --legal --clean

# 6. 打开 ~/kb/合同通则/books/合同通则司法解释理解与适用/_目录.md
#    与 _知识库索引.json，手工校对：
#    - 索引条文数是否 = 69（司法解释总条文数）
#    - 是否混入了正文引用的民法典条文
#    - 抽 3–5 条 Grep 验证位置
#    修正后保存 _知识库索引.json

# 7. 构建 BM25 + 向量索引
python3 scripts/legal_kb.py --kb ~/kb/合同通则 build-search-index \
    --book "合同通则司法解释理解与适用"

# 8. 三路融合检索
python3 scripts/hybrid_search.py --kb ~/kb/合同通则 --query "格式条款效力认定"
```

## 样例 2 · 批量入库裁判文书（`case_kb`）

假设你有一个目录：`~/judgments_docx/`，里面 200 份合同纠纷判决书（Word）。

```bash
# 1. 全部转 md（parser_adapter 会挑合适的后端）
mkdir -p /tmp/judgments_md/
for f in ~/judgments_docx/*.docx; do
    python3 scripts/parser_adapter.py "$f" -o /tmp/judgments_md/
done

# 2. 初始化 KB
python3 scripts/judgment_kb.py --kb ~/kb/合同案例库/ init --name "合同纠纷案例库"

# 3. 批量导入（自动做要素抽取 → JSON 落盘）
python3 scripts/judgment_kb.py --kb ~/kb/合同案例库/ batch /tmp/judgments_md/

# 4. 构建 BM25 + 向量索引
python3 scripts/judgment_kb.py --kb ~/kb/合同案例库/ rebuild

# 5. 检索
python3 scripts/judgment_kb.py --kb ~/kb/合同案例库/ search --query "违约金调整标准"
python3 scripts/judgment_kb.py --kb ~/kb/合同案例库/ search --cause "买卖合同纠纷"
python3 scripts/judgment_kb.py --kb ~/kb/合同案例库/ search --case "(2023)京0108民初12345号"

# 6. 统计
python3 scripts/judgment_kb.py --kb ~/kb/合同案例库/ stats
```

## 样例 3 · 单本书随查随用（`agentic`）

场景：临时想在《王泽鉴·损害赔偿》里找"与有过失"相关论述，不想建常驻索引。

```bash
# 1. 转 md 并落到 agentic 缓存目录
BOOK=~/legal_books/王泽鉴_损害赔偿.pdf
CACHE=~/legal_books/.agentic_cache/王泽鉴_损害赔偿/
mkdir -p "$CACHE"
python3 scripts/parser_adapter.py "$BOOK" -o "$CACHE"

# 2. 先扫一眼文献库结构
python3 scripts/scan_library.py "$CACHE"

# 3. DEEP 模式蒙特卡洛检索
python3 scripts/monte_carlo_sampler.py \
    --library "$CACHE" \
    --query "与有过失的构成要件与法律效果" \
    --mode deep

# 4. 相似查询会命中集群缓存（>=0.85 相似度）自动加速
python3 scripts/monte_carlo_sampler.py \
    --library "$CACHE" \
    --query "过失相抵的适用条件" \
    --mode fast
```

## 样例 4 · 目录级自动路由

不想自己判断类型？把一批混合材料整个目录丢给 `material_router.py`：

```bash
for f in ~/legal_materials/*.md ~/legal_materials/*.pdf; do
    python3 scripts/material_router.py "$f"
done
# 观察输出的 target_skill：
#   book_kb   → 按样例 1 处理
#   case_kb   → 按样例 2 处理
#   agentic   → 按样例 3 处理
#   direct    → 手动 cp 到 references/
```

## 常见故障排查

| 现象 | 排查方向 |
|---|---|
| `parser_adapter` 报 "所有已启用后端都未能解析" | 检查 `parser-backends.yaml` 里至少一个后端 `enabled: true` 且 token/cmd 正确；用 `--dry-run` 观察选中的后端。 |
| `hybrid_search` 报缺 sentence-transformers/faiss | 未装可选依赖；要么 `pip install sentence-transformers faiss-cpu torch`，要么接受降级为 BM25 + 结构索引两路融合。 |
| 索引条文数远超书的目录 | 违反"目录即锚"原则——回到书的真实目录，手工核对索引条目并删除误入的正文引用条目。 |
| 中文路径导致解析后端报错 | 先 `cp` 到无中文的临时目录再调用（`parser_adapter.py` 内部用 `shlex.quote()` 已尽量规避）。 |
| 问答集被误判为书籍 | `routing-rules.yaml` 中 `qa_indicators.min_qa_pairs` 阈值过高；降低到 2 或检查问答标记模式是否覆盖你的格式。 |
| 咨询智能体返回"未找到相关内容" | 检查 `consultation-config.yaml` 中 `fusion.min_relevance` 是否过低/过高（RRF 分数量级约 0.01-0.10）；确认知识库已 `rebuild`。 |
| API 服务 `/ask` 返回空 | 确认 `--qa-kb` 参数已传入；检查 `/kb/list` 是否列出知识库；确认 `qa_kb rebuild` 已执行。 |
| 钉钉机器人无回复 | 确认 Outgoing webhook 地址正确；检查服务器防火墙放行端口；钉钉安全设置需配置 IP 白名单。 |
| 飞书 challenge 验证失败 | 确认 POST 地址可达；飞书要求 3 秒内返回 `{"challenge": "xxx"}`，API 已自动处理。 |

## 样例 5 · 构建问答集知识库（`qa_kb`）

假设你有一份业务咨询 FAQ：`~/legal_materials/合同法FAQ.md`，内容如下：

```
Q1: 格式条款无效的情形有哪些？
A1: 根据《民法典》第497条，格式条款无效的情形包括...

Q2: 违约金过高时如何调整？
A2: 当事人可以请求人民法院或者仲裁机构予以适当减少...
```

```bash
cd /path/to/legal-kb-builder

# 1. 路由检测（确认会被识别为 qa_kb）
python3 scripts/material_router.py ~/legal_materials/合同法FAQ.md
#   → target_skill: qa_kb

# 2. 初始化问答知识库
python3 scripts/qa_kb.py --kb ~/kbs/合同法FAQ init --name "合同法FAQ"

# 3. 导入问答对（自动检测格式）
python3 scripts/qa_kb.py --kb ~/kbs/合同法FAQ add ~/legal_materials/合同法FAQ.md --category "合同法"

# 4. 重建索引（问题索引 + 答案索引）
python3 scripts/qa_kb.py --kb ~/kbs/合同法FAQ rebuild

# 5. 检索
python3 scripts/qa_kb.py --kb ~/kbs/合同法FAQ search --query "格式条款无效" --top-k 5

# 6. 推荐相似问题
python3 scripts/qa_kb.py --kb ~/kbs/合同法FAQ recommend "违约金" --top-k 5

# 7. 统计
python3 scripts/qa_kb.py --kb ~/kbs/合同法FAQ stats
```

或用工厂入口一键完成步骤 1-4：

```bash
python3 scripts/kb_factory.py build ~/legal_materials/合同法FAQ.md \
    --output ~/kbs/合同法FAQ --name "合同法FAQ" --category "合同法"
```

## 样例 6 · 业务咨询智能体 + HTTP API

把问答库、案例库、书籍库统一编排为业务咨询智能体，通过 HTTP API 对外服务。

```bash
# 1. 启动 API 服务（同时加载多个知识库）
python3 scripts/api_server.py \
    --qa-kb ~/kbs/合同法FAQ \
    --case-kb ~/kbs/合同案例库 \
    --book-kb ~/kbs/合同通则 \
    --port 8000 --host 127.0.0.1

# 2. 前端调用通用咨询接口
curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question":"格式条款无效的情形有哪些？","session_id":"user_001"}'

# 3. 问答库直接检索
curl -X POST http://localhost:8000/qa/ask \
    -H "Content-Type: application/json" \
    -d '{"query":"违约金调整","category":"合同法","top_k":5}'

# 4. 查看知识库列表
curl http://localhost:8000/kb/list

# 5. 多轮对话（同一 session_id 自动保留上下文）
curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question":"那这个的构成要件是什么？","session_id":"user_001"}'
#   ↑ "这个"会自动指代上一轮的话题

# 6. 清除会话上下文
curl -X DELETE http://localhost:8000/session/user_001
```

## 样例 7 · 钉钉/飞书机器人接入

### 钉钉

```bash
# 1. 启动 API 服务（公网可达）
python3 scripts/api_server.py --qa-kb ~/kbs/合同法FAQ --port 8000 --host 127.0.0.1

# 2. 钉钉群 → 群设置 → 智能群助手 → 添加自定义机器人
#    安全设置 → IP 地址段 → 填入服务器公网 IP

# 3.（如支持 Outgoing）POST 地址填：
#    http://your-server:8000/webhook/dingtalk

# 4. 测试：在钉钉群 @机器人 发送 "违约金过高怎么调整？"
#    机器人自动返回结构化回答
```

### 飞书

```bash
# 1. 飞书开放平台 → 创建企业自建应用
# 2. 事件与回调 → 事件配置 → 请求地址：
#    http://your-server:8000/webhook/feishu
# 3. 添加事件：im.message.receive_v1（接收消息）
# 4. 权限管理：开通"读取用户发给机器人的单聊消息"

# API 自动处理飞书 URL 验证（challenge）和消息事件
# 用户在飞书中 @机器人 发送问题，机器人自动返回回答
```
