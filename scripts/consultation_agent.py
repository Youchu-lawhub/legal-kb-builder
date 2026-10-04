#!/usr/bin/env python3
"""
业务咨询智能体（Consultation Agent）

把 legal-kb-builder 的多条流水线（book_kb / case_kb / qa_kb / agentic）
编排成一个统一的业务咨询智能体：

    用户提问
       │
    意图分类（article_lookup / case_search / faq_consult / ...）
       │
    KB 路由（每种意图查询 1-N 个知识库）
       │
    多源检索（qa_kb / case_kb / book_kb 各自检索）
       │
    证据融合（RRF + 去重 + 阈值过滤）
       │
    结构化回答（按意图选择回答模板，拼接证据）

支持多轮对话上下文（指代消解 + 上下文衰减）。

可被以下场景调用：
  - 前端问答平台（通过 api_server.py HTTP 接口）
  - 钉钉/飞书机器人后端（通过 api_server.py webhook 适配）
  - MCP 客户端（通过 mcp_server.py 的 consult 工具）
  - 命令行直接调用

配置：assets/consultation-config.yaml
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shared_utils import load_yaml_config

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_DEFAULT_CONFIG = _ASSETS_DIR / "consultation-config.yaml"


# ─── 数据结构 ───

@dataclass
class Evidence:
    """单条检索证据"""
    source_kb: str               # qa_kb / case_kb / book_kb / agentic
    content: str                 # 证据文本（问题/答案/条文/裁判要旨等）
    score: float                 # 相关度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元信息
    matched_via: str = ""        # question / answer / article / case / ...

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_kb": self.source_kb,
            "content": self.content,
            "score": round(self.score, 4),
            "metadata": self.metadata,
            "matched_via": self.matched_via,
        }


@dataclass
class ConsultationResult:
    """咨询结果"""
    answer: str                       # 最终回答文本
    intent: str                       # 识别的意图
    routed_kbs: List[str]             # 实际查询的知识库
    evidence: List[Evidence]          # 使用的证据
    retrieval_score: float             # 检索器原始分数，未经概率校准
    suggestions: List[str] = field(default_factory=list)  # 建议/澄清

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "routed_kbs": self.routed_kbs,
            "evidence": [e.to_dict() for e in self.evidence],
            "retrieval_score": round(self.retrieval_score, 4),
            "suggestions": self.suggestions,
        }


# ─── 咨询智能体 ───

class ConsultationAgent:
    """
    业务咨询智能体

    Args:
        config_path: 咨询配置 YAML 路径
        kb_paths: 知识库路径映射 {"qa_kb": "...", "case_kb": "...", "book_kb": "..."}
                  未提供的类型会被跳过
    """

    def __init__(
        self,
        config_path: str = None,
        kb_paths: Dict[str, str] = None,
        query_rewrite_mode: str = "rule",
    ):
        path = Path(config_path) if config_path else _DEFAULT_CONFIG
        self._config = load_yaml_config(str(path))
        self._kb_paths = kb_paths or {}

        # 各 KB 实例的懒加载缓存
        self._qa_kb = None
        self._case_kb = None
        self._book_kb = None

        # 对话上下文（每轮 {"question","intent","evidence_top1"}）
        self._dialogue_history: List[Dict] = []

        # 查询改写器（v2 新增，规则模式默认，可选 LLM 模式）
        self._query_rewrite_mode = query_rewrite_mode
        self._rewriter = None  # 延迟初始化

    # ─── 主入口 ───

    def ask(self, question: str, context: List[Dict] = None) -> ConsultationResult:
        """
        回答业务咨询。

        Args:
            question: 用户提问
            context: 对话上下文（可选，覆盖内部历史）

        Returns:
            ConsultationResult
        """
        # 1. 查询改写（v2 新增：规范化 + 同义词扩展 + 指代消解 + 子问题拆解）
        rewrite_result = self._rewrite_query(question, context or self._dialogue_history)
        resolved_q = rewrite_result.get("rewritten", question)
        sub_queries = rewrite_result.get("sub_queries", [])
        expansions = rewrite_result.get("expansions", [])

        # 2. 意图分类
        intent = self._classify_intent(resolved_q)

        # 3. KB 路由
        routed_kbs = self._route_to_kbs(intent)

        # 4. 多源检索（若有子问题，对每个子问题分别检索后融合）
        all_evidence: List[Evidence] = []
        if sub_queries:
            # LLM 模式拆解了子问题：对每个子查询分别检索
            for sub_q in sub_queries:
                for kb_type in routed_kbs:
                    evidence = self._search_kb(kb_type, sub_q)
                    all_evidence.extend(evidence)
            # 也对改写后的主查询检索一次
            for kb_type in routed_kbs:
                evidence = self._search_kb(kb_type, resolved_q)
                all_evidence.extend(evidence)
        else:
            # 规则模式或简单问题：单次检索（查询已改写+扩展）
            search_q = resolved_q
            if expansions:
                # 把同义词附加到查询中（OR 语义，BM25 会自然处理）
                search_q = f"{resolved_q} {' '.join(expansions[:3])}"
            for kb_type in routed_kbs:
                evidence = self._search_kb(kb_type, search_q)
                all_evidence.extend(evidence)

        # 5. 证据融合
        fused_evidence = self._fuse_evidence(all_evidence)

        # 6. 生成回答
        answer, retrieval_score = self._generate_answer(intent, resolved_q, fused_evidence)

        # 7. 建议/澄清
        suggestions = self._generate_suggestions(intent, resolved_q, fused_evidence)

        result = ConsultationResult(
            answer=answer,
            intent=intent,
            routed_kbs=routed_kbs,
            evidence=fused_evidence,
            retrieval_score=retrieval_score,
            suggestions=suggestions,
        )

        # 8. 更新对话历史
        self._dialogue_history.append({
            "question": question,
            "resolved": resolved_q,
            "intent": intent,
            "evidence_top1": fused_evidence[0].content[:100] if fused_evidence else "",
            "timestamp": datetime.now().isoformat(),
        })

        # 上下文窗口截断
        window = self._config.get("dialogue", {}).get("context_window", 5)
        if len(self._dialogue_history) > window:
            self._dialogue_history = self._dialogue_history[-window:]

        return result

    def reset_dialogue(self):
        """重置对话上下文"""
        self._dialogue_history.clear()

    # ─── 查询改写（v2 新增） ───

    def _rewrite_query(
        self, question: str, history: List[Dict]
    ) -> Dict[str, Any]:
        """
        查询改写入口（替代原 _resolve_coreference 的单一功能）。

        优先用 query_rewriter 模块（规则/LLM 双模式），
        依赖不可用时降级为原 _resolve_coreference。
        """
        # 延迟初始化改写器
        if self._rewriter is None:
            try:
                from query_rewriter import QueryRewriter
                self._rewriter = QueryRewriter(mode=self._query_rewrite_mode)
            except ImportError:
                # 降级：用原指代消解
                resolved = self._resolve_coreference(question, history)
                return {"rewritten": resolved, "expansions": [], "sub_queries": []}

        try:
            return self._rewriter.rewrite(question, history)
        except Exception as e:
            print(f"[ConsultationAgent] 查询改写异常，降级: {e}")
            resolved = self._resolve_coreference(question, history)
            return {"rewritten": resolved, "expansions": [], "sub_queries": []}

    # ─── 意图分类 ───

    def _classify_intent(self, question: str) -> str:
        """基于关键词规则的意图分类"""
        rules = self._config.get("intent_classification", {}).get("rules", [])

        for rule in rules:
            for kw_pattern in rule.get("keywords", []):
                try:
                    if re.search(kw_pattern, question):
                        return rule.get("intent", "faq_consult")
                except re.error:
                    if kw_pattern in question:
                        return rule.get("intent", "faq_consult")

        default = self._config.get("intent_classification", {}).get("default_intent", "faq_consult")
        return default

    # ─── KB 路由 ───

    def _route_to_kbs(self, intent: str) -> List[str]:
        """根据意图返回要查询的知识库列表（按优先级）"""
        routes = self._config.get("kb_routing", {}).get("routes", {})
        kbs = routes.get(intent, ["qa_kb", "book_kb"])

        # 过滤掉未配置路径的 KB
        available = [kb for kb in kbs if kb in self._kb_paths or kb == "agentic"]
        return available if available else kbs  # 如果都没配置，返回原始列表（让 _search_kb 容错）

    # ─── 多源检索 ───

    def _search_kb(self, kb_type: str, question: str) -> List[Evidence]:
        """搜索单个知识库（容错：不可用则返回空）"""
        try:
            if kb_type == "qa_kb":
                return self._search_qa_kb(question)
            elif kb_type == "case_kb":
                return self._search_case_kb(question)
            elif kb_type == "book_kb":
                return self._search_book_kb(question)
            elif kb_type == "agentic":
                return self._search_agentic(question)
        except Exception as e:
            print(f"[ConsultationAgent] {kb_type} 检索失败: {e}", file=sys.stderr)
        return []

    def _search_qa_kb(self, question: str) -> List[Evidence]:
        """检索问答知识库"""
        kb_path = self._kb_paths.get("qa_kb")
        if not kb_path:
            return []

        if self._qa_kb is None:
            from qa_kb import QAKnowledgeBase
            self._qa_kb = QAKnowledgeBase(kb_path)

        max_per_kb = self._config.get("answer_generation", {}).get("fusion", {}).get("max_per_kb", 5)
        results = self._qa_kb.search(question, top_k=max_per_kb)

        evidence = []
        for r in results:
            content = f"问：{r.get('question', '')}\n答：{r.get('answer', '')}"
            evidence.append(Evidence(
                source_kb="qa_kb",
                content=content,
                score=r.get("score", 0.0),
                metadata={
                    "id": r.get("id"),
                    "category": r.get("category"),
                    "tags": r.get("tags"),
                    "source": r.get("source"),
                },
                matched_via=",".join(r.get("matched_via", [])),
            ))
        return evidence

    def _search_case_kb(self, question: str) -> List[Evidence]:
        """检索裁判文书知识库"""
        kb_path = self._kb_paths.get("case_kb")
        if not kb_path:
            return []

        if self._case_kb is None:
            from judgment_kb import JudgmentKnowledgeBase
            self._case_kb = JudgmentKnowledgeBase(kb_path)

        max_per_kb = self._config.get("answer_generation", {}).get("fusion", {}).get("max_per_kb", 5)
        results = self._case_kb.search(query=question, top_k=max_per_kb)

        evidence = []
        for r in results:
            snippet = r.get("snippet", "")
            evidence.append(Evidence(
                source_kb="case_kb",
                content=snippet,
                score=r.get("score", 0.0),
                metadata={
                    "case_number": r.get("case_number"),
                    "court": r.get("court"),
                    "cause_of_action": r.get("cause_of_action"),
                },
                matched_via="case",
            ))
        return evidence

    def _search_book_kb(self, question: str) -> List[Evidence]:
        """检索书籍知识库（通过子进程调用 hybrid_search.py）"""
        kb_path = self._kb_paths.get("book_kb")
        if not kb_path:
            return []

        max_per_kb = self._config.get("answer_generation", {}).get("fusion", {}).get("max_per_kb", 5)
        try:
            result = subprocess.run(
                [
                    sys.executable, str(_SCRIPT_DIR / "hybrid_search.py"),
                    "--kb", kb_path,
                    "--query", question,
                    "--top-k", str(max_per_kb),
                    "--json",
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return []
            data = json.loads(result.stdout)
            # hybrid_search 的输出格式可能是 {"results": [...]} 或 [...]
            items = data if isinstance(data, list) else data.get("results", data.get("matches", []))
            evidence = []
            for item in items:
                # HybridSearch 按书籍聚合结果；把 top_matches 展开为证据。
                matches = item.get("top_matches") or [item]
                for match in matches:
                    snippets = match.get("snippets", [])
                    content = (snippets[0] if snippets else match.get(
                        "snippet", match.get("text", match.get("content", ""))
                    ))
                    if not content:
                        continue
                    evidence.append(Evidence(
                        source_kb="book_kb",
                        content=content,
                        score=float(match.get("score", item.get("score", item.get("rrf_score", 0.0)))),
                        metadata={
                            "book": item.get("book"),
                            "article": item.get("article"),
                            "file": match.get("file", item.get("file")),
                        },
                        matched_via=item.get("matched_via", "hybrid"),
                    ))
            return evidence
        except subprocess.TimeoutExpired as e:
            print(f"[WARN] _search_book_kb 检索超时: {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"[WARN] _search_book_kb 执行异常: {e}", file=sys.stderr)
            return []

    def _search_agentic(self, question: str) -> List[Evidence]:
        """Agentic 实时检索（通过子进程调用 monte_carlo_sampler.py）"""
        library_path = self._kb_paths.get("agentic_library")
        if not library_path:
            return []

        try:
            result = subprocess.run(
                [
                    sys.executable, str(_SCRIPT_DIR / "monte_carlo_sampler.py"),
                    "--library", library_path,
                    "--query", question,
                    "--mode", "fast",
                    "--json",
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return []
            data = json.loads(result.stdout)
            items = data if isinstance(data, list) else data.get("results", data.get("evidence", []))
            evidence = []
            for item in items:
                content = item.get("text", item.get("snippet", item.get("content", "")))
                evidence.append(Evidence(
                    source_kb="agentic",
                    content=content,
                    score=float(item.get("score", item.get("confidence", 0.0))),
                    metadata={"file": item.get("file"), "source": item.get("source")},
                    matched_via="agentic",
                ))
            return evidence
        except subprocess.TimeoutExpired as e:
            print(f"[WARN] _search_agentic 检索超时: {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"[WARN] _search_agentic 执行异常: {e}", file=sys.stderr)
            return []

    # ─── 证据融合 ───

    def _fuse_evidence(self, evidence: List[Evidence]) -> List[Evidence]:
        """融合多源证据：阈值过滤 + 去重 + 截断"""
        gen_cfg = self._config.get("answer_generation", {})
        fusion_cfg = gen_cfg.get("fusion", {})

        min_rel = fusion_cfg.get("min_relevance", 0.35)
        max_total = fusion_cfg.get("max_total_evidence", 8)
        dedup_threshold = fusion_cfg.get("dedup_threshold", 0.75)

        # 1. 阈值过滤
        filtered = [e for e in evidence if e.score >= min_rel]

        # 2. 按分数排序
        filtered.sort(key=lambda e: e.score, reverse=True)

        # 3. 去重（基于文本相似度——简化版：Jaccard 字符重叠）
        deduped: List[Evidence] = []
        for e in filtered:
            is_dup = False
            for d in deduped:
                if self._text_similarity(e.content, d.content) >= dedup_threshold:
                    # 保留分数更高的（已排序，所以 d 分数 >= e 分数）
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(e)

        # 4. 截断
        return deduped[:max_total]

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """简化的文本相似度（字符级 Jaccard）"""
        if not a or not b:
            return 0.0
        set_a = set(a)
        set_b = set(b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    # ─── 回答生成 ───

    def _generate_answer(
        self, intent: str, question: str, evidence: List[Evidence]
    ) -> Tuple[str, float]:
        """按意图模板生成结构化回答"""
        gen_cfg = self._config.get("answer_generation", {})
        templates = gen_cfg.get("templates", {})
        template = templates.get(intent, templates.get("faq_consult", {}))
        structure = template.get("structure", ["【回答】", "【依据】"])
        max_evidence = template.get("max_evidence", 3)

        fallback_cfg = self._config.get("fallback", {})
        low_score_threshold = fallback_cfg.get("min_retrieval_score",
                                            fallback_cfg.get("low_confidence_threshold", 0.02))

        if not evidence:
            return fallback_cfg.get("no_result_message", "未找到相关内容。"), 0.0

        # 原始检索分数用于显示与阈值规则；不解释为答案正确概率
        retrieval_score = evidence[0].score

        parts: List[str] = []

        # 低检索分提示
        if retrieval_score < low_score_threshold:
            parts.append(fallback_cfg.get("low_confidence_message", "以下回答仅供参考："))

        # 按模板结构拼接
        ev_idx = 0
        for section in structure:
            # 模板节必须有可支撑的证据才输出，避免生成空的“法律依据”等栏位。
            if ev_idx >= len(evidence):
                break
            parts.append(f"\n{section}")
            count = 0
            while ev_idx < len(evidence) and count < max_evidence:
                ev = evidence[ev_idx]
                source_label = self._source_label(ev.source_kb)
                parts.append(f"  [{source_label}] {ev.content}")
                if ev.metadata:
                    meta_str = "；".join(f"{k}: {v}" for k, v in ev.metadata.items() if v)
                    if meta_str:
                        parts.append(f"  （{meta_str}）")
                ev_idx += 1
                count += 1
                if count >= max_evidence:
                    # 让下一个 section 从新的证据开始
                    break

        answer = "\n".join(parts).strip()
        return answer, retrieval_score

    @staticmethod
    def _source_label(kb_type: str) -> str:
        labels = {
            "qa_kb": "问答库",
            "case_kb": "案例库",
            "book_kb": "书籍库",
            "agentic": "实时检索",
        }
        return labels.get(kb_type, kb_type)

    # ─── 建议/澄清 ───

    def _generate_suggestions(
        self, intent: str, question: str, evidence: List[Evidence]
    ) -> List[str]:
        """生成建议或澄清提示"""
        suggestions = []
        clarify_threshold = self._config.get("dialogue", {}).get("clarify_threshold", 0.30)

        if not evidence:
            suggestions.append("尝试换一种表述重新提问")
            suggestions.append("提供更多背景信息（如涉及的法律关系、当事人角色）")
        elif evidence[0].score < clarify_threshold:
            suggestions.append("您的问题可能涉及多个方面，能否进一步明确具体场景？")

        # 如果有多个分类，建议用户指定
        categories = set()
        for ev in evidence:
            cat = ev.metadata.get("category")
            if cat:
                categories.add(cat)
        if len(categories) > 2:
            suggestions.append(f"检索结果涉及多个分类：{', '.join(categories)}，可指定分类缩小范围")

        return suggestions

    # ─── 指代消解 ───

    def _resolve_coreference(
        self, question: str, history: List[Dict]
    ) -> str:
        """简单的指代消解：当问题过短或含指代词时，拼接上一轮主题"""
        if not history or not self._config.get("dialogue", {}).get("coreference_resolution", True):
            return question

        # 检测指代词
        coref_words = ["这个", "那个", "它", "该", "上述", "前面提到", "刚才说的", "这种情况", "这种"]
        has_coref = any(w in question for w in coref_words)

        # 检测过短问题
        is_short = len(question.strip()) <= 6

        if not (has_coref or is_short):
            return question

        # 取上一轮的问题主题
        last = history[-1]
        last_q = last.get("resolved", last.get("question", ""))

        # 简单拼接（不做复杂句法分析）
        if has_coref:
            return f"关于「{last_q}」的进一步问题：{question}"
        else:
            return f"{last_q} {question}"

    # ─── 知识库自动发现 ───

    def discover_kbs(self) -> Dict[str, str]:
        """自动发现已构建的知识库"""
        discovery_cfg = self._config.get("auto_discovery", {})
        search_dirs = discovery_cfg.get("search_dirs", ["~/legal_kb", "./kbs"])
        type_detection = discovery_cfg.get("type_detection", {})

        found: Dict[str, str] = {}

        for dir_pattern in search_dirs:
            search_dir = Path(dir_pattern).expanduser()
            if not search_dir.exists():
                continue

            # 搜索子目录
            for sub in search_dir.iterdir():
                if not sub.is_dir():
                    continue

                for kb_type, detection in type_detection.items():
                    has_files = all((sub / f).exists() for f in detection.get("has_files", []))
                    has_dirs = all((sub / d).exists() for d in detection.get("has_dirs", []))
                    if has_files and has_dirs:
                        found[kb_type] = str(sub)
                        break

        return found


# ─── CLI ───

def main():
    import argparse

    parser = argparse.ArgumentParser(description="业务咨询智能体")
    parser.add_argument("--qa-kb", help="问答知识库路径")
    parser.add_argument("--case-kb", help="裁判文书知识库路径")
    parser.add_argument("--book-kb", help="书籍知识库路径")
    parser.add_argument("--agentic-library", help="Agentic 检索库路径")
    parser.add_argument("--config", help="咨询配置 YAML 路径")
    parser.add_argument("--discover", action="store_true", help="自动发现知识库")
    parser.add_argument("--reset", action="store_true", help="重置对话上下文")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    subparsers = parser.add_subparsers(dest="command")

    ask_p = subparsers.add_parser("ask", help="提问")
    ask_p.add_argument("question", help="问题")
    ask_p.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()

    # 构建知识库路径映射
    kb_paths: Dict[str, str] = {}
    if args.qa_kb:
        kb_paths["qa_kb"] = args.qa_kb
    if args.case_kb:
        kb_paths["case_kb"] = args.case_kb
    if args.book_kb:
        kb_paths["book_kb"] = args.book_kb
    if args.agentic_library:
        kb_paths["agentic_library"] = args.agentic_library

    agent = ConsultationAgent(config_path=args.config, kb_paths=kb_paths)

    # 自动发现
    if args.discover:
        discovered = agent.discover_kbs()
        kb_paths.update(discovered)
        agent = ConsultationAgent(config_path=args.config, kb_paths=kb_paths)
        print(f"自动发现知识库: {discovered}", file=sys.stderr)

    if args.reset:
        agent.reset_dialogue()

    if args.command == "ask":
        result = agent.ask(args.question)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"【意图】{result.intent}")
            print(f"【路由】{', '.join(result.routed_kbs)}")
            print(f"【检索分数】{result.retrieval_score:.4f}（未校准，不代表正确概率）")
            print(f"\n{result.answer}")
            if result.suggestions:
                print("\n【建议】")
                for s in result.suggestions:
                    print(f"  - {s}")
    elif not args.command:
        parser.print_help()


if __name__ == "__main__":
    main()
