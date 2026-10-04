#!/usr/bin/env python3
"""
问答对解析器

从 Markdown / TXT / JSON 等格式的问答集语料中抽取结构化的 Q-A 对。

支持的格式：
  1. 显式标记型：  Q: 问题 / A: 答案  或  问:xxx / 答:xxx
  2. 编号问答型：  Q1: / A1:  或  1、问：xxx  答：xxx
  3. Markdown 标题型：### 问题 / ### 答案  或  ## Q1
  4. 表格型：      | 问题 | 答案 |（Markdown 表格）
  5. JSON 型：     [{"question":"...","answer":"..."}]
  6. 连续段落型：  问句行（以？结尾）后紧跟答案段落

抽取规则配置：assets/qa-patterns.yaml

使用方式：
    from qa_parser import QAParser

    parser = QAParser()
    pairs = parser.parse(text)
    # -> [QAPair(question=..., answer=..., ...), ...]
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shared_utils import load_yaml_config

# ─── 默认配置路径 ───
_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_DEFAULT_PATTERNS = _ASSETS_DIR / "qa-patterns.yaml"


@dataclass
class QAPair:
    """单个问答对"""
    question: str                              # 问题文本
    answer: str                                # 答案文本
    category: str = ""                         # 业务分类
    tags: List[str] = field(default_factory=list)  # 关键词标签
    source: str = ""                           # 答案依据（法条/规章）
    confidence: float = 1.0                    # 可靠性 0-1
    question_index: Optional[int] = None       # 编号（Q1 中的 1）
    line_range: Tuple[int, int] = (0, 0)       # 在原文中的行范围
    raw_format: str = ""                       # 解析来源格式

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "tags": self.tags,
            "source": self.source,
            "confidence": self.confidence,
            "question_index": self.question_index,
            "line_range": list(self.line_range),
            "raw_format": self.raw_format,
        }


class QAParser:
    """
    问答对解析器

    Args:
        patterns_path: 问答抽取规则 YAML 路径（默认使用内置）
    """

    def __init__(self, patterns_path: str = None):
        path = Path(patterns_path) if patterns_path else _DEFAULT_PATTERNS
        self._config = load_yaml_config(str(path))

    # ─── 主入口 ───

    def parse(self, text: str, fmt: str = "auto") -> List[QAPair]:
        """
        解析文本，提取问答对。

        Args:
            text: 原始文本
            fmt: 指定格式 auto|explicit|markdown|table|json|heuristic

        Returns:
            QAPair 列表
        """
        if not text or not text.strip():
            return []

        if fmt == "auto":
            fmt = self._detect_format(text)

        if fmt == "json":
            pairs = self._parse_json(text)
        elif fmt == "table":
            pairs = self._parse_table(text)
        elif fmt == "markdown":
            pairs = self._parse_markdown(text)
        elif fmt == "heuristic":
            pairs = self._parse_heuristic(text)
        else:  # explicit
            pairs = self._parse_explicit(text)

        # 元信息补充
        pairs = self._enrich_metadata(pairs, text)

        # 过滤与去重
        pairs = self._filter(pairs)

        return pairs

    def parse_file(self, file_path: str, fmt: str = "auto") -> List[QAPair]:
        """从文件解析问答对"""
        path = Path(file_path)
        if not path.exists():
            return []

        # JSON 文件直接按 JSON 解析
        if path.suffix.lower() == ".json":
            fmt = "json"

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gb18030", errors="replace")

        return self.parse(text, fmt=fmt)

    # ─── 格式检测 ───

    def _detect_format(self, text: str) -> str:
        """自动检测问答格式"""
        stripped = text.strip()

        # JSON
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                json.loads(stripped)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass

        # 同时评估 explicit 和 markdown 命中数，取较高的
        explicit_cfg = self._config.get("explicit_qa", {})
        q_pats = explicit_cfg.get("question_patterns", [])
        a_pats = explicit_cfg.get("answer_patterns", [])
        explicit_q_hits = 0
        explicit_a_hits = 0
        for p in q_pats:
            try:
                explicit_q_hits += len(re.findall(p, text))
            except re.error:
                continue
        for p in a_pats:
            try:
                explicit_a_hits += len(re.findall(p, text))
            except re.error:
                continue
        # explicit 有效对数 = min(q_hits, a_hits)（成对才算）
        explicit_pairs = min(explicit_q_hits, explicit_a_hits)

        # Markdown 标题命中数
        md_cfg = self._config.get("markdown_qa", {})
        md_pats = md_cfg.get("question_heading_patterns", [])
        md_hits = 0
        for p in md_pats:
            try:
                md_hits += len(re.findall(p, text))
            except re.error:
                continue

        # 如果 markdown 标题命中 >= 3 且 >= explicit 有效对数，优先 markdown
        # （explicit 只有问题没有答案标记时无法成对，应该 fallback 到 markdown）
        if md_hits >= 3 and md_hits >= explicit_pairs:
            return "markdown"

        # explicit 有效对数 >= 2
        if explicit_pairs >= 2:
            return "explicit"

        # explicit 问题命中 >= 2（即使答案标记不匹配，也尝试 explicit，可能答案用了非标准标记）
        if explicit_q_hits >= 2 and explicit_a_hits == 0 and md_hits >= 3:
            return "markdown"  # 有问题标记但无答案标记 + 有 markdown 标题 → markdown

        if explicit_q_hits >= 2:
            return "explicit"

        # 表格型
        table_cfg = self._config.get("table_qa", {})
        header_aliases = table_cfg.get("header_aliases", {})
        all_headers = []
        for v in header_aliases.values():
            all_headers.extend(v)
        if "|" in text and any(h in text for h in all_headers):
            return "table"

        # 启发式：检查是否有以问号结尾的行
        question_lines = re.findall(r"(?m)^.{4,120}？\s*$", text)
        if len(question_lines) >= 3:
            return "heuristic"

        # 默认尝试显式标记型
        return "explicit"

    # ─── 显式标记型解析 ───

    def _parse_explicit(self, text: str) -> List[QAPair]:
        """解析 Q:/A: 或 问:/答: 格式"""
        cfg = self._config.get("explicit_qa", {})
        q_pats = cfg.get("question_patterns", [])
        a_pats = cfg.get("answer_patterns", [])

        lines = text.splitlines()
        pairs: List[QAPair] = []

        current_q: Optional[str] = None
        current_q_idx: Optional[int] = None
        current_q_line: int = 0
        current_a_lines: List[str] = []
        current_a_line: int = 0
        in_answer = False

        def _flush():
            nonlocal current_q, current_q_idx, current_a_lines, in_answer
            if current_q and current_a_lines:
                answer_text = "\n".join(current_a_lines).strip()
                if answer_text:
                    pairs.append(QAPair(
                        question=current_q.strip(),
                        answer=answer_text,
                        question_index=current_q_idx,
                        line_range=(current_q_line, current_a_line),
                        raw_format="explicit",
                    ))
            current_q = None
            current_q_idx = None
            current_a_lines = []
            in_answer = False

        for i, line in enumerate(lines):
            matched_q = False
            matched_a = False

            # 检查是否为问题行
            for pat in q_pats:
                m = re.match(pat, line)
                if m:
                    _flush()
                    # 带编号的格式：Q1: xxx → groups = (1, xxx)
                    if m.lastindex and m.lastindex >= 2:
                        current_q_idx = self._safe_int(m.group(1))
                        current_q = m.group(2)
                    elif m.lastindex == 1:
                        current_q = m.group(1)
                    else:
                        current_q = m.group(0)
                    current_q_line = i
                    in_answer = False
                    matched_q = True
                    break

            if matched_q:
                continue

            # 检查是否为答案行
            for pat in a_pats:
                m = re.match(pat, line)
                if m:
                    if m.lastindex and m.lastindex >= 2:
                        current_a_lines = [m.group(2)]
                    elif m.lastindex == 1:
                        current_a_lines = [m.group(1)]
                    else:
                        current_a_lines = []
                    current_a_line = i
                    in_answer = True
                    matched_a = True
                    break

            if matched_a:
                continue

            # 普通行：如果在答案中，追加到答案
            if in_answer and line.strip():
                current_a_lines.append(line)
                current_a_line = i
            # 如果还没进答案但有当前问题，也尝试追加（问题与答案之间可能有空行）

        _flush()
        return pairs

    # ─── Markdown 标题型解析 ───

    def _parse_markdown(self, text: str) -> List[QAPair]:
        """解析 Markdown 标题型问答（支持智能分段）"""
        cfg = self._config.get("markdown_qa", {})
        q_pats = cfg.get("question_heading_patterns", [])
        end_pats = cfg.get("answer_end_markers", [r"(?m)^#{1,4}\s", r"(?m)^---\s*$"])

        # 智能分段配置
        structured_cfg = cfg.get("structured_sections", {})

        lines = text.splitlines()
        pairs: List[QAPair] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # 检查是否为问题标题
            matched_q = None
            for pat in q_pats:
                m = re.match(pat, line)
                if m:
                    # 取最后一个捕获组作为问题文本
                    q_text = m.group(m.lastindex) if m.lastindex else line
                    # 清理：去掉 markdown 格式残留
                    q_text = q_text.strip().lstrip("#").strip()
                    matched_q = q_text
                    break

            if matched_q is None:
                i += 1
                continue

            # 收集标题下方的答案段落
            q_line = i
            i += 1
            a_lines: List[str] = []
            while i < len(lines):
                line2 = lines[i]
                # 检查是否到达答案结束标记
                is_end = False
                for ep in end_pats:
                    if re.match(ep, line2):
                        is_end = True
                        break
                if is_end:
                    break
                a_lines.append(line2)
                i += 1

            raw_answer = "\n".join(a_lines).strip()
            if not raw_answer or not matched_q:
                continue

            # 智能分段：尝试从 raw_answer 中提取"问题详情/解析/律师建议"
            question_detail, structured_answer = self._extract_structured_sections(
                raw_answer, structured_cfg
            )

            # 如果成功分段，用分段结果；否则用原始答案
            if structured_answer:
                final_question = matched_q
                if question_detail:
                    final_question = f"{matched_q}\n（详细问题：{question_detail}）"
                final_answer = structured_answer
            else:
                final_question = matched_q
                final_answer = raw_answer

            pairs.append(QAPair(
                question=final_question.strip(),
                answer=final_answer.strip(),
                line_range=(q_line, i - 1),
                raw_format="markdown",
            ))

        return pairs

    def _extract_structured_sections(
        self, text: str, structured_cfg: Dict
    ) -> tuple:
        """
        从答案文本中智能分段，提取问题详情和结构化答案。

        适用于"## 标题"下方含"问题：/解析：/解释：/律师建议："分段的格式。

        Returns:
            (question_detail, structured_answer)
            - question_detail: "问题："标记后的详细问题描述
            - structured_answer: "解析/解释"+"律师建议"的组合
            如果未检测到分段标记，返回 ("", "")
        """
        if not structured_cfg:
            return ("", "")

        q_detail_pats = structured_cfg.get("question_detail_markers", [])
        analysis_pats = structured_cfg.get("analysis_markers", [])
        advice_pats = structured_cfg.get("advice_markers", [])

        # 如果没有任何分段标记，返回空
        all_pats = q_detail_pats + analysis_pats + advice_pats
        if not all_pats:
            return ("", "")

        # 检查是否含分段标记
        has_any_marker = False
        for p in all_pats:
            try:
                if re.search(p, text):
                    has_any_marker = True
                    break
            except re.error:
                continue
        if not has_any_marker:
            return ("", "")

        lines = text.splitlines()

        # 按标记分段
        sections: Dict[str, List[str]] = {
            "question_detail": [],
            "analysis": [],
            "advice": [],
            "other": [],
        }
        current_section = "other"

        for line in lines:
            matched = False
            # 检查问题详情标记
            for p in q_detail_pats:
                m = re.match(p, line)
                if m:
                    current_section = "question_detail"
                    if m.lastindex:
                        sections[current_section].append(m.group(m.lastindex))
                    matched = True
                    break
            if matched:
                continue
            # 检查解析/解释标记
            for p in analysis_pats:
                m = re.match(p, line)
                if m:
                    current_section = "analysis"
                    if m.lastindex:
                        sections[current_section].append(m.group(m.lastindex))
                    matched = True
                    break
            if matched:
                continue
            # 检查建议标记
            for p in advice_pats:
                m = re.match(p, line)
                if m:
                    current_section = "advice"
                    if m.lastindex:
                        sections[current_section].append(m.group(m.lastindex))
                    matched = True
                    break
            if matched:
                continue
            # 普通行追加到当前段
            if line.strip():
                sections[current_section].append(line)

        question_detail = "\n".join(sections["question_detail"]).strip()

        # 答案 = 解析/解释 + 律师建议
        answer_parts = []
        if sections["analysis"]:
            answer_parts.append("【解析】\n" + "\n".join(sections["analysis"]).strip())
        if sections["advice"]:
            answer_parts.append("【律师建议】\n" + "\n".join(sections["advice"]).strip())

        structured_answer = "\n\n".join(answer_parts).strip() if answer_parts else ""

        return (question_detail, structured_answer)

    # ─── 表格型解析 ───

    def _parse_table(self, text: str) -> List[QAPair]:
        """解析 Markdown 表格型问答"""
        cfg = self._config.get("table_qa", {})
        header_aliases = cfg.get("header_aliases", {})

        # 反向映射：表头文本 → 字段名
        header_map: Dict[str, str] = {}
        for field_name, aliases in header_aliases.items():
            for alias in aliases:
                header_map[alias.strip()] = field_name

        lines = text.splitlines()
        # 找到表格起始行（包含 | 且含问题/答案表头）
        table_lines = [ln for ln in lines if "|" in ln and ln.strip().startswith("|")]

        if len(table_lines) < 2:
            return []

        # 解析表头
        header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
        col_map: Dict[int, str] = {}  # 列索引 → 字段名
        for idx, cell in enumerate(header_cells):
            if cell in header_map:
                col_map[idx] = header_map[cell]

        if "question" not in col_map.values() or "answer" not in col_map.values():
            return []

        pairs: List[QAPair] = []
        # 跳过表头行和分隔行（第二行通常是 |---|---|）
        start = 2
        for line in table_lines[start:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            row: Dict[str, str] = {}
            for idx, cell in enumerate(cells):
                field = col_map.get(idx)
                if field:
                    row[field] = cell

            if row.get("question") and row.get("answer"):
                tags_raw = row.get("tags", "")
                tags = [t.strip() for t in tags_raw.split(",，、;；".replace("", " ")) if t.strip()] if tags_raw else []
                pairs.append(QAPair(
                    question=row["question"],
                    answer=row["answer"],
                    category=row.get("category", ""),
                    tags=tags,
                    source=row.get("source", ""),
                    raw_format="table",
                ))

        return pairs

    # ─── JSON 型解析 ───

    def _parse_json(self, text: str) -> List[QAPair]:
        """解析 JSON 型问答"""
        cfg = self._config.get("json_qa", {})
        q_fields = cfg.get("question_fields", ["question", "q", "问题"])
        a_fields = cfg.get("answer_fields", ["answer", "a", "答案"])
        cat_fields = cfg.get("category_fields", ["category", "分类"])
        src_fields = cfg.get("source_fields", ["source", "来源"])

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        pairs: List[QAPair] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            question = self._get_first_field(item, q_fields)
            answer = self._get_first_field(item, a_fields)
            if not question or not answer:
                continue
            category = self._get_first_field(item, cat_fields) or ""
            source = self._get_first_field(item, src_fields) or ""
            tags_val = item.get("tags") or item.get("tag") or item.get("标签")
            tags = []
            if isinstance(tags_val, list):
                tags = [str(t) for t in tags_val]
            elif isinstance(tags_val, str):
                tags = [t.strip() for t in re.split(r"[,，、;；]", tags_val) if t.strip()]

            pairs.append(QAPair(
                question=str(question).strip(),
                answer=str(answer).strip(),
                category=str(category).strip() if category else "",
                tags=tags,
                source=str(source).strip() if source else "",
                raw_format="json",
            ))

        return pairs

    # ─── 启发式连续段落型解析 ───

    def _parse_heuristic(self, text: str) -> List[QAPair]:
        """启发式解析：以问号结尾的行视为问题，后续段落为答案"""
        cfg = self._config.get("heuristic_qa", {})
        heur = cfg.get("question_heuristics", {})
        min_len = heur.get("min_length", 4)
        max_len = heur.get("max_length", 120)
        must_qmark = heur.get("must_contain_question_mark", True)
        single_line = heur.get("must_be_single_line", True)
        max_answer_lines = cfg.get("answer_max_lines", 80)

        lines = text.splitlines()
        pairs: List[QAPair] = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # 判断是否为问题行
            is_q = (
                min_len <= len(line) <= max_len
                and (not must_qmark or line.endswith("？") or line.endswith("?"))
                and (not single_line or "\n" not in line)
            )

            if not is_q:
                i += 1
                continue

            q_line = i
            q_text = line.rstrip("？?").strip()
            i += 1

            # 收集答案段落
            a_lines: List[str] = []
            a_count = 0
            while i < len(lines) and a_count < max_answer_lines:
                next_line = lines[i].strip()
                # 下一个问题行出现则停止
                if (
                    min_len <= len(next_line) <= max_len
                    and (next_line.endswith("？") or next_line.endswith("?"))
                ):
                    break
                if next_line:
                    a_lines.append(lines[i])
                    a_count += 1
                i += 1

            answer_text = "\n".join(a_lines).strip()
            if answer_text:
                pairs.append(QAPair(
                    question=q_text,
                    answer=answer_text,
                    line_range=(q_line, i - 1),
                    raw_format="heuristic",
                ))

        return pairs

    # ─── 元信息抽取 ───

    def _enrich_metadata(self, pairs: List[QAPair], text: str) -> List[QAPair]:
        """从全文和问答文本中补充元信息"""
        meta_cfg = self._config.get("metadata", {})

        for pair in pairs:
            combined = pair.question + "\n" + pair.answer

            # 分类
            if not pair.category:
                for pat in meta_cfg.get("category_patterns", []):
                    m = re.search(pat, combined)
                    if m:
                        pair.category = m.group(1).strip()
                        break

            # 来源
            if not pair.source:
                for pat in meta_cfg.get("source_patterns", []):
                    m = re.search(pat, combined)
                    if m:
                        pair.source = m.group(1).strip()
                        break

            # 标签
            if not pair.tags:
                for pat in meta_cfg.get("tags_patterns", []):
                    m = re.search(pat, combined)
                    if m:
                        raw_tags = m.group(1)
                        pair.tags = [t.strip() for t in re.split(r"[,，、;；]", raw_tags) if t.strip()]
                        break

        return pairs

    # ─── 过滤与去重 ───

    def _filter(self, pairs: List[QAPair]) -> List[QAPair]:
        """过滤无效问答对并去重"""
        cfg = self._config.get("filters", {})
        min_q = cfg.get("min_question_chars", 2)
        min_a = cfg.get("min_answer_chars", 2)
        discard_pats = cfg.get("discard_patterns", [])
        dedup = cfg.get("dedup_by_question", True)

        seen_questions = set()
        filtered: List[QAPair] = []

        for pair in pairs:
            # 长度过滤
            if len(pair.question) < min_q or len(pair.answer) < min_a:
                continue

            # 丢弃模式过滤
            combined = pair.question + "\n" + pair.answer
            skip = False
            for pat in discard_pats:
                if re.search(pat, combined):
                    skip = True
                    break
            if skip:
                continue

            # 去重
            if dedup:
                q_key = pair.question.strip()
                if q_key in seen_questions:
                    continue
                seen_questions.add(q_key)

            filtered.append(pair)

        return filtered

    # ─── 工具方法 ───

    @staticmethod
    def _safe_int(val: Any) -> Optional[int]:
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _get_first_field(d: Dict, fields: List[str]) -> Optional[str]:
        for f in fields:
            if f in d and d[f]:
                return d[f]
        return None


# ─── CLI 入口 ───

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="问答对解析器")
    parser.add_argument("file", help="问答集文件路径（md/txt/json）")
    parser.add_argument("--format", default="auto",
                        choices=["auto", "explicit", "markdown", "table", "json", "heuristic"],
                        help="问答格式（默认 auto 自动检测）")
    parser.add_argument("--stats", action="store_true", help="只输出统计信息")

    args = parser.parse_args()

    qp = QAParser()
    pairs = qp.parse_file(args.file, fmt=args.format)

    if args.stats:
        print(json.dumps({
            "total_pairs": len(pairs),
            "formats": list(set(p.raw_format for p in pairs)),
            "categories": list(set(p.category for p in pairs if p.category)),
            "with_tags": sum(1 for p in pairs if p.tags),
            "with_source": sum(1 for p in pairs if p.source),
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps([p.to_dict() for p in pairs], ensure_ascii=False, indent=2))
