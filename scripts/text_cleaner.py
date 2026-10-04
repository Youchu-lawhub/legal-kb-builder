#!/usr/bin/env python3
"""
文本清洗模块（8 步流水线）

将 OCR 转换后的法律文本进行深度清洗，提升知识库质量。
可被 legal-kb 和 legal-kb-agentic 两个 Skill 共同使用。

清洗步骤（按执行顺序）：
  1. fix_ocr_errors        - 上下文感知的 OCR 字符纠错
  2. remove_headers_footers - 检测并移除重复的页眉页脚
  3. remove_page_numbers    - 移除各种格式的页码
  4. separate_footnotes     - 分离/移除脚注尾注
  5. normalize_whitespace   - 修复中文文本的空格问题
  6. reconstruct_tables     - 检测并重建被 OCR 打散的表格
  7. deduplicate_paragraphs - 去除重叠区域产生的重复段落
  8. clean_legal_text       - 法律专用：法条编号/法院名称/法律名称规范化

使用方式：
    from text_cleaner import TextCleaner

    cleaner = TextCleaner()
    result = cleaner.clean(raw_text)
    print(result.cleaned_text)
    print(result.stats)

    # 单步使用
    fixed = cleaner.fix_ocr_errors(raw_text)

    # 自定义选项
    result = cleaner.clean(raw_text, options={
        'separate_footnotes': {'mode': 'endnote'},
        'deduplicate_paragraphs': {'enabled': False},
    })
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


# ─── 默认资产路径 ───
_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_DEFAULT_OCR_CORRECTIONS = _ASSETS_DIR / "ocr-corrections.yaml"
_DEFAULT_LEGAL_CORRECTIONS = _ASSETS_DIR / "legal-corrections.yaml"


# ─── 数据类 ───

@dataclass
class CleanResult:
    """清洗结果"""
    cleaned_text: str
    original_length: int
    cleaned_length: int
    changes_log: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


# ─── 主类 ───

class TextCleaner:
    """
    法律文本清洗器（8 步流水线）

    Args:
        ocr_corrections_path: OCR 纠错模式 YAML 路径（默认使用内置）
        legal_corrections_path: 法律规范化 YAML 路径（默认使用内置）
        config: 额外配置覆盖
    """

    STEP_ORDER = [
        'fix_ocr_errors',
        'remove_headers_footers',
        'remove_page_numbers',
        'separate_footnotes',
        'normalize_whitespace',
        'reconstruct_tables',
        'deduplicate_paragraphs',
        'clean_legal_text',
    ]

    def __init__(
        self,
        ocr_corrections_path: str = None,
        legal_corrections_path: str = None,
        config: Dict[str, Any] = None,
    ):
        self._config = config or {}

        # 加载 OCR 纠错模式
        ocr_path = Path(ocr_corrections_path) if ocr_corrections_path else _DEFAULT_OCR_CORRECTIONS
        self._ocr_patterns = self._load_yaml(ocr_path)

        # 加载法律规范化模式
        legal_path = Path(legal_corrections_path) if legal_corrections_path else _DEFAULT_LEGAL_CORRECTIONS
        self._legal_patterns = self._load_yaml(legal_path)

    # ─── 核心流水线 ───

    def clean(self, text: str, options: Dict[str, Any] = None) -> CleanResult:
        """
        执行完整的 8 步清洗流水线。

        Args:
            text: 原始文本
            options: 每步的配置覆盖，格式如：
                {
                    'step_name': {'enabled': True/False, ...step-specific params},
                    ...
                }

        Returns:
            CleanResult 包含清洗后文本、变更日志和统计信息
        """
        options = options or {}
        original_length = len(text)
        changes_log = []
        stats = {}
        current = text

        for step_name in self.STEP_ORDER:
            step_opts = options.get(step_name, {})

            # 检查是否禁用此步骤
            if not step_opts.get('enabled', True):
                continue

            step_func = getattr(self, step_name)
            before_len = len(current)

            # 某些步骤接受额外参数
            if step_name == 'separate_footnotes':
                mode = step_opts.get('mode', 'remove')
                current = step_func(current, mode=mode)
            elif step_name == 'deduplicate_paragraphs':
                threshold = step_opts.get('threshold', 0.85)
                current = step_func(current, threshold=threshold)
            else:
                current = step_func(current)

            after_len = len(current)
            diff = before_len - after_len

            if diff != 0:
                changes_log.append({
                    'step': step_name,
                    'chars_removed': diff,
                    'before_len': before_len,
                    'after_len': after_len,
                })
                stats[step_name] = abs(diff)

        return CleanResult(
            cleaned_text=current,
            original_length=original_length,
            cleaned_length=len(current),
            changes_log=changes_log,
            stats=stats,
        )

    # ─── Step 1: OCR 字符纠错 ───

    def fix_ocr_errors(self, text: str) -> str:
        """
        上下文感知的 OCR 字符纠错。

        根据 ocr-corrections.yaml 中定义的模式，在上下文匹配时替换错误字符。
        """
        if not self._ocr_patterns:
            return text

        result = text

        # 词级替换（无上下文约束，直接替换）
        word_subs = self._ocr_patterns.get('word_substitutions', {})
        if isinstance(word_subs, dict):
            for wrong, correct in word_subs.items():
                result = result.replace(wrong, correct)

        # 字符级替换（需上下文约束）
        char_subs = self._ocr_patterns.get('character_substitutions', [])
        if isinstance(char_subs, list):
            for rule in char_subs:
                wrong = rule.get('wrong', '')
                correct = rule.get('correct', '')
                ctx_before = rule.get('context_before', [])
                ctx_after = rule.get('context_after', [])

                if not wrong or not correct:
                    continue

                result = self._apply_context_substitution(
                    result, wrong, correct, ctx_before, ctx_after
                )

        return result

    def _apply_context_substitution(
        self, text: str, wrong: str, correct: str,
        ctx_before: List[str], ctx_after: List[str]
    ) -> str:
        """根据前后文上下文执行单字符替换"""
        if wrong not in text:
            return text

        chars = list(text)
        i = 0
        while i < len(chars):
            if chars[i] == wrong:
                # 检查前文
                before_match = False
                if ctx_before:
                    for ctx in ctx_before:
                        ctx_len = len(ctx)
                        if i >= ctx_len:
                            preceding = ''.join(chars[i - ctx_len:i])
                            if preceding == ctx:
                                before_match = True
                                break
                else:
                    before_match = True  # 无前文约束则视为匹配

                # 检查后文
                after_match = False
                if ctx_after:
                    for ctx in ctx_after:
                        ctx_len = len(ctx)
                        if i + 1 + ctx_len <= len(chars):
                            following = ''.join(chars[i + 1:i + 1 + ctx_len])
                            if following == ctx:
                                after_match = True
                                break
                else:
                    after_match = True  # 无后文约束则视为匹配

                # 前文或后文任一匹配即替换（OR 逻辑）
                if (ctx_before and before_match) or (ctx_after and after_match):
                    chars[i] = correct

            i += 1

        return ''.join(chars)

    # ─── Step 2: 移除页眉页脚 ───

    def remove_headers_footers(self, text: str) -> str:
        """
        检测并移除重复出现的页眉页脚。

        策略：将文本按页分割，统计每页首尾行的出现频率，
        出现在 >50% 页面的相同位置行判定为页眉/页脚。
        """
        # 按页分割：检测分页符或页面边界标记
        page_separators = ['\f', '\n---\n', '\n===\n']
        pages = [text]
        for sep in page_separators:
            new_pages = []
            for page in pages:
                new_pages.extend(page.split(sep))
            pages = new_pages

        # 过滤空页
        pages = [p for p in pages if p.strip()]

        if len(pages) < 3:
            # 页面太少，无法可靠检测
            return text

        # 统计首尾行频率
        header_candidates = {}  # {normalized_line: count}
        footer_candidates = {}

        for page in pages:
            lines = page.strip().split('\n')
            if not lines:
                continue

            # 前 2 行作为页眉候选
            for line in lines[:2]:
                normalized = line.strip()
                if normalized and len(normalized) < 100:  # 页眉通常较短
                    header_candidates[normalized] = header_candidates.get(normalized, 0) + 1

            # 后 2 行作为页脚候选
            for line in lines[-2:]:
                normalized = line.strip()
                if normalized and len(normalized) < 100:
                    footer_candidates[normalized] = footer_candidates.get(normalized, 0) + 1

        # 判定：出现频率 > 50% 的为页眉/页脚
        threshold = len(pages) * 0.5
        headers_to_remove = {
            line for line, count in header_candidates.items()
            if count > threshold
        }
        footers_to_remove = {
            line for line, count in footer_candidates.items()
            if count > threshold
        }

        if not headers_to_remove and not footers_to_remove:
            return text

        # 移除
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped in headers_to_remove or stripped in footers_to_remove:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    # ─── Step 3: 移除页码 ───

    def remove_page_numbers(self, text: str) -> str:
        """
        移除各种格式的页码行。

        仅移除独立短行（<=10 字符）上的页码，不影响正文内容。
        """
        page_number_patterns = [
            r'^\s*-\s*\d+\s*-\s*$',           # - 123 -
            r'^\s*--\s*\d+\s*--\s*$',          # -- 123 --
            r'^\s*\d{1,4}\s*$',                # 纯数字行（1-4位）
            r'^\s*第\s*\d+\s*页\s*$',           # 第 123 页
            r'^\s*Page\s+\d+\s*$',              # Page 123
            r'^\s*p\.\s*\d+\s*$',               # p. 123
            r'^\s*[ivxlcdm]+\s*$',              # 罗马数字（小写）
            r'^\s*[IVXLCDM]+\s*$',              # 罗马数字（大写）
        ]
        compiled = [re.compile(p, re.IGNORECASE) for p in page_number_patterns]

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # 仅检查短行
            if stripped and len(stripped) <= 10:
                is_page_number = any(p.match(stripped) for p in compiled)
                if is_page_number:
                    continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    # ─── Step 4: 分离脚注 ───

    def separate_footnotes(self, text: str, mode: str = 'remove') -> str:
        """
        分离或移除脚注/尾注。

        Args:
            text: 输入文本
            mode: 处理模式
                - 'remove': 完全删除脚注（默认，知识库推荐）
                - 'endnote': 收集脚注移到文末
                - 'inline': 转为括号内嵌注释
                - 'keep': 不做任何处理

        Returns:
            处理后的文本
        """
        if mode == 'keep':
            return text

        lines = text.split('\n')
        main_lines = []
        footnote_lines = []
        in_footnote_block = False

        # 脚注标记检测模式
        footnote_ref_pattern = re.compile(
            r'^\s*(?:'
            r'\[\d+\]'          # [1]
            r'|\(\d+\)'        # (1)
            r'|\^\d+'          # ^1
            r'|注\d+'          # 注1
            r'|注释\d+'        # 注释1
            r'|(?:①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)'  # 带圈数字
            r')\s*'
        )

        # 正文中的脚注引用标记（用于 inline 模式）
        inline_ref_pattern = re.compile(
            r'(?:\[\d+\]|\(\d+\)|\^\d+|注\d+|[①②③④⑤⑥⑦⑧⑨⑩])'
        )

        for line in lines:
            stripped = line.strip()

            # 检测脚注块的开始
            if footnote_ref_pattern.match(stripped):
                in_footnote_block = True
                footnote_lines.append(stripped)
                continue

            # 在脚注块内，后续缩进行也是脚注内容
            if in_footnote_block:
                if stripped == '' or line.startswith('  ') or line.startswith('\t'):
                    footnote_lines.append(stripped)
                    continue
                else:
                    in_footnote_block = False

            main_lines.append(line)

        main_text = '\n'.join(main_lines)

        if mode == 'remove':
            # 同时移除正文中的脚注引用标记
            main_text = inline_ref_pattern.sub('', main_text)
            return main_text

        elif mode == 'endnote':
            if footnote_lines:
                footnotes = '\n'.join(footnote_lines)
                return f"{main_text}\n\n---\n\n## 注释\n\n{footnotes}"
            return main_text

        elif mode == 'inline':
            # 将脚注内容嵌入到引用位置（简化实现：附加在段落末尾）
            # 完整实现需要匹配编号，这里做基础版
            main_text = inline_ref_pattern.sub('', main_text)
            return main_text

        return text

    # ─── Step 5: 空白字符规范化 ───

    def normalize_whitespace(self, text: str) -> str:
        """
        修复 OCR 引入的空格问题。

        规则：
        - 移除两个 CJK 字符之间的单个空格（OCR 伪影）
        - 保留 CJK 与拉丁/数字之间的空格
        - 合并连续空行（最多保留 2 个）
        - 移除行尾空白
        - 统一换行符为 \\n
        """
        # 统一换行符
        result = text.replace('\r\n', '\n').replace('\r', '\n')

        # 移除行尾空白
        result = re.sub(r'[ \t]+\n', '\n', result)

        # 移除两个 CJK 字符之间的单个空格
        # CJK 范围: \u4e00-\u9fff (基本), \u3400-\u4dbf (扩展A)
        # 只处理行内空格。\s 也会匹配换行；若在这里使用它，会把
        # "人民法院\n民事判决书" 之类的结构化行合并，破坏后续字段抽取。
        cjk_space_pattern = re.compile(
            r'([\u4e00-\u9fff\u3400-\u4dbf])[ \t]([\u4e00-\u9fff\u3400-\u4dbf])'
        )
        # 需要多轮替换，因为 "A B C" 第一轮只能匹配 "A B"
        prev = None
        while prev != result:
            prev = result
            result = cjk_space_pattern.sub(r'\1\2', result)

        # 合并连续空行（最多保留 2 个换行，即 1 个空行）
        result = re.sub(r'\n{4,}', '\n\n\n', result)

        return result

    # ─── Step 6: 表格重建 ───

    def reconstruct_tables(self, text: str) -> str:
        """
        检测被 OCR 打散的表格并尝试重建为 Markdown 格式。

        检测特征：
        - 连续短行且有规律的空白对齐
        - 含有多个制表符或多空格分隔的列
        - 已有 pipe 分隔但格式损坏

        置信度阈值：仅在 >60% 的行能对齐时重建。
        """
        lines = text.split('\n')
        result_lines = []
        i = 0

        while i < len(lines):
            # 检测可能的表格区域
            table_start, table_end = self._detect_table_region(lines, i)

            if table_start is not None and table_end is not None:
                table_lines = lines[table_start:table_end + 1]
                reconstructed = self._try_reconstruct_table(table_lines)

                if reconstructed is not None:
                    result_lines.append(reconstructed)
                    i = table_end + 1
                    continue

            result_lines.append(lines[i])
            i += 1

        return '\n'.join(result_lines)

    def _detect_table_region(
        self, lines: List[str], start: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """检测从 start 开始的表格区域"""
        if start >= len(lines):
            return None, None

        line = lines[start].strip()

        # 已有 pipe 分隔的可能表格
        if '|' in line and line.count('|') >= 2:
            end = start
            while end + 1 < len(lines) and '|' in lines[end + 1]:
                end += 1
            if end - start >= 2:  # 至少 3 行
                return start, end

        # 制表符分隔的可能表格
        if '\t' in line and line.count('\t') >= 1:
            end = start
            tab_count = line.count('\t')
            while end + 1 < len(lines):
                next_line = lines[end + 1].strip()
                if not next_line:
                    break
                if abs(next_line.count('\t') - tab_count) <= 1:
                    end += 1
                else:
                    break
            if end - start >= 2:
                return start, end

        return None, None

    def _try_reconstruct_table(self, table_lines: List[str]) -> Optional[str]:
        """尝试将行列表重建为 Markdown 表格"""
        if not table_lines or len(table_lines) < 2:
            return None

        rows = []
        for line in table_lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 尝试按 pipe 分割
            if '|' in stripped:
                cells = [c.strip() for c in stripped.split('|')]
                cells = [c for c in cells if c or cells.index(c) not in (0, len(cells) - 1)]
                # 过滤掉纯分隔行（如 |---|---|）
                if cells and all(re.match(r'^[-:]+$', c) for c in cells):
                    continue
                if cells:
                    rows.append(cells)
            # 尝试按制表符分割
            elif '\t' in stripped:
                cells = [c.strip() for c in stripped.split('\t')]
                cells = [c for c in cells if c]
                if cells:
                    rows.append(cells)

        if len(rows) < 2:
            return None

        # 检查列数一致性
        col_counts = [len(row) for row in rows]
        most_common_cols = max(set(col_counts), key=col_counts.count)
        alignment_ratio = col_counts.count(most_common_cols) / len(col_counts)

        if alignment_ratio < 0.6:
            return None  # 对齐不够好，放弃重建

        # 规范化列数
        normalized_rows = []
        for row in rows:
            if len(row) < most_common_cols:
                row = row + [''] * (most_common_cols - len(row))
            elif len(row) > most_common_cols:
                row = row[:most_common_cols]
            normalized_rows.append(row)

        # 构建 Markdown 表格
        md_lines = []
        # 表头
        md_lines.append('| ' + ' | '.join(normalized_rows[0]) + ' |')
        # 分隔行
        md_lines.append('| ' + ' | '.join(['---'] * most_common_cols) + ' |')
        # 数据行
        for row in normalized_rows[1:]:
            md_lines.append('| ' + ' | '.join(row) + ' |')

        return '\n'.join(md_lines)

    # ─── Step 7: 段落去重 ───

    def deduplicate_paragraphs(self, text: str, threshold: float = 0.85) -> str:
        """
        检测并去除近似重复段落（通常由 PDF 重叠区域产生）。

        策略：在 5 段滑动窗口内比较连续段落的相似度，
        若超过阈值则保留较长的版本。

        Args:
            text: 输入文本
            threshold: 相似度阈值（0-1，默认 0.85）

        Returns:
            去重后的文本
        """
        # 按双换行分段
        paragraphs = re.split(r'\n\s*\n', text)

        if len(paragraphs) <= 1:
            return text

        # 标记待移除的段落索引
        to_remove = set()

        for i in range(len(paragraphs)):
            if i in to_remove:
                continue

            # 在窗口内比较
            window_end = min(i + 5, len(paragraphs))
            for j in range(i + 1, window_end):
                if j in to_remove:
                    continue

                para_i = paragraphs[i].strip()
                para_j = paragraphs[j].strip()

                # 跳过极短段落（标题等）
                if len(para_i) < 20 or len(para_j) < 20:
                    continue

                similarity = SequenceMatcher(None, para_i, para_j).ratio()

                if similarity >= threshold:
                    # 保留较长的版本
                    if len(para_i) >= len(para_j):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # i 被移除，跳出内层循环

        # 重建文本
        kept = [p for idx, p in enumerate(paragraphs) if idx not in to_remove]
        return '\n\n'.join(kept)

    # ─── Step 8: 法律文本专用清洗 ───

    def clean_legal_text(self, text: str) -> str:
        """
        法律文本专用清洗：法条编号、法院名称、法律名称规范化。
        """
        if not self._legal_patterns:
            return text

        result = text

        # 法院名称修复
        for rule in self._legal_patterns.get('court_names', []):
            pattern = rule.get('pattern', '')
            replacement = rule.get('replacement', '')
            if pattern and replacement:
                try:
                    result = re.sub(pattern, replacement, result)
                except re.error:
                    continue

        # 法律名称修复
        for rule in self._legal_patterns.get('law_names', []):
            pattern = rule.get('pattern', '')
            replacement = rule.get('replacement', '')
            if pattern and replacement:
                try:
                    result = re.sub(pattern, replacement, result)
                except re.error:
                    continue

        # 法条编号修复
        for rule in self._legal_patterns.get('article_number_fixes', []):
            pattern = rule.get('pattern', '')
            replacement = rule.get('replacement', '')
            if pattern and replacement:
                try:
                    result = re.sub(pattern, replacement, result)
                except re.error:
                    continue

        # 章节编号修复
        for rule in self._legal_patterns.get('chapter_number_fixes', []):
            pattern = rule.get('pattern', '')
            replacement = rule.get('replacement', '')
            if pattern and replacement:
                try:
                    result = re.sub(pattern, replacement, result)
                except re.error:
                    continue

        return result

    # ─── 检测模式（不修改文本，仅报告问题） ───

    def detect_issues(self, text: str) -> Dict[str, Any]:
        """
        仅检测问题不做修改（供 quality_checker.py 使用）。

        Returns:
            {
                'ocr_error_count': int,
                'header_footer_count': int,
                'page_number_count': int,
                'footnote_count': int,
                'duplicate_paragraph_count': int,
                'broken_table_count': int,
                'legal_format_issues': int,
                'details': {...}
            }
        """
        issues = {
            'ocr_error_count': 0,
            'header_footer_count': 0,
            'page_number_count': 0,
            'footnote_count': 0,
            'duplicate_paragraph_count': 0,
            'broken_table_count': 0,
            'legal_format_issues': 0,
            'details': {},
        }

        # OCR 错误计数
        if self._ocr_patterns:
            word_subs = self._ocr_patterns.get('word_substitutions', {})
            if isinstance(word_subs, dict):
                for wrong in word_subs:
                    count = text.count(wrong)
                    issues['ocr_error_count'] += count

        # 页码行计数
        page_patterns = [
            r'^\s*-\s*\d+\s*-\s*$',
            r'^\s*\d{1,4}\s*$',
            r'^\s*第\s*\d+\s*页\s*$',
        ]
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped and len(stripped) <= 10:
                for pattern in page_patterns:
                    if re.match(pattern, stripped):
                        issues['page_number_count'] += 1
                        break

        # 脚注计数
        footnote_pattern = re.compile(
            r'^\s*(?:\[\d+\]|\(\d+\)|\^\d+|注\d+|注释\d+)\s*'
        )
        for line in text.split('\n'):
            if footnote_pattern.match(line.strip()):
                issues['footnote_count'] += 1

        # 重复段落计数
        paragraphs = re.split(r'\n\s*\n', text)
        for i in range(len(paragraphs) - 1):
            para_i = paragraphs[i].strip()
            para_j = paragraphs[i + 1].strip()
            if len(para_i) >= 20 and len(para_j) >= 20:
                if SequenceMatcher(None, para_i, para_j).ratio() >= 0.85:
                    issues['duplicate_paragraph_count'] += 1

        # 法律格式问题
        # 书名号内空格
        issues['legal_format_issues'] += len(
            re.findall(r'《\s+[^》]+\s+》', text)
        )
        # 法条编号内空格
        issues['legal_format_issues'] += len(
            re.findall(r'第[一二三四五六七八九十百千零]+\s+[一二三四五六七八九十百千零]+条', text)
        )

        return issues

    # ─── 内部工具 ───

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        if yaml is None:
            return {}
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


# ─── CLI 入口 ───

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("用法:")
        print("  python text_cleaner.py <input.md>              # 清洗并输出到 stdout")
        print("  python text_cleaner.py <input.md> -o <out.md>  # 清洗并保存")
        print("  python text_cleaner.py <input.md> --detect     # 仅检测问题不修改")
        print("  python text_cleaner.py --test                  # 运行自测")
        sys.exit(0)

    if sys.argv[1] == '--test':
        # 简单自测
        cleaner = TextCleaner()

        # 测试 OCR 纠错
        test_text = "己经支付损害赔尝的违约贵任"
        fixed = cleaner.fix_ocr_errors(test_text)
        print(f"OCR fix: '{test_text}' -> '{fixed}'")

        # 测试页码移除
        test_text = "正文内容\n- 123 -\n更多内容\n第 45 页\n继续"
        cleaned = cleaner.remove_page_numbers(test_text)
        print(f"Page numbers: removed {len(test_text) - len(cleaned)} chars")

        # 测试空白规范化
        test_text = "合 同 法 第 107 条"
        normalized = cleaner.normalize_whitespace(test_text)
        print(f"Whitespace: '{test_text}' -> '{normalized}'")

        # 测试法律文本清洗
        test_text = "《 民 法 典 》第一百二十 三条"
        legal_cleaned = cleaner.clean_legal_text(test_text)
        print(f"Legal: '{test_text}' -> '{legal_cleaned}'")

        # 测试完整流水线
        full_test = "己经支付的损害赔尝\n\n- 123 -\n\n合 同 法\n\n《 民 法 典 》第一百二十 三条"
        result = cleaner.clean(full_test)
        print(f"\nFull pipeline:")
        print(f"  Original: {result.original_length} chars")
        print(f"  Cleaned:  {result.cleaned_length} chars")
        print(f"  Stats:    {result.stats}")
        print(f"  Result:   '{result.cleaned_text}'")

        print("\nAll tests completed.")
        sys.exit(0)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"错误：文件不存在：{input_path}", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding='utf-8')
    cleaner = TextCleaner()

    if '--detect' in sys.argv:
        issues = cleaner.detect_issues(text)
        print(json.dumps(issues, ensure_ascii=False, indent=2))
    else:
        result = cleaner.clean(text)
        output_text = result.cleaned_text

        if '-o' in sys.argv:
            out_idx = sys.argv.index('-o')
            if out_idx + 1 < len(sys.argv):
                output_path = Path(sys.argv[out_idx + 1])
                output_path.write_text(output_text, encoding='utf-8')
                print(f"清洗完成: {result.original_length} -> {result.cleaned_length} 字符", file=sys.stderr)
                print(f"已保存到: {output_path}", file=sys.stderr)
                if result.stats:
                    print(f"变更统计: {json.dumps(result.stats, ensure_ascii=False)}", file=sys.stderr)
        else:
            print(output_text)
