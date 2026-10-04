#!/usr/bin/env python3
"""
裁判文书要素抽取模块

从法院裁判文书（判决书/裁定书/调解书）中提取结构化要素：
  案号、法院、程序、当事人、案由、诉讼请求、
  查明事实、本院认为、判决结果、适用法律、争议焦点

使用方式：
    from judgment_parser import JudgmentParser

    parser = JudgmentParser()
    elements = parser.parse(judgment_text)
    print(elements.case_number)
    print(elements.court_reasoning)  # 最有价值的部分
"""

import re
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# 共享模块与本文件同目录，将 scripts/ 加入 sys.path
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from text_cleaner import TextCleaner
    HAS_CLEANER = True
except ImportError:
    HAS_CLEANER = False


_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_DEFAULT_PATTERNS = _ASSETS_DIR / "element-patterns.yaml"


@dataclass
class Party:
    """当事人"""
    role: str = ""          # 原告/被告/第三人/上诉人/...
    name: str = ""
    representative: Optional[str] = None
    attorney: Optional[str] = None


@dataclass
class JudgmentElements:
    """裁判文书结构化要素"""
    case_number: str = ""
    court: str = ""
    procedure: str = ""             # 一审/二审/再审
    document_type: str = ""         # 判决书/裁定书/调解书
    parties: List[Party] = field(default_factory=list)
    cause_of_action: str = ""
    claims: List[str] = field(default_factory=list)
    found_facts: str = ""           # 经审理查明
    court_reasoning: str = ""       # 本院认为（最有价值）
    judgment_result: str = ""       # 判决/裁定如下
    applied_laws: List[str] = field(default_factory=list)
    dispute_focus: Optional[str] = None
    raw_text: str = ""
    parse_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['parties'] = [asdict(p) for p in self.parties]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class JudgmentParser:
    """
    裁判文书解析器

    Args:
        patterns_path: 抽取模式 YAML 路径（默认使用内置）
        clean_first: 是否先用 TextCleaner 清洗文本
    """

    def __init__(self, patterns_path: str = None, clean_first: bool = True):
        path = Path(patterns_path) if patterns_path else _DEFAULT_PATTERNS
        self._patterns = self._load_yaml(path)
        self._cleaner = TextCleaner() if (clean_first and HAS_CLEANER) else None

    def parse(self, text: str) -> JudgmentElements:
        """
        完整解析裁判文书。

        Args:
            text: 裁判文书原文

        Returns:
            JudgmentElements 结构化要素
        """
        # 可选：先清洗 OCR 错误
        if self._cleaner:
            result = self._cleaner.clean(text, options={
                'remove_headers_footers': {'enabled': True},
                'remove_page_numbers': {'enabled': True},
                'separate_footnotes': {'enabled': False},  # 判决书通常没有脚注
                'reconstruct_tables': {'enabled': False},
                'deduplicate_paragraphs': {'enabled': True},
            })
            clean_text = result.cleaned_text
        else:
            clean_text = text

        elements = JudgmentElements(raw_text=text)

        # 按顺序提取各要素
        # 案号与法院位于文书首部，必须保留原始行边界。清洗后的正文仍用于
        # 长段落提取；首部字段则优先从原文提取，以兼容 OCR 清洗策略。
        elements.case_number = self._extract_case_number(text) or self._extract_case_number(clean_text)
        elements.court = self._extract_court(text) or self._extract_court(clean_text)
        elements.procedure = self._extract_procedure(clean_text, elements.case_number)
        elements.document_type = self._extract_document_type(clean_text)
        elements.parties = self._extract_parties(clean_text)
        elements.cause_of_action = self._extract_cause_of_action(clean_text)
        elements.claims = self._extract_claims(clean_text)
        elements.found_facts = self._extract_section(clean_text, 'found_facts')
        elements.court_reasoning = self._extract_section(clean_text, 'court_reasoning')
        elements.judgment_result = self._extract_section(clean_text, 'judgment_result')
        elements.applied_laws = self._extract_applied_laws(clean_text)
        elements.dispute_focus = self._extract_dispute_focus(clean_text)

        # 计算整体解析置信度
        elements.parse_confidence = self._calculate_confidence(elements)

        return elements

    # ─── 要素提取方法 ───

    def _extract_case_number(self, text: str) -> str:
        """提取案号"""
        patterns = self._patterns.get('case_number', {}).get('patterns', [])
        if not patterns:
            patterns = [r'[（(](\d{4})[）)]\s*[\u4e00-\u9fa5]+[\u4e00-\u9fa5\d]*\d+号']

        # 只在前 20 行中查找
        lines = text.split('\n')[:20]
        header = '\n'.join(lines)

        for pattern in patterns:
            match = re.search(pattern, header)
            if match:
                return match.group(0)
        return ""

    def _extract_court(self, text: str) -> str:
        """提取审理法院"""
        lines = text.split('\n')[:10]
        for line in lines:
            stripped = line.strip()
            if re.match(r'^[\u4e00-\u9fa5]+(?:人民法院|仲裁委员会)\s*$', stripped):
                return stripped
        return ""

    def _extract_procedure(self, text: str, case_number: str) -> str:
        """从案号和文本特征判断审判程序"""
        indicators = self._patterns.get('procedure', {}).get('indicators', {})

        for proc_name, keywords in indicators.items():
            for kw in keywords:
                if kw in case_number:
                    return proc_name

        # 降级：在文本前部查找
        header = text[:2000]
        for proc_name, keywords in indicators.items():
            for kw in keywords:
                if kw in header:
                    return proc_name

        return "一审"  # 默认

    def _extract_document_type(self, text: str) -> str:
        """判断文书类型"""
        header = text[:500]
        if '判决书' in header:
            return '判决书'
        elif '裁定书' in header:
            return '裁定书'
        elif '调解书' in header:
            return '调解书'
        elif '决定书' in header:
            return '决定书'
        # 根据内容推断
        if '判决如下' in text:
            return '判决书'
        elif '裁定如下' in text:
            return '裁定书'
        return '判决书'

    def _extract_parties(self, text: str) -> List[Party]:
        """提取当事人信息"""
        parties = []
        roles = self._patterns.get('party_roles', [
            '原告', '被告', '第三人', '上诉人', '被上诉人',
            '申请人', '被申请人'
        ])

        lines = text.split('\n')

        for i, line in enumerate(lines[:80]):  # 当事人信息通常在前 80 行
            stripped = line.strip()
            for role in roles:
                if stripped.startswith(role):
                    # 提取角色后的名称
                    name_part = stripped[len(role):].strip()
                    # 移除冒号等分隔符
                    name_part = re.sub(r'^[：:（(]\s*', '', name_part)
                    name_part = re.sub(r'[）)]\s*$', '', name_part)
                    # 截取到逗号或句号
                    name_match = re.match(r'^([^，。,.\s]+)', name_part)
                    name = name_match.group(1) if name_match else name_part

                    if name and len(name) <= 30:
                        party = Party(role=role, name=name)

                        # 查找下一行的代理人信息
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if '委托' in next_line and '代理人' in next_line:
                                atty_match = re.search(r'代理人[：:]\s*(.+)', next_line)
                                if atty_match:
                                    party.attorney = atty_match.group(1).strip()

                        parties.append(party)
                    break

        return parties

    def _extract_cause_of_action(self, text: str) -> str:
        """提取案由"""
        # 显式案由标记
        match = re.search(r'案由[：:]\s*(.+)', text[:3000])
        if match:
            return match.group(1).strip()

        # 从文书标题推断（"...纠纷" 模式）
        lines = text.split('\n')[:20]
        for line in lines:
            match = re.search(r'([\u4e00-\u9fa5]+纠纷)', line)
            if match:
                return match.group(1)

        # 从"一案"前的短语推断
        match = re.search(r'([\u4e00-\u9fa5]+)一案', text[:3000])
        if match:
            cause = match.group(1).strip()
            if len(cause) <= 20:
                return cause

        return ""

    def _extract_claims(self, text: str) -> List[str]:
        """提取诉讼请求"""
        claims = []

        # 查找诉讼请求段落
        match = re.search(
            r'(?:诉讼请求|诉称|请求)[：:]\s*(.+?)(?=经审理查明|被告|辩称|本院)',
            text, re.DOTALL
        )
        if match:
            claims_text = match.group(1)
            # 按编号分割
            items = re.split(r'[；;]\s*|\n\s*\d+[、.．]\s*', claims_text)
            claims = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]

        return claims[:10]  # 最多 10 条

    def _extract_section(self, text: str, section_name: str) -> str:
        """
        按段落标记提取特定段落。

        使用 section_markers 配置中的 start/end 标记定位。
        """
        markers = self._patterns.get('section_markers', {}).get(section_name, {})
        start_markers = markers.get('start', [])
        end_markers = markers.get('end', [])

        if not start_markers:
            return ""

        # 找到起始位置
        start_pos = -1
        for marker in start_markers:
            match = re.search(re.escape(marker) if not any(c in marker for c in '.+*?[]()\\') else marker, text)
            if match:
                start_pos = match.end()
                break

        if start_pos < 0:
            return ""

        # 找到结束位置
        end_pos = len(text)
        remaining = text[start_pos:]
        for marker in end_markers:
            try:
                match = re.search(marker if any(c in marker for c in '.+*?[]()\\') else re.escape(marker), remaining)
                if match:
                    candidate = start_pos + match.start()
                    if candidate < end_pos:
                        end_pos = candidate
            except re.error:
                # 某些模式可能有语法错误，跳过
                continue

        section_text = text[start_pos:end_pos].strip()

        # 清理前导标点
        section_text = re.sub(r'^[，。：:；;\s]+', '', section_text)

        return section_text

    def _extract_applied_laws(self, text: str) -> List[str]:
        """提取适用的法律条文"""
        pattern = self._patterns.get(
            'law_citation_pattern',
            r'《([^》]+)》[\s]*第[一二三四五六七八九十百千零\d]+条'
        )

        citations = []
        seen = set()

        for match in re.finditer(pattern, text):
            citation = match.group(0)
            if citation not in seen:
                citations.append(citation)
                seen.add(citation)

        return citations

    def _extract_dispute_focus(self, text: str) -> Optional[str]:
        """提取争议焦点"""
        markers = self._patterns.get('dispute_focus_markers', [
            '本案争议焦点', '双方争议焦点', '本案的争议焦点'
        ])

        for marker in markers:
            match = re.search(
                re.escape(marker) + r'[：:为是]?\s*(.+?)(?:\n\n|\n[一二三四五六七八九十])',
                text, re.DOTALL
            )
            if match:
                focus = match.group(1).strip()
                if len(focus) > 10:
                    return focus[:500]

        return None

    def _calculate_confidence(self, elements: JudgmentElements) -> float:
        """计算解析置信度"""
        score = 0.0
        total_weight = 0.0

        checks = [
            (bool(elements.case_number), 2.0),
            (bool(elements.court), 1.0),
            (bool(elements.parties), 1.5),
            (bool(elements.cause_of_action), 1.0),
            (bool(elements.found_facts), 2.0),
            (bool(elements.court_reasoning), 3.0),  # 最重要
            (bool(elements.judgment_result), 2.0),
            (bool(elements.applied_laws), 1.0),
            (len(elements.court_reasoning) > 100, 1.5),
            (len(elements.found_facts) > 50, 1.0),
        ]

        for passed, weight in checks:
            total_weight += weight
            if passed:
                score += weight

        return round(score / total_weight, 2) if total_weight > 0 else 0.0

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


# ─── CLI ───

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python judgment_parser.py <judgment.txt>   # 解析并输出 JSON")
        print("  python judgment_parser.py --test           # 自测")
        sys.exit(0)

    if sys.argv[1] == '--test':
        parser = JudgmentParser(clean_first=False)

        test_text = """北京市海淀区人民法院
民事判决书
（2023）京0108民初12345号

原告：张三，男，1980年1月1日出生。
委托代理人：王律师，北京某律师事务所律师。

被告：李四，男，1975年5月5日出生。

原告张三与被告李四买卖合同纠纷一案，本院受理后，依法组成合议庭公开审理了本案。

诉讼请求：1、判令被告支付货款人民币50万元；2、判令被告承担诉讼费用。

经审理查明：2022年1月，原告与被告签订买卖合同一份，约定被告向原告购买设备一批，合同总价款为人民币50万元。原告已按约交付货物，被告至今未支付货款。

以上事实，有买卖合同、送货单、催款函等证据予以证实。

本院认为，原告与被告之间的买卖合同关系合法有效。被告未按约定支付货款，构成违约，应当承担违约责任。依照《中华人民共和国民法典》第五百七十七条、第五百七十九条之规定。

判决如下：
一、被告李四于本判决生效之日起十日内支付原告张三货款人民币50万元。
二、驳回原告张三的其他诉讼请求。

审判长  赵法官
审判员  钱法官
人民陪审员  孙陪审

二〇二三年六月十五日
书记员  周书记
"""

        elements = parser.parse(test_text)
        print(f"案号: {elements.case_number}")
        print(f"法院: {elements.court}")
        print(f"程序: {elements.procedure}")
        print(f"文书类型: {elements.document_type}")
        print(f"当事人: {[(p.role, p.name) for p in elements.parties]}")
        print(f"案由: {elements.cause_of_action}")
        print(f"诉讼请求: {elements.claims}")
        print(f"查明事实: {elements.found_facts[:80]}...")
        print(f"本院认为: {elements.court_reasoning[:80]}...")
        print(f"判决结果: {elements.judgment_result[:80]}...")
        print(f"适用法律: {elements.applied_laws}")
        print(f"解析置信度: {elements.parse_confidence}")
        print("\nAll tests completed.")
        sys.exit(0)

    path = Path(sys.argv[1])
    text = path.read_text(encoding='utf-8')
    parser = JudgmentParser()
    elements = parser.parse(text)
    print(elements.to_json())
