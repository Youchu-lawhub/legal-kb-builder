#!/usr/bin/env python3
"""
文档类型路由器（决策树）

根据文件类型、大小和内容样本，自动判断文档应进入哪条流水线：
  - book_kb   ：结构化法律书籍（评注、司法解释、专著等） → 走 legal_kb.py
  - case_kb   ：法院裁判文书（判决/裁定/调解书）       → 走 judgment_kb.py
  - qa_kb     ：问答集语料（FAQ/业务咨询问答/培训问答）  → 走 qa_kb.py
  - agentic   ：探索性学术资料（论文、比较法、报告等） → 走 monte_carlo_sampler.py
  - direct    ：方法论/参考文档，直接放入 references/

使用方式：
    from material_router import MaterialRouter

    router = MaterialRouter()
    decision = router.route("/path/to/document.pdf")
    print(decision.target_skill)    # 'book_kb'
    print(decision.reasoning)       # '文档包含完整目录结构...'
    print(decision.confidence)      # 0.85
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shared_utils import load_yaml_config

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# ─── 默认路径 ───
_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_DEFAULT_ROUTING_RULES = _ASSETS_DIR / "routing-rules.yaml"


@dataclass
class RoutingDecision:
    """路由决策结果"""
    target_skill: str               # 'book_kb' | 'case_kb' | 'qa_kb' | 'agentic' | 'direct'
    book_type: Optional[str] = None # 对 book_kb：7 种书籍类型之一
    confidence: float = 0.0         # 0-1
    reasoning: str = ""             # 人类可读的推理过程
    suggested_params: Dict[str, Any] = field(default_factory=dict)


class MaterialRouter:
    """
    文档类型路由器

    Args:
        rules_path: 路由规则 YAML 路径（默认使用内置）
    """

    def __init__(self, rules_path: str = None):
        path = Path(rules_path) if rules_path else _DEFAULT_ROUTING_RULES
        self._rules = load_yaml_config(str(path))

    def route(self, file_path: str) -> RoutingDecision:
        """
        对给定文件执行完整的路由决策。

        Args:
            file_path: 文件路径

        Returns:
            RoutingDecision 包含推荐的 Skill 和参数
        """
        path = Path(file_path)
        if not path.exists():
            return RoutingDecision(
                target_skill='unknown',
                reasoning=f"文件不存在: {file_path}",
                confidence=0.0,
            )

        # Step 1: 检测文件类型
        file_type = self._detect_file_type(path)

        # Step 2: 测量文件大小
        size_info = self._measure_file_size(path)

        # Step 3: 采样内容
        sample = self._sample_content(path, file_type)

        # Step 4: 分类内容
        classification = self._classify_content(sample, size_info)

        # Step 5: 综合决策
        return self._determine_strategy(file_type, size_info, classification)

    # ─── 分析步骤 ───

    def _detect_file_type(self, path: Path) -> str:
        """检测文件类型"""
        suffix = path.suffix.lower()
        type_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.md': 'markdown',
            '.txt': 'text',
            '.markdown': 'markdown',
            # 图片类：路由到 image，由 parser_adapter 走 OCR 后重新分类
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.tif': 'image',
            '.tiff': 'image',
            '.bmp': 'image',
        }
        return type_map.get(suffix, 'unknown')

    def _measure_file_size(self, path: Path) -> Dict[str, Any]:
        """测量文件大小，PDF 另外获取页数"""
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        info = {
            'size_bytes': size_bytes,
            'size_mb': round(size_mb, 2),
            'pages': None,
            'category': 'small',  # small / medium / large
        }

        # PDF 页数检测
        if path.suffix.lower() == '.pdf' and HAS_PYPDF:
            try:
                reader = PdfReader(str(path))
                info['pages'] = len(reader.pages)
            except Exception:
                pass

        # 估算页数（非 PDF 或无法读取时）
        if info['pages'] is None:
            if path.suffix.lower() in ('.md', '.txt', '.markdown'):
                try:
                    line_count = sum(1 for _ in open(path, 'r', encoding='utf-8'))
                    info['pages'] = max(1, line_count // 40)
                except Exception:
                    info['pages'] = max(1, int(size_mb * 5))
            else:
                info['pages'] = max(1, int(size_mb * 5))

        # 分类
        thresholds = self._rules.get('size_thresholds', {})
        small_threshold = thresholds.get('small', 50)
        large_threshold = thresholds.get('large', 500)

        if info['pages'] <= small_threshold:
            info['category'] = 'small'
        elif info['pages'] <= large_threshold:
            info['category'] = 'medium'
        else:
            info['category'] = 'large'

        return info

    def _sample_content(self, path: Path, file_type: str, sample_size: int = 5000) -> str:
        """从文件中提取内容样本。v2 新增 docx 支持；图片文件目前只返回空样本
        （由下游 parser_adapter 走 OCR 转 md 后再重新路由）。"""
        if file_type in ('markdown', 'text'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read(sample_size)
            except UnicodeDecodeError:
                # 常见 GB18030 老文书
                try:
                    with open(path, 'r', encoding='gb18030', errors='replace') as f:
                        return f.read(sample_size)
                except Exception:
                    return ""
            except Exception:
                return ""

        elif file_type == 'pdf' and HAS_PYPDF:
            try:
                reader = PdfReader(str(path))
                text_parts = []
                for page in reader.pages[:20]:  # 前 20 页
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    if sum(len(t) for t in text_parts) >= sample_size:
                        break
                return '\n'.join(text_parts)[:sample_size]
            except Exception:
                return ""

        elif file_type == 'docx':
            try:
                from docx import Document  # python-docx
                doc = Document(str(path))
                parts = []
                for para in doc.paragraphs:
                    parts.append(para.text)
                    if sum(len(p) for p in parts) >= sample_size:
                        break
                return '\n'.join(parts)[:sample_size]
            except ImportError:
                return ""
            except Exception:
                return ""

        return ""

    def _classify_content(self, sample: str, size_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析样本内容，判断文档类型。

        返回分类结果字典。
        """
        result = {
            'is_judgment': False,
            'is_structured_book': False,
            'is_qa_set': False,
            'is_methodology': False,
            'is_academic': False,
            'book_type': None,
            'judgment_score': 0,
            'structure_score': 0,
            'qa_score': 0,
            'reasoning_parts': [],
        }

        if not sample:
            result['reasoning_parts'].append("无法提取内容样本")
            return result

        # ── Check 1: 是否为裁判文书？ ──
        judgment_rules = self._rules.get('judgment_indicators', {})
        judgment_patterns = judgment_rules.get('patterns', [
            # 案号（简繁 + 允许中间空白/OCR 噪音）
            r'[（(]\s*\d{4}\s*[）)][^)）]{0,30}?\d+\s*号',
            # 简体核心标记
            '本院认为', '本院查明', '判决如下', '裁定如下', '调解书', '判决书',
            # 繁体（港澳台判决书）
            '本院認為', '本院查明', '判決如下', '裁定如下', '調解書', '判決書',
            # 结构性标记（跨文书类型通用）
            r'原告[^\n]{0,80}被告', r'审理.*终结', r'審理.*終結',
            # OCR 常见变体
            '经审理查明', '經審理查明',
        ])
        min_matches = judgment_rules.get('min_matches', 3)

        judgment_hits = 0
        for pattern in judgment_patterns:
            try:
                if re.search(pattern, sample):
                    judgment_hits += 1
            except re.error:
                # 用户在 rules 里配的正则如果有语法错就跳过（不影响其他）
                continue

        result['judgment_score'] = judgment_hits
        if judgment_hits >= min_matches:
            result['is_judgment'] = True
            result['reasoning_parts'].append(
                f"匹配 {judgment_hits}/{len(judgment_patterns)} 个裁判文书特征"
            )

        # ── Check 1.5: 是否为问答集语料？ ──
        qa_rules = self._rules.get('qa_indicators', {})
        q_patterns = qa_rules.get('question_markers', [
            r'^Q[:：]', r'^问[:：]', r'^问题[:：]',
            r'^Q\d+[:：.、]', r'^问\d+[:：.、]',
        ])
        a_patterns = qa_rules.get('answer_markers', [
            r'^A[:：]', r'^答[:：]', r'^答案[:：]', r'^解答[:：]',
        ])
        hints = qa_rules.get('structural_hints', ['FAQ', '常见问题', '问答', 'Q&A'])
        min_pairs = qa_rules.get('min_qa_pairs', 3)
        min_density = qa_rules.get('min_density', 0.15)

        # 统计问答标记行
        non_empty_lines = [ln for ln in sample.splitlines() if ln.strip()]
        q_hits = 0
        a_hits = 0
        for ln in non_empty_lines:
            stripped = ln.strip()
            for pat in q_patterns:
                try:
                    if re.match(pat, stripped):
                        q_hits += 1
                        break
                except re.error:
                    continue
            else:
                for pat in a_patterns:
                    try:
                        if re.match(pat, stripped):
                            a_hits += 1
                            break
                    except re.error:
                        continue

        # 结构性提示词加分
        hint_hits = sum(1 for h in hints if h.lower() in sample.lower())
        # 问答对数 = min(q_hits, a_hits)（成对计数）+ 不成对的标记折半
        qa_pairs = min(q_hits, a_hits) + abs(q_hits - a_hits) // 2
        density = (q_hits + a_hits) / max(1, len(non_empty_lines))

        result['qa_score'] = qa_pairs
        if qa_pairs >= min_pairs or (qa_pairs >= 2 and density >= min_density) or \
           (hint_hits >= 1 and qa_pairs >= 2):
            result['is_qa_set'] = True
            result['reasoning_parts'].append(
                f"问答集语料（{qa_pairs} 个问答对，密度 {density:.2%}，提示词 {hint_hits}）"
            )

        # ── Check 2: 是否为结构化法律书籍？ ──
        book_rules = self._rules.get('structured_book_indicators', {})
        toc_article = book_rules.get(
            'toc_article_pattern',
            r'第[一二三四五六七八九十百千零\d]+条'
        )
        toc_chapter = book_rules.get(
            'toc_chapter_pattern',
            r'第[一二三四五六七八九十百千零\d]+[章节编]'
        )
        min_toc = book_rules.get('min_toc_entries', 5)

        article_count = len(re.findall(toc_article, sample[:3000]))
        chapter_count = len(re.findall(toc_chapter, sample[:3000]))

        if article_count >= min_toc:
            result['is_structured_book'] = True
            result['structure_score'] = article_count
            # 子分类
            if '司法解释' in sample[:2000] or '理解与适用' in sample[:2000]:
                result['book_type'] = 'judicial_interpretation'
                result['reasoning_parts'].append(
                    f"司法解释适用类书籍（{article_count} 个条文标记）"
                )
            elif '评注' in sample[:2000] or '注释' in sample[:2000]:
                result['book_type'] = 'code_commentary'
                result['reasoning_parts'].append(
                    f"法典评注类书籍（{article_count} 个条文标记）"
                )
            else:
                result['book_type'] = 'code_commentary'
                result['reasoning_parts'].append(
                    f"条文驱动类书籍（{article_count} 个条文标记）"
                )

        elif chapter_count >= min_toc:
            result['is_structured_book'] = True
            result['structure_score'] = chapter_count

            # 案例/法规/教科书/专著/实务指引 细分
            case_markers = ['案例', '判决', '裁定', '案号', '当事人']
            case_hits = sum(1 for m in case_markers if m in sample[:3000])

            if case_hits >= 3:
                result['book_type'] = 'case_compilation'
                result['reasoning_parts'].append("案例汇编类书籍")
            elif any(kw in sample[:2000] for kw in ['课后', '习题', '思考题', '案例研讨']):
                result['book_type'] = 'textbook'
                result['reasoning_parts'].append("教科书类书籍")
            elif any(kw in sample[:2000] for kw in ['操作流程', '注意事项', '常见问题', '实务']):
                result['book_type'] = 'practice_guide'
                result['reasoning_parts'].append("实务指引类书籍")
            else:
                result['book_type'] = 'monograph'
                result['reasoning_parts'].append(
                    f"专著类书籍（{chapter_count} 个章节标记）"
                )

        # ── Check 3: 是否为方法论/参考文档？ ──
        method_rules = self._rules.get('methodology_indicators', {})
        max_pages = method_rules.get('max_pages', 20)
        method_keywords = method_rules.get('keywords', [
            '操作指南', '使用说明', '方法论', '工作流', '步骤'
        ])

        if size_info.get('pages', 0) <= max_pages:
            method_hits = sum(1 for kw in method_keywords if kw in sample)
            if method_hits >= 2:
                result['is_methodology'] = True
                result['reasoning_parts'].append(
                    f"方法论文档（{size_info['pages']} 页，匹配 {method_hits} 个指示词）"
                )

        # ── Check 4: 默认学术/探索性材料 ──
        if not any([result['is_judgment'], result['is_structured_book'],
                    result['is_qa_set'], result['is_methodology']]):
            result['is_academic'] = True
            result['reasoning_parts'].append("未检测到明确结构，归类为学术/探索性材料")

        return result

    def _determine_strategy(
        self, file_type: str, size_info: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> RoutingDecision:
        """综合所有信号，生成最终路由决策"""

        reasoning = '; '.join(classification.get('reasoning_parts', []))

        # 图片：无内容样本可判，先提示用户走 parser_adapter 做 OCR，然后重新路由
        if file_type == 'image':
            return RoutingDecision(
                target_skill='direct',
                confidence=0.4,
                reasoning="图片文件：先调用 parser_adapter 走 OCR 转 Markdown，"
                          "再对产出的 .md 重新调用 route() 分类",
                suggested_params={
                    'requires_ocr': True,
                    'next_step': 'parser_adapter.parse_to_markdown',
                },
            )

        # 冲突判定：书 vs 判决书（如案例汇编、含案例的教科书）
        if classification['is_judgment'] and classification['is_structured_book']:
            # 判决书特征密度足够高才判为 case_kb；否则视作案例汇编书籍
            jscore = classification.get('judgment_score', 0)
            sscore = classification.get('structure_score', 0)
            if jscore >= max(3, sscore * 0.6):
                return RoutingDecision(
                    target_skill='case_kb',
                    confidence=min(0.9, 0.55 + jscore * 0.08),
                    reasoning=f"冲突判定→case_kb（judgment={jscore}, structure={sscore}）: {reasoning}",
                    suggested_params={
                        'file_type': file_type,
                        'pages': size_info.get('pages'),
                    },
                )
            else:
                return RoutingDecision(
                    target_skill='book_kb',
                    book_type='case_compilation',
                    confidence=min(0.9, 0.55 + sscore * 0.02),
                    reasoning=f"冲突判定→book_kb 案例汇编（judgment={jscore}, structure={sscore}）: {reasoning}",
                    suggested_params={
                        'file_type': file_type,
                        'pages': size_info.get('pages'),
                        'book_type': 'case_compilation',
                    },
                )

        # 裁判文书 -> case_kb
        if classification['is_judgment']:
            return RoutingDecision(
                target_skill='case_kb',
                confidence=min(0.95, 0.6 + classification['judgment_score'] * 0.1),
                reasoning=f"裁判文书: {reasoning}",
                suggested_params={
                    'file_type': file_type,
                    'pages': size_info.get('pages'),
                },
            )

        # 问答集 -> qa_kb（优先于 book_kb：问答集正文可能引用"第X条"，但本质是问答）
        if classification['is_qa_set']:
            return RoutingDecision(
                target_skill='qa_kb',
                confidence=min(0.95, 0.65 + classification['qa_score'] * 0.03),
                reasoning=f"问答集语料: {reasoning}",
                suggested_params={
                    'file_type': file_type,
                    'pages': size_info.get('pages'),
                    'qa_pairs': classification['qa_score'],
                },
            )

        # 结构化法律书籍 -> book_kb
        if classification['is_structured_book']:
            return RoutingDecision(
                target_skill='book_kb',
                book_type=classification.get('book_type'),
                confidence=min(0.95, 0.6 + classification['structure_score'] * 0.02),
                reasoning=f"结构化书籍: {reasoning}",
                suggested_params={
                    'file_type': file_type,
                    'pages': size_info.get('pages'),
                    'book_type': classification.get('book_type'),
                    'size_category': size_info.get('category'),
                },
            )

        # 方法论文档 -> 直接放置
        if classification['is_methodology']:
            return RoutingDecision(
                target_skill='direct',
                confidence=0.7,
                reasoning=f"方法论/参考文档: {reasoning}",
                suggested_params={
                    'placement': 'references/',
                },
            )

        # 学术/探索性材料 -> agentic
        if classification['is_academic']:
            # 大文件可能还是需要 book_kb
            if size_info.get('pages', 0) >= 200:
                return RoutingDecision(
                    target_skill='book_kb',
                    book_type='monograph',
                    confidence=0.5,
                    reasoning=f"大型学术材料（{size_info['pages']} 页），建议使用索引模式: {reasoning}",
                    suggested_params={
                        'file_type': file_type,
                        'pages': size_info.get('pages'),
                        'book_type': 'monograph',
                    },
                )
            else:
                return RoutingDecision(
                    target_skill='agentic',
                    confidence=0.6,
                    reasoning=f"学术/探索性材料: {reasoning}",
                    suggested_params={
                        'file_type': file_type,
                        'pages': size_info.get('pages'),
                    },
                )

        # 兜底
        return RoutingDecision(
            target_skill='agentic',
            confidence=0.3,
            reasoning=f"无法确定类型，默认使用 agentic 模式: {reasoning}",
        )


# ─── CLI 入口 ───

if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("用法: python material_router.py <file_path>")
        print("      python material_router.py --test")
        sys.exit(0)

    if sys.argv[1] == '--test':
        router = MaterialRouter()

        # 模拟裁判文书样本
        class FakePath:
            def __init__(self, suffix, size, content):
                self.suffix = suffix
                self._size = size
                self._content = content
            def exists(self): return True
            def stat(self):
                class S: st_size = self._size
                S.st_size = self._size
                return S()

        # 测试分类逻辑
        judgment_sample = "北京市海淀区人民法院\n民事判决书\n（2023）京0108民初12345号\n原告张三\n被告李四\n经审理查明\n本院认为\n判决如下"
        result = router._classify_content(judgment_sample, {'pages': 5})
        print(f"Judgment test: is_judgment={result['is_judgment']}, score={result['judgment_score']}")

        book_sample = "目录\n第一条 总则\n第二条 基本原则\n第三条 适用范围\n第四条 定义\n第五条 管辖\n第六条"
        result = router._classify_content(book_sample, {'pages': 200})
        print(f"Book test: is_structured={result['is_structured_book']}, type={result['book_type']}")

        qa_sample = "常见问题解答\nQ: 格式条款无效的情形有哪些？\nA: 根据《民法典》第497条...\nQ: 违约金过高如何调整？\nA: 当事人可以请求法院...\nQ: 表见代理的构成要件是什么？\nA: 1.无权代理 2.相对人善意..."
        result = router._classify_content(qa_sample, {'pages': 10})
        print(f"QA test: is_qa_set={result['is_qa_set']}, score={result['qa_score']}")

        print("All tests completed.")
        sys.exit(0)

    path = sys.argv[1]
    router = MaterialRouter()
    decision = router.route(path)
    print(json.dumps({
        'target_skill': decision.target_skill,
        'book_type': decision.book_type,
        'confidence': decision.confidence,
        'reasoning': decision.reasoning,
        'suggested_params': decision.suggested_params,
    }, ensure_ascii=False, indent=2))
