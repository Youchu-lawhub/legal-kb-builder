#!/usr/bin/env python3
"""
增强索引生成模块（v2 - 目录驱动版）

核心改进：
- 以书的真实目录（TOC）为唯一依据构建索引
- 区分"结构条文"（书的组织单元）和"引用条文"（正文中引用的其他法律）
- 支持七种书籍类型的专用索引策略
- 数量一致性校验，防止正文引用被误当作结构条目

v1 的错误教训：
  _extract_articles() 盲目扫描所有"第X条"，导致一本只有69条
  司法解释的书被生成了489条索引——混入了正文中引用的民法典条文。
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict


# ─── 书籍类型常量 ───
BOOK_TYPE_CODE_COMMENTARY = "code_commentary"           # 法典评注类
BOOK_TYPE_JUDICIAL_INTERPRETATION = "judicial_interpretation"  # 司法解释适用类
BOOK_TYPE_MONOGRAPH = "monograph"                       # 专著类
BOOK_TYPE_TEXTBOOK = "textbook"                         # 教科书类
BOOK_TYPE_CASE_COMPILATION = "case_compilation"         # 案例汇编类
BOOK_TYPE_STATUTE_COMPILATION = "statute_compilation"   # 法规汇编类
BOOK_TYPE_PRACTICE_GUIDE = "practice_guide"             # 实务指引类

ALL_BOOK_TYPES = [
    BOOK_TYPE_CODE_COMMENTARY,
    BOOK_TYPE_JUDICIAL_INTERPRETATION,
    BOOK_TYPE_MONOGRAPH,
    BOOK_TYPE_TEXTBOOK,
    BOOK_TYPE_CASE_COMPILATION,
    BOOK_TYPE_STATUTE_COMPILATION,
    BOOK_TYPE_PRACTICE_GUIDE,
]


class EnhancedIndex:
    """
    增强索引生成器（v2 - 目录驱动版）
    
    使用方法：
        indexer = EnhancedIndex(kb_path)
        
        # 法典评注类 / 司法解释适用类（条文驱动）
        index = indexer.build(
            md_dir=Path("md_files/"),
            book_info={"name": "...", "author": "..."},
            book_type="judicial_interpretation",
            toc_articles=[1, 2, 3, ..., 69],  # 从目录提取的结构条文号
        )
        
        # 专著类 / 教科书类（章节驱动）
        index = indexer.build(
            md_dir=Path("md_files/"),
            book_info={"name": "...", "author": "..."},
            book_type="monograph",
            toc_chapters=["第一章 概述", "第二章 损害", ...],
        )
    """
    
    def __init__(self, kb_path: Path):
        self.kb_path = Path(kb_path)
        
        # 结构索引：书的组织单元（由目录决定）
        self.structural_index = {}
        
        # 引用索引：正文中引用的其他法律条文（仅供参考，不是结构条目）
        self.citation_index = defaultdict(list)
        
        # 章节索引
        self.chapter_index = {}
        
        # 学者观点索引
        self.scholar_index = defaultdict(list)
        
        # 引用关系图
        self.citation_graph = defaultdict(list)
        
        # 主题索引（手工辅助生成）
        self.topic_index = defaultdict(list)
        
        # 案例索引（案例汇编类使用）
        self.case_index = {}
        
        # 法规索引（法规汇编类使用）
        self.statute_index = {}
        
        # 专题索引（实务指引类使用）
        self.guide_topic_index = {}
        
        # 校验用
        self._toc_articles: Set[int] = set()
        self._toc_chapters: List[str] = []
        self._toc_cases: List[str] = []
        self._toc_topics: List[str] = []
        self._toc_statutes: List[str] = []
        self._book_type: str = ""
    
    def build(
        self,
        md_dir: Path,
        book_info: Dict[str, Any],
        book_type: str = BOOK_TYPE_MONOGRAPH,
        toc_articles: Optional[List[int]] = None,
        toc_chapters: Optional[List[str]] = None,
        toc_cases: Optional[List[str]] = None,
        toc_topics: Optional[List[str]] = None,
        toc_statutes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        构建目录驱动的增强索引
        
        Args:
            md_dir: Markdown文件目录
            book_info: 书籍基本信息 {"name", "author", "publisher", ...}
            book_type: 书籍类型（见常量定义）
            toc_articles: 目录中的结构条文号列表（法典评注/司法解释类必填）
            toc_chapters: 目录中的章节标题列表（专著/教科书类必填）
            toc_cases: 目录中的案例名称列表（案例汇编类必填）
            toc_topics: 目录中的专题标题列表（实务指引类必填）
            toc_statutes: 目录中的法规名称列表（法规汇编类必填）
            
        Returns:
            完整的索引数据字典
        """
        if book_type not in ALL_BOOK_TYPES:
            raise ValueError(f"未知的书籍类型: {book_type}。支持: {ALL_BOOK_TYPES}")
        
        self._book_type = book_type
        self._toc_articles = set(toc_articles or [])
        self._toc_chapters = toc_chapters or []
        self._toc_cases = toc_cases or []
        self._toc_topics = toc_topics or []
        self._toc_statutes = toc_statutes or []
        
        md_files = sorted(Path(md_dir).glob("*.md"))
        if not md_files:
            # 尝试子目录
            md_files = sorted(Path(md_dir).glob("**/*.md"))
        
        print(f"[EnhancedIndex] 书籍类型: {book_type}")
        print(f"[EnhancedIndex] 找到 {len(md_files)} 个MD文件")
        
        if book_type in (BOOK_TYPE_CODE_COMMENTARY, BOOK_TYPE_JUDICIAL_INTERPRETATION):
            if not toc_articles:
                raise ValueError(
                    f"书籍类型 {book_type} 必须提供 toc_articles 参数！\n"
                    "请先分析书的目录，提取所有结构条文号。"
                )
            print(f"[EnhancedIndex] 目录声明的结构条文: {len(toc_articles)} 条")
        elif book_type == BOOK_TYPE_CASE_COMPILATION:
            if not toc_cases:
                raise ValueError(
                    f"书籍类型 {book_type} 必须提供 toc_cases 参数！\n"
                    "请先分析书的目录，提取所有案例名称。"
                )
            print(f"[EnhancedIndex] 目录声明的案例: {len(toc_cases)} 个")
        elif book_type == BOOK_TYPE_STATUTE_COMPILATION:
            if not toc_statutes:
                raise ValueError(
                    f"书籍类型 {book_type} 必须提供 toc_statutes 参数！\n"
                    "请先分析书的目录，提取所有法规名称。"
                )
            print(f"[EnhancedIndex] 目录声明的法规: {len(toc_statutes)} 个")
        elif book_type == BOOK_TYPE_PRACTICE_GUIDE:
            if not toc_topics:
                raise ValueError(
                    f"书籍类型 {book_type} 必须提供 toc_topics 参数！\n"
                    "请先分析书的目录，提取所有专题标题。"
                )
            print(f"[EnhancedIndex] 目录声明的专题: {len(toc_topics)} 个")
        
        # 处理每个文件
        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')
            self._process_file(md_file.name, content, book_type)
        
        # 构建索引
        index = self._assemble_index(
            book_info, book_type, toc_articles, toc_chapters,
            toc_cases, toc_topics, toc_statutes
        )
        
        # 校验
        warnings = self._validate(
            book_type, toc_articles, toc_chapters,
            toc_cases, toc_topics, toc_statutes
        )
        if warnings:
            index["_validation_warnings"] = warnings
            for w in warnings:
                print(f"[警告] {w}")
        
        return index
    
    def _process_file(self, filename: str, content: str, book_type: str):
        """根据书籍类型处理单个文件"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # 1. 识别章节标题（所有类型通用）
            chapter_match = self._match_chapter_title(line)
            if chapter_match:
                self.chapter_index[chapter_match] = {
                    "file": filename,
                    "line": line_num,
                    "title": chapter_match
                }
            
            # 2. 根据书籍类型处理条文
            if book_type in (BOOK_TYPE_CODE_COMMENTARY, BOOK_TYPE_JUDICIAL_INTERPRETATION):
                self._process_article_line(filename, line, line_num, lines)
            
            # 2b. 案例汇编类：匹配案例名称/案号
            if book_type == BOOK_TYPE_CASE_COMPILATION:
                self._process_case_line(filename, line, line_num)
            
            # 2c. 法规汇编类：匹配法规名称
            if book_type == BOOK_TYPE_STATUTE_COMPILATION:
                self._process_statute_line(filename, line, line_num)
            
            # 2d. 实务指引类：匹配专题标题
            if book_type == BOOK_TYPE_PRACTICE_GUIDE:
                self._process_topic_line(filename, line, line_num)
            
            # 3. 识别学者观点（所有类型通用）
            scholar_views = self._extract_scholar_views(line)
            for scholar, view in scholar_views:
                self.scholar_index[scholar].append({
                    "file": filename,
                    "line": line_num,
                    "view": view,
                    "context": self._get_context(lines, line_num)
                })
    
    def _process_article_line(self, filename: str, line: str, line_num: int, lines: List[str]):
        """
        处理条文行（法典评注类 / 司法解释适用类专用）
        
        核心逻辑：区分结构条文和引用条文
        - 结构条文：编号在 toc_articles 中 → 记入 structural_index
        - 引用条文：编号不在 toc_articles 中 → 记入 citation_index（仅供参考）
        """
        # 提取所有"第X条"
        pattern = r'第([一二三四五六七八九十百千零\d]+)条'
        matches = re.finditer(pattern, line)
        
        for match in matches:
            num = self._normalize_number(match.group(1))
            if num is None:
                continue
            
            article_key = f"第{num}条"
            location = {
                "file": filename,
                "line": line_num,
                "context": self._get_context(lines, line_num, context_size=2),
                "match_text": match.group(0),
                "match_position": match.start(),
            }
            
            if num in self._toc_articles:
                # ✅ 结构条文：属于本书的组织单元
                is_heading = self._is_structural_heading(line, match, num)
                location["is_heading"] = is_heading
                
                if article_key not in self.structural_index:
                    self.structural_index[article_key] = []
                self.structural_index[article_key].append(location)
            else:
                # ⚠️ 引用条文：正文中引用的其他法律（不是本书的结构条目）
                # 尝试识别引用的是哪部法律
                cited_law = self._identify_cited_law(line, match.start())
                location["cited_law"] = cited_law
                self.citation_index[article_key].append(location)
    
    def _is_structural_heading(self, line: str, match, num: int) -> bool:
        """
        判断这个条文号是否作为结构标题出现（而非正文中的普通提及）
        
        结构标题的特征：
        - 独立成行或接近行首
        - 通常是 Markdown 标题（以 # 开头）
        - 后面跟着条文正文或【条文主旨】等
        """
        stripped = line.strip()
        
        # Markdown 标题
        if re.match(r'^#{1,6}\s+第[一二三四五六七八九十百千零\d]+条', stripped):
            return True
        
        # 独立成行或行首位置
        if match.start() <= 5:  # 条文号在行的前5个字符内
            return True
        
        # 紧跟方括号注释（如"第一条【条文主旨】"）
        if re.search(r'第[一二三四五六七八九十百千零\d]+条\s*[【\[]', stripped):
            return True
        
        return False
    
    def _identify_cited_law(self, line: str, match_pos: int) -> str:
        """
        识别被引用的法律名称
        
        常见格式：
        - "《民法典》第584条"
        - "民法典第584条"  
        - "合同法第107条"
        - "《最高人民法院关于...的解释》第10条"
        """
        # 往前搜索，查找最近的书名号法律名称
        before_text = line[:match_pos]
        
        # 优先匹配书名号格式：《法律名称》
        guillemet_match = re.search(r'《([^》]+)》\s*$', before_text)
        if guillemet_match:
            return guillemet_match.group(1)
        
        # 匹配无书名号格式：XX法/XX条例/XX规定/XX解释
        law_match = re.search(
            r'([\u4e00-\u9fa5]{2,20}(?:法|条例|规定|解释|规则|办法|意见|通知|批复))\s*$',
            before_text
        )
        if law_match:
            return law_match.group(1)
        
        return "未识别"
    
    def _process_case_line(self, filename: str, line: str, line_num: int):
        """
        处理案例行（案例汇编类专用）
        
        匹配目录中声明的案例名称或案号格式
        """
        stripped = line.strip()
        
        # 匹配目录声明的案例名称
        for case_name in self._toc_cases:
            if case_name in stripped:
                # 判断是否是标题行（以#开头或独立成行）
                is_heading = stripped.startswith('#') or stripped == case_name
                if case_name not in self.case_index or is_heading:
                    self.case_index[case_name] = {
                        "file": filename,
                        "line": line_num,
                        "title": case_name,
                        "is_heading": is_heading,
                    }
        
        # 匹配案号格式：(2020)最高法民终XXX号
        case_number_pattern = r'[（(]\d{4}[）)]\s*[\u4e00-\u9fa5]+\d*[\u4e00-\u9fa5]*\d+号'
        case_match = re.search(case_number_pattern, stripped)
        if case_match:
            case_num = case_match.group(0)
            if case_num not in self.case_index:
                self.case_index[case_num] = {
                    "file": filename,
                    "line": line_num,
                    "title": case_num,
                    "is_heading": False,
                }
    
    def _process_statute_line(self, filename: str, line: str, line_num: int):
        """
        处理法规行（法规汇编类专用）
        
        匹配目录中声明的法规名称
        """
        stripped = line.strip()
        
        for statute_name in self._toc_statutes:
            if statute_name in stripped:
                is_heading = stripped.startswith('#') or len(stripped) < len(statute_name) + 20
                if statute_name not in self.statute_index or is_heading:
                    self.statute_index[statute_name] = {
                        "file": filename,
                        "line": line_num,
                        "title": statute_name,
                        "is_heading": is_heading,
                    }
    
    def _process_topic_line(self, filename: str, line: str, line_num: int):
        """
        处理专题行（实务指引类专用）
        
        匹配目录中声明的专题标题
        """
        stripped = line.strip()
        
        for topic_name in self._toc_topics:
            if topic_name in stripped:
                is_heading = stripped.startswith('#') or stripped == topic_name
                if topic_name not in self.guide_topic_index or is_heading:
                    self.guide_topic_index[topic_name] = {
                        "file": filename,
                        "line": line_num,
                        "title": topic_name,
                        "is_heading": is_heading,
                    }
    
    def _match_chapter_title(self, line: str) -> Optional[str]:
        """匹配章节标题"""
        patterns = [
            r'^(#{1,6}\s+)?(第[一二三四五六七八九十百千零\d]+[章节编部分])\s*[、.．]?\s*(.+)$',
            r'^(#{1,6}\s+)?(\d+[.．])\s*(.+)$',
            r'^(#{1,6}\s+)?(第\d+章)\s+(.+)$',
            r'^(#{1,6}\s+)?([一二三四五六七八九十]+[、.])\s*(.+)$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                # 去除 Markdown 标题标记
                full = match.group(0).lstrip('#').strip()
                return full
        
        return None
    
    def _extract_scholar_views(self, line: str) -> List[Tuple[str, str]]:
        """提取学者观点（与 v1 相同）"""
        results = []
        
        # 通用模式：XX教授/先生 认为/指出/主张
        general_pattern = r'([\u4e00-\u9fa5]{2,4})\s*(?:教授|先生|老师|博士|法官|大法官)?\s*(?:认为|指出|主张|强调|提出|论述)'
        matches = re.finditer(general_pattern, line)
        
        seen = set()
        for match in matches:
            scholar = match.group(1)
            if scholar not in seen:
                seen.add(scholar)
                results.append((scholar, line.strip()))
        
        return results
    
    def _get_context(self, lines: List[str], line_num: int, context_size: int = 3) -> str:
        """获取上下文"""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        return '\n'.join(lines[start:end])
    
    def _normalize_number(self, num_str: str) -> Optional[int]:
        """将中文/阿拉伯数字统一转换为阿拉伯数字"""
        if num_str.isdigit():
            return int(num_str)
        
        chinese_nums = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10, '百': 100, '千': 1000
        }
        
        result = 0
        temp = 0
        
        for char in num_str:
            if char in chinese_nums:
                num = chinese_nums[char]
                if num >= 10:
                    if temp == 0:
                        temp = 1
                    result += temp * num
                    temp = 0
                else:
                    temp = temp * 10 + num if temp else num
        
        result += temp
        return result if result > 0 else None
    
    def _assemble_index(
        self,
        book_info: Dict[str, Any],
        book_type: str,
        toc_articles: Optional[List[int]],
        toc_chapters: Optional[List[str]],
        toc_cases: Optional[List[str]] = None,
        toc_topics: Optional[List[str]] = None,
        toc_statutes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """组装最终索引"""
        
        index = {
            "book_info": book_info,
            "book_type": book_type,
            "index_version": "v2_toc_driven",
        }
        
        if book_type in (BOOK_TYPE_CODE_COMMENTARY, BOOK_TYPE_JUDICIAL_INTERPRETATION):
            index["structural_articles"] = {
                "description": "本书的结构条文（由目录决定的组织单元）",
                "total_count": len(self._toc_articles),
                "found_count": len(self.structural_index),
                "articles": {}
            }
            
            # 为每个目录条文记录位置
            for num in sorted(self._toc_articles):
                key = f"第{num}条"
                if key in self.structural_index:
                    # 找到该条文的标题位置（is_heading=True 的优先）
                    locations = self.structural_index[key]
                    heading_locations = [l for l in locations if l.get("is_heading")]
                    primary = heading_locations[0] if heading_locations else locations[0]
                    
                    index["structural_articles"]["articles"][key] = {
                        "number": num,
                        "primary_file": primary["file"],
                        "primary_line": primary["line"],
                        "total_mentions": len(locations),
                        "heading_found": bool(heading_locations),
                    }
                else:
                    index["structural_articles"]["articles"][key] = {
                        "number": num,
                        "primary_file": None,
                        "primary_line": None,
                        "total_mentions": 0,
                        "heading_found": False,
                        "_warning": "目录中声明但未在正文中找到",
                    }
            
            # 引用统计
            index["citation_stats"] = {
                "description": "正文中引用的其他法律条文（非本书结构条目）",
                "total_cited_articles": len(self.citation_index),
                "top_cited_laws": self._get_citation_law_stats(),
            }
        
        elif book_type in (BOOK_TYPE_MONOGRAPH, BOOK_TYPE_TEXTBOOK):
            index["chapters"] = {
                "description": "本书的章节结构",
                "total_count": len(self.chapter_index),
                "entries": self.chapter_index,
            }
        
        elif book_type == BOOK_TYPE_CASE_COMPILATION:
            index["cases"] = {
                "description": "本书收录的案例（由目录决定）",
                "total_declared": len(self._toc_cases),
                "found_count": len(self.case_index),
                "entries": {}
            }
            for case_name in self._toc_cases:
                if case_name in self.case_index:
                    index["cases"]["entries"][case_name] = self.case_index[case_name]
                else:
                    index["cases"]["entries"][case_name] = {
                        "file": None,
                        "line": None,
                        "title": case_name,
                        "is_heading": False,
                        "_warning": "目录中声明但未在正文中找到",
                    }
        
        elif book_type == BOOK_TYPE_STATUTE_COMPILATION:
            index["statutes"] = {
                "description": "本书收录的法规（由目录决定）",
                "total_declared": len(self._toc_statutes),
                "found_count": len(self.statute_index),
                "entries": {}
            }
            for statute_name in self._toc_statutes:
                if statute_name in self.statute_index:
                    index["statutes"]["entries"][statute_name] = self.statute_index[statute_name]
                else:
                    index["statutes"]["entries"][statute_name] = {
                        "file": None,
                        "line": None,
                        "title": statute_name,
                        "is_heading": False,
                        "_warning": "目录中声明但未在正文中找到",
                    }
        
        elif book_type == BOOK_TYPE_PRACTICE_GUIDE:
            index["topics"] = {
                "description": "本书的专题结构（由目录决定）",
                "total_declared": len(self._toc_topics),
                "found_count": len(self.guide_topic_index),
                "entries": {}
            }
            for topic_name in self._toc_topics:
                if topic_name in self.guide_topic_index:
                    index["topics"]["entries"][topic_name] = self.guide_topic_index[topic_name]
                else:
                    index["topics"]["entries"][topic_name] = {
                        "file": None,
                        "line": None,
                        "title": topic_name,
                        "is_heading": False,
                        "_warning": "目录中声明但未在正文中找到",
                    }
        
        # 通用部分
        index["scholar_index"] = {
            "total_scholars": len(self.scholar_index),
            "scholars": {
                scholar: {
                    "mention_count": len(views),
                    "files": list(set(v["file"] for v in views)),
                }
                for scholar, views in self.scholar_index.items()
            }
        }
        
        # 统计：根据类型计算结构条目数
        structural_count = 0
        if book_type in (BOOK_TYPE_CODE_COMMENTARY, BOOK_TYPE_JUDICIAL_INTERPRETATION):
            structural_count = len(self.structural_index)
        elif book_type in (BOOK_TYPE_MONOGRAPH, BOOK_TYPE_TEXTBOOK):
            structural_count = len(self.chapter_index)
        elif book_type == BOOK_TYPE_CASE_COMPILATION:
            structural_count = len(self.case_index)
        elif book_type == BOOK_TYPE_STATUTE_COMPILATION:
            structural_count = len(self.statute_index)
        elif book_type == BOOK_TYPE_PRACTICE_GUIDE:
            structural_count = len(self.guide_topic_index)
        
        index["stats"] = {
            "structural_entries": structural_count,
            "citation_entries": len(self.citation_index),
            "scholar_count": len(self.scholar_index),
            "scholar_mention_count": sum(len(v) for v in self.scholar_index.values()),
        }
        
        return index
    
    def _get_citation_law_stats(self) -> Dict[str, int]:
        """统计引用的法律来源分布"""
        law_counts = defaultdict(int)
        for article_key, locations in self.citation_index.items():
            for loc in locations:
                law = loc.get("cited_law", "未识别")
                law_counts[law] += 1
        
        # 按引用次数排序，取前10
        sorted_laws = sorted(law_counts.items(), key=lambda x: -x[1])[:10]
        return dict(sorted_laws)
    
    def _validate(
        self,
        book_type: str,
        toc_articles: Optional[List[int]],
        toc_chapters: Optional[List[str]],
        toc_cases: Optional[List[str]] = None,
        toc_topics: Optional[List[str]] = None,
        toc_statutes: Optional[List[str]] = None,
    ) -> List[str]:
        """校验索引的一致性"""
        warnings = []
        
        if book_type in (BOOK_TYPE_CODE_COMMENTARY, BOOK_TYPE_JUDICIAL_INTERPRETATION):
            if toc_articles:
                expected = len(toc_articles)
                found = len(self.structural_index)
                
                if found < expected:
                    missing = self._toc_articles - set(
                        self._normalize_number(k.replace("第", "").replace("条", "")) or 0
                        for k in self.structural_index.keys()
                    )
                    warnings.append(
                        f"结构条文覆盖不足：目录声明 {expected} 条，"
                        f"实际找到 {found} 条。缺失：{sorted(missing)[:10]}..."
                    )
                
                if found > expected * 1.5:
                    warnings.append(
                        f"⚠️ 疑似混入引用条文！目录声明 {expected} 条，"
                        f"但找到了 {found} 条。请检查 toc_articles 是否正确。"
                    )
                
                # 检查引用条文数量
                cited_count = len(self.citation_index)
                if cited_count > 0:
                    warnings.append(
                        f"[信息] 正文中引用了 {cited_count} 个非结构条文编号"
                        f"（已正确分类到 citation_index，不影响结构索引）"
                    )
        
        elif book_type in (BOOK_TYPE_MONOGRAPH, BOOK_TYPE_TEXTBOOK):
            if toc_chapters:
                expected = len(toc_chapters)
                found = len(self.chapter_index)
                if found < expected * 0.5:
                    warnings.append(
                        f"章节覆盖不足：目录声明 {expected} 个章节，"
                        f"但只找到 {found} 个。可能是标题格式不匹配。"
                    )
        
        elif book_type == BOOK_TYPE_CASE_COMPILATION:
            if toc_cases:
                expected = len(toc_cases)
                found = sum(1 for c in toc_cases if c in self.case_index)
                if found < expected * 0.5:
                    missing = [c for c in toc_cases if c not in self.case_index]
                    warnings.append(
                        f"案例覆盖不足：目录声明 {expected} 个案例，"
                        f"但只匹配到 {found} 个。"
                        f"未匹配的前5个：{missing[:5]}"
                    )
        
        elif book_type == BOOK_TYPE_STATUTE_COMPILATION:
            if toc_statutes:
                expected = len(toc_statutes)
                found = sum(1 for s in toc_statutes if s in self.statute_index)
                if found < expected * 0.5:
                    missing = [s for s in toc_statutes if s not in self.statute_index]
                    warnings.append(
                        f"法规覆盖不足：目录声明 {expected} 个法规，"
                        f"但只匹配到 {found} 个。"
                        f"未匹配的前5个：{missing[:5]}"
                    )
        
        elif book_type == BOOK_TYPE_PRACTICE_GUIDE:
            if toc_topics:
                expected = len(toc_topics)
                found = sum(1 for t in toc_topics if t in self.guide_topic_index)
                if found < expected * 0.5:
                    missing = [t for t in toc_topics if t not in self.guide_topic_index]
                    warnings.append(
                        f"专题覆盖不足：目录声明 {expected} 个专题，"
                        f"但只匹配到 {found} 个。"
                        f"未匹配的前5个：{missing[:5]}"
                    )
        
        return warnings
    
    def save(self, index: Dict[str, Any], output_path: Path):
        """保存索引到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        print(f"[EnhancedIndex] 索引已保存: {output_path}")
    
    def load(self, index_path: Path) -> Dict[str, Any]:
        """从文件加载索引"""
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ─── 便捷函数 ───

def build_article_index(
    md_dir: str,
    book_name: str,
    book_type: str,
    toc_articles: List[int],
    output_path: Optional[str] = None,
    **book_info
) -> Dict[str, Any]:
    """
    便捷函数：为条文驱动的书籍构建索引
    
    示例：
        index = build_article_index(
            md_dir="kb/",
            book_name="合同编通则司法解释理解与适用",
            book_type="judicial_interpretation",
            toc_articles=list(range(1, 70)),  # 第1条到第69条
            author="最高人民法院民事审判第二庭",
        )
    """
    indexer = EnhancedIndex(Path(md_dir))
    index = indexer.build(
        md_dir=Path(md_dir),
        book_info={"name": book_name, **book_info},
        book_type=book_type,
        toc_articles=toc_articles,
    )
    
    if output_path:
        indexer.save(index, Path(output_path))
    
    return index


def build_chapter_index(
    md_dir: str,
    book_name: str,
    book_type: str,
    toc_chapters: List[str],
    output_path: Optional[str] = None,
    **book_info
) -> Dict[str, Any]:
    """
    便捷函数：为章节驱动的书籍构建索引
    
    示例：
        index = build_chapter_index(
            md_dir="kb/",
            book_name="损害赔偿",
            book_type="monograph",
            toc_chapters=["第一章 概述", "第二章 损害"],
            author="王泽鉴",
        )
    """
    indexer = EnhancedIndex(Path(md_dir))
    index = indexer.build(
        md_dir=Path(md_dir),
        book_info={"name": book_name, **book_info},
        book_type=book_type,
        toc_chapters=toc_chapters,
    )
    
    if output_path:
        indexer.save(index, Path(output_path))
    
    return index


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
增强索引生成器 v2（目录驱动版）

用法:
  # 司法解释适用类（条文驱动）
  python enhanced_index.py <md_dir> --type judicial_interpretation --articles 1-69
  
  # 法典评注类（条文驱动）
  python enhanced_index.py <md_dir> --type code_commentary --articles 1-1260
  
  # 专著类（章节驱动）
  python enhanced_index.py <md_dir> --type monograph

选项:
  --type TYPE       书籍类型（必填）
  --articles RANGE  结构条文范围，如 "1-69"（条文驱动类型必填）
  --output PATH     输出索引文件路径
  --name NAME       书籍名称
""")
        sys.exit(0)
    
    import argparse
    parser = argparse.ArgumentParser(description="增强索引生成器 v2")
    parser.add_argument('md_dir', help='Markdown文件目录')
    parser.add_argument('--type', required=True, choices=ALL_BOOK_TYPES, help='书籍类型')
    parser.add_argument('--articles', help='结构条文范围，如 "1-69"')
    parser.add_argument('--output', help='输出索引文件路径')
    parser.add_argument('--name', default='未命名', help='书籍名称')
    
    args = parser.parse_args()
    
    toc_articles = None
    if args.articles:
        parts = args.articles.split('-')
        if len(parts) == 2:
            toc_articles = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            toc_articles = [int(x) for x in args.articles.split(',')]
    
    indexer = EnhancedIndex(Path("."))
    result = indexer.build(
        Path(args.md_dir),
        {"name": args.name},
        book_type=args.type,
        toc_articles=toc_articles,
    )
    
    if args.output:
        indexer.save(result, Path(args.output))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
