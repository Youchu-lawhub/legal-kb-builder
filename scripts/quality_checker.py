#!/usr/bin/env python3
"""
质量校验模块
检查PDF解析和转换的质量

v2 改进：
- 集成 TextCleaner 的 detect_issues() 进行深度质量评估
- 使用 shared_utils 消除重复的 chinese_to_number 实现
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加当前目录到路径以导入兄弟模块
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shared_utils import chinese_to_number
from text_cleaner import TextCleaner


class QualityChecker:
    """质量检查器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.ocr_threshold = self.config.get("ocr_confidence_threshold", 0.85)
        self.check_continuity = self.config.get("check_chapter_continuity", True)
        self._text_cleaner = TextCleaner()
    
    def check(self, md_dir: Path) -> Dict[str, Any]:
        """
        检查Markdown文件质量
        
        Returns:
            {
                "overall_status": "ok/warning/error",
                "issues": [],
                "warnings": [],
                "stats": {}
            }
        """
        md_files = sorted(md_dir.glob("*.md"))
        if not md_files:
            return {
                "overall_status": "error",
                "issues": ["未找到Markdown文件"],
                "warnings": [],
                "stats": {}
            }
        
        issues = []
        warnings = []
        all_stats = {
            "file_count": len(md_files),
            "total_chars": 0,
            "article_refs": [],
            "page_numbers": []
        }
        
        prev_page = None
        
        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')
            stats = self._analyze_file(content, md_file.name)
            all_stats["total_chars"] += stats["char_count"]
            all_stats["article_refs"].extend(stats["article_refs"])
            
            # 检查OCR质量（通过文本特征推断）
            ocr_score = self._estimate_ocr_quality(content)
            if ocr_score < self.ocr_threshold:
                warnings.append(f"{md_file.name}: OCR质量可能较低 (估计得分: {ocr_score:.2f})")
            
            # 检查章节连续性
            if self.check_continuity:
                page_num = self._extract_page_number(content, md_file.name)
                if page_num:
                    all_stats["page_numbers"].append((md_file.name, page_num))
                    if prev_page and page_num != prev_page + 1:
                        if page_num < prev_page:
                            issues.append(f"{md_file.name}: 页码回退 ({prev_page} -> {page_num})")
                        elif page_num > prev_page + 10:
                            warnings.append(f"{md_file.name}: 页码跳跃较大 ({prev_page} -> {page_num})")
                    prev_page = page_num
            
            # 检查法条编号格式
            invalid_articles = self._check_article_format(content)
            if invalid_articles:
                warnings.append(f"{md_file.name}: 发现 {len(invalid_articles)} 个可疑法条编号")
        
        # 检查引用完整性
        broken_refs = self._check_citation_integrity(all_stats["article_refs"])
        if broken_refs:
            warnings.append(f"发现 {len(broken_refs)} 个无法解析的法条引用")
        
        # 深度检测（使用 TextCleaner）
        full_text = '\n'.join(
            md_file.read_text(encoding='utf-8') for md_file in md_files
        )
        cleaner_issues = self._text_cleaner.detect_issues(full_text)
        
        if cleaner_issues['ocr_error_count'] > 0:
            warnings.append(
                f"TextCleaner 检测到 {cleaner_issues['ocr_error_count']} 个可能的 OCR 错误词"
            )
        if cleaner_issues['duplicate_paragraph_count'] > 0:
            warnings.append(
                f"检测到 {cleaner_issues['duplicate_paragraph_count']} 组近似重复段落（可能来自 PDF 重叠区域）"
            )
        if cleaner_issues['legal_format_issues'] > 0:
            warnings.append(
                f"检测到 {cleaner_issues['legal_format_issues']} 处法律格式问题（书名号/法条编号空格）"
            )
        
        all_stats["cleaner_issues"] = cleaner_issues
        
        # 确定整体状态
        if issues:
            overall_status = "error"
        elif warnings:
            overall_status = "warning"
        else:
            overall_status = "ok"
        
        return {
            "overall_status": overall_status,
            "issues": issues,
            "warnings": warnings,
            "stats": {
                "file_count": all_stats["file_count"],
                "total_chars": all_stats["total_chars"],
                "article_count": len(set(all_stats["article_refs"])),
                "page_count": len(all_stats["page_numbers"])
            }
        }
    
    def _analyze_file(self, content: str, filename: str) -> Dict[str, Any]:
        """分析单个文件"""
        # 提取法条引用
        article_pattern = r'第[一二三四五六七八九十百千零]+条|第\d+条'
        articles = re.findall(article_pattern, content)
        
        # 标准化法条编号
        normalized_articles = []
        for article in articles:
            num = self._chinese_to_number(article)
            if num:
                normalized_articles.append(num)
        
        return {
            "char_count": len(content),
            "article_refs": normalized_articles
        }
    
    def _estimate_ocr_quality(self, content: str) -> float:
        """
        估计OCR质量分数
        基于以下指标：
        - 乱码字符比例
        - 常见OCR错误模式
        - 文本连贯性
        """
        if not content:
            return 0.0
        
        score = 1.0
        
        # 检查乱码字符（非中文、非英文、非标点、非数字）
        # ] 放在 [^ 之后第一个位置即为字面量；- 放末尾即为字面量；[ 在类内始终为字面量
        garbled_pattern = r'[^]\u4e00-\u9fa5a-zA-Z0-9\s\n\r\t.,;:!?，。；：！？、""''（）()【】《》<>{}/\\——_＿@＠#＃$＄%％^＆*＊+＋=＝|｜~～`｀[{}-]'
        garbled_chars = len(re.findall(garbled_pattern, content))
        garbled_ratio = garbled_chars / len(content) if content else 0
        score -= garbled_ratio * 2
        
        # 检查常见OCR错误
        ocr_errors = [
            (r'[〇○]', '零'),  # 数字零的变体
            (r'[—–]', '—'),    # 破折号统一
        ]
        
        # 检查段落连贯性（异常短的段落可能表示识别错误）
        lines = content.split('\n')
        very_short_lines = sum(1 for line in lines if len(line.strip()) < 3 and line.strip())
        short_line_ratio = very_short_lines / len(lines) if lines else 0
        score -= short_line_ratio * 0.5
        
        return max(0.0, min(1.0, score))
    
    def _extract_page_number(self, content: str, filename: str) -> int:
        """从内容或文件名提取页码"""
        # 尝试从内容中提取页码标记
        page_patterns = [
            r'-\s*(\d+)\s*-',  # - 123 -
            r'第\s*(\d+)\s*页',  # 第 123 页
            r'Page\s+(\d+)',  # Page 123
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, content)
            if match:
                return int(match.group(1))
        
        # 尝试从文件名提取
        # 假设文件名格式为: 001_章节名.md 或 page_001.md
        num_pattern = r'^(\d+)[_-]'
        match = re.search(num_pattern, filename)
        if match:
            return int(match.group(1))
        
        return None
    
    def _check_article_format(self, content: str) -> List[str]:
        """检查法条编号格式"""
        invalid = []
        
        # 查找可疑的法条引用（如"第 一 条"有空格）
        suspicious_pattern = r'第\s+[一二三四五六七八九十百千零\d]+\s*条'
        matches = re.findall(suspicious_pattern, content)
        
        for match in matches:
            if re.search(r'第\s', match):  # 有空格
                invalid.append(match)
        
        return invalid
    
    def _check_citation_integrity(self, article_refs: List[int]) -> List[int]:
        """检查引用完整性"""
        # 这里简化处理，实际应该检查引用是否指向存在的法条
        # 返回无法解析的引用（如编号为0或负数）
        broken = [ref for ref in article_refs if ref <= 0 or ref > 2000]
        return broken
    
    def _chinese_to_number(self, chinese_str: str) -> int:
        """将中文数字转换为阿拉伯数字（委托给 shared_utils）"""
        return chinese_to_number(chinese_str)


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        checker = QualityChecker()
        result = checker.check(Path(sys.argv[1]))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("用法: python quality_checker.py <md_dir>")
