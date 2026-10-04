#!/usr/bin/env python3
"""
Markdown知识库合并与索引生成工具（v2 - 边界去重版）
将MinerU转换后的多个MD文件合并为结构化知识库，生成JSON索引和目录。

v2 改进：
- 从 split_manifest.json 读取重叠页信息
- 相邻分段的重叠区域自动去重，防止内容重复
- 去重采用文本相似度匹配，容忍 OCR 轻微差异

v2.1 改进：
- 新增 --clean 选项：合并前使用 TextCleaner 清洗每个 MD 文件
- 使用 shared_utils 消除重复的 chinese_to_arabic 实现
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

try:
    import yaml
except ImportError:
    yaml = None

# 添加当前目录以导入兄弟模块
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shared_utils import chinese_to_number as _cn_to_num


# ─── 去重相关常量 ───
OVERLAP_SIMILARITY_THRESHOLD = 0.6   # 文本相似度阈值，超过即判定为重叠
OVERLAP_SEARCH_LINES = 80            # 在文件首尾搜索重叠的最大行数


def read_manifest(manifest_path):
    """读取split_manifest.json获取文件顺序和重叠信息"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_md_files(input_path, manifest=None):
    """
    查找 Markdown 输入。接受单个 .md 文件或目录；如果有 manifest，
    按 manifest 顺序读取目录中的文件。
    """
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() != '.md':
            return []
        return [{'md_path': str(input_path), 'title': input_path.stem, 'index': 1,
                 'overlap_before': 0, 'overlap_after': 0}]

    if not input_path.is_dir():
        return []
    md_dir = input_path
    if manifest:
        ordered_files = []
        for item in manifest.get('files', []):
            pdf_name = Path(item['filename']).stem
            candidates = list(md_dir.glob(f"*{pdf_name}*/*.md")) + \
                         list(md_dir.glob(f"*{pdf_name}*.md")) + \
                         list(md_dir.glob(f"{pdf_name}*.md"))
            if candidates:
                ordered_files.append({
                    'md_path': str(candidates[0]),
                    'title': item.get('title', pdf_name),
                    'index': item.get('index', 0),
                    'start_page': item.get('start_page', 0),
                    'end_page': item.get('end_page', 0),
                    'overlap_before': item.get('overlap_before', 0),
                    'overlap_after': item.get('overlap_after', 0),
                })
            else:
                print(f"  警告：未找到 {pdf_name} 对应的MD文件", file=sys.stderr)
        return ordered_files
    else:
        md_files = sorted(md_dir.glob('**/*.md'), key=lambda p: p.name)
        return [{'md_path': str(f), 'title': f.stem, 'index': i,
                 'overlap_before': 0, 'overlap_after': 0}
                for i, f in enumerate(md_files, 1)]


def _read_file_content(filepath):
    """读取文件内容，自动处理编码"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='gbk', errors='replace') as f:
            return f.read()


def _normalize_text(text):
    """标准化文本以便比较（去除空白差异、OCR微小差异）"""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除 Markdown 格式标记（不影响内容比较）
    text = re.sub(r'[#*_`\-=|]', '', text)
    return text


def deduplicate_overlap(prev_content, curr_content, overlap_pages_hint):
    """
    对相邻两个分段的重叠区域进行去重。

    策略：
    1. 取前一个文件的最后 N 行作为"尾部"
    2. 取当前文件的前 N 行作为"头部"
    3. 在"头部"中找到与"尾部"最长匹配的重叠区域
    4. 从当前文件中裁掉重叠的部分

    参数：
        prev_content: 前一个文件的完整内容
        curr_content: 当前文件的完整内容
        overlap_pages_hint: 预期重叠页数（来自manifest，用于估算搜索范围）

    返回：
        去重后的 curr_content
    """
    if overlap_pages_hint <= 0:
        return curr_content

    prev_lines = prev_content.split('\n')
    curr_lines = curr_content.split('\n')

    if len(prev_lines) < 5 or len(curr_lines) < 5:
        return curr_content

    # 估算重叠区域的行数范围（每页约30-50行文本）
    estimated_overlap_lines = min(overlap_pages_hint * 50, OVERLAP_SEARCH_LINES)

    # 取前一文件的尾部
    tail_start = max(0, len(prev_lines) - estimated_overlap_lines)
    tail_text = _normalize_text('\n'.join(prev_lines[tail_start:]))

    if len(tail_text) < 20:
        return curr_content

    # 在当前文件的头部寻找最佳切割点
    best_cut = 0
    best_ratio = 0

    # 逐渐扩大搜索窗口，找到最长的重叠
    search_limit = min(len(curr_lines), estimated_overlap_lines + 20)

    for cut_point in range(10, search_limit, 5):
        head_text = _normalize_text('\n'.join(curr_lines[:cut_point]))

        if len(head_text) < 20:
            continue

        # 比较尾部和头部的相似度
        # 用尾部的后半段和头部的前半段比较（重叠区域在中间相遇）
        tail_segment = tail_text[-min(len(tail_text), len(head_text)):]
        head_segment = head_text[:min(len(tail_text), len(head_text))]

        ratio = SequenceMatcher(None, tail_segment, head_segment).ratio()

        if ratio > best_ratio and ratio >= OVERLAP_SIMILARITY_THRESHOLD:
            best_ratio = ratio
            best_cut = cut_point

    if best_cut > 0:
        # 找到重叠区域，裁掉当前文件的重叠部分
        trimmed = '\n'.join(curr_lines[best_cut:])
        overlap_line_count = best_cut
        print(f"    → 去重：裁掉前 {overlap_line_count} 行（相似度 {best_ratio:.2f}）")
        return trimmed
    else:
        # 未找到明显重叠，保留完整内容（宁可重复也不丢失）
        print(f"    → 未检测到明显重叠，保留完整内容")
        return curr_content


def extract_headings(content):
    """从MD内容中提取标题结构"""
    headings = []
    for line in content.split('\n'):
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append({'level': level, 'title': title})
    return headings


def extract_legal_articles(content):
    """从内容中提取法律条文编号（如 第XXX条）"""
    articles = []
    patterns = [
        r'第(\d+)条',
        r'第([一二三四五六七八九十百千零]+)条',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, content):
            article_text = m.group(0)
            pos = m.start()
            line_num = content[:pos].count('\n') + 1
            articles.append({
                'article': article_text,
                'raw_number': m.group(1),
                'line': line_num,
                'char_offset': pos
            })
    return articles


def chinese_to_arabic(cn_num):
    """中文数字转阿拉伯数字（委托给 shared_utils）"""
    result = _cn_to_num(cn_num)
    return result if result is not None else 0


def merge_into_volumes(md_files_info, max_size_bytes, kb_name, legal_mode=False, clean_mode=False):
    """
    将MD文件合并为知识库卷，每卷不超过max_size_bytes。
    v2: 合并前对重叠区域去重。
    v2.1: 可选 clean_mode，合并前用 TextCleaner 清洗每个文件。
    返回 [(volume_title, volume_content, sections_info), ...]
    """
    # 延迟导入 TextCleaner（仅在 clean_mode 时需要）
    text_cleaner = None
    if clean_mode:
        try:
            from text_cleaner import TextCleaner
            text_cleaner = TextCleaner()
            print("  TextCleaner 已启用，将对每个文件进行清洗")
        except ImportError:
            print("  警告：无法导入 TextCleaner，跳过清洗步骤", file=sys.stderr)

    volumes = []
    current_content = ""
    current_sections = []
    current_size = 0
    volume_idx = 0

    prev_raw_content = None  # 用于去重比较

    for i, info in enumerate(md_files_info):
        md_path = info['md_path']
        raw_content = _read_file_content(md_path)

        title = info['title']
        overlap_before = info.get('overlap_before', 0)

        # v2: 对重叠区域去重
        if prev_raw_content is not None and overlap_before > 0:
            print(f"  处理重叠去重：{Path(md_path).name}（前重叠 {overlap_before} 页）")
            content = deduplicate_overlap(prev_raw_content, raw_content, overlap_before)
        else:
            content = raw_content

        # v2.1: TextCleaner 清洗
        if text_cleaner is not None:
            clean_result = text_cleaner.clean(content)
            if clean_result.stats:
                print(f"    清洗 {Path(md_path).name}: {clean_result.stats}")
            content = clean_result.cleaned_text

        # 保存原始内容用于下一个文件的去重
        prev_raw_content = raw_content

        content_size = len(content.encode('utf-8'))

        # 如果当前卷加上新内容会超限，先提交当前卷
        if current_content and (current_size + content_size > max_size_bytes):
            volume_idx += 1
            vol_title = _make_volume_title(volume_idx, current_sections, kb_name)
            volumes.append((vol_title, current_content, current_sections))
            current_content = ""
            current_sections = []
            current_size = 0

        # 来源信息保留在 JSON 索引的 sections 中，不写入正文。否则分隔符和
        # HTML 注释会成为每个卷的开头，直接污染检索片段。
        if current_content and not current_content.endswith('\n'):
            current_content += '\n'
        if current_content:
            current_content += '\n'
        current_content += content.lstrip('\n')
        current_size += content_size

        section_info = {
            'title': title,
            'source_file': Path(md_path).name,
            'start_page': info.get('start_page', 0),
            'end_page': info.get('end_page', 0),
            'headings': extract_headings(content),
        }

        if legal_mode:
            articles = extract_legal_articles(content)
            if articles:
                section_info['articles'] = articles
                section_info['article_range'] = {
                    'first': articles[0]['article'],
                    'last': articles[-1]['article']
                }

        current_sections.append(section_info)

    # 提交最后一卷
    if current_content:
        volume_idx += 1
        vol_title = _make_volume_title(volume_idx, current_sections, kb_name)
        volumes.append((vol_title, current_content, current_sections))

    return volumes


def _make_volume_title(idx, sections, kb_name):
    """生成卷标题"""
    if not sections:
        return f"第{idx}部分"
    first = sections[0]['title']
    last = sections[-1]['title']
    first_short = first.split('～')[0].split('（')[0][:20]
    last_short = last.split('～')[-1].split('（')[0][:20]
    if first_short == last_short:
        return f"{kb_name}_{first_short}"
    return f"{kb_name}_{first_short}至{last_short}"


def generate_index(volumes, kb_dir, kb_name, legal_mode=False):
    """生成知识库索引JSON"""
    index = {
        "name": kb_name,
        "base_path": str(kb_dir.resolve()),
        "total_volumes": len(volumes),
        "legal_mode": legal_mode,
        "merge_version": "v2.2",
        "volumes": []
    }

    all_articles = {}

    for vol_title, _, sections in volumes:
        safe_title = "".join(c for c in vol_title if c.isalnum() or c in "._-～至（）() ").strip()
        filename = f"{safe_title}.md"

        vol_info = {
            "filename": filename,
            "title": vol_title,
            "sections": []
        }

        for sec in sections:
            sec_info = {
                "title": sec['title'],
                "source_file": sec['source_file'],
                "page_range": f"{sec.get('start_page', '?')}-{sec.get('end_page', '?')}",
                "top_headings": [h['title'] for h in sec.get('headings', []) if h['level'] <= 2][:10]
            }
            if legal_mode and 'article_range' in sec:
                sec_info['article_range'] = sec['article_range']
                for art in sec.get('articles', []):
                    all_articles[art['article']] = filename
            vol_info['sections'].append(sec_info)

        index['volumes'].append(vol_info)

    if legal_mode and all_articles:
        index['article_index'] = all_articles

    return index


def generate_toc(volumes, kb_name):
    """生成人可读的目录MD文件"""
    lines = [
        f"# {kb_name} - 知识库目录\n",
        f"本知识库共 {len(volumes)} 个文件。\n",
    ]

    for vol_title, _, sections in volumes:
        safe_title = "".join(c for c in vol_title if c.isalnum() or c in "._-～至（）() ").strip()
        filename = f"{safe_title}.md"
        lines.append(f"\n## {vol_title}")
        lines.append(f"文件：`{filename}`\n")

        for sec in sections:
            page_info = ""
            if sec.get('start_page') and sec.get('end_page'):
                page_info = f"（原书p.{sec['start_page']}-{sec['end_page']}）"
            lines.append(f"- **{sec['title']}** {page_info}")

            for h in sec.get('headings', []):
                if h['level'] <= 2:
                    indent = "  " * h['level']
                    lines.append(f"  {indent}- {h['title']}")

            if 'article_range' in sec:
                rng = sec['article_range']
                lines.append(f"  - 法条范围：{rng['first']} ~ {rng['last']}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="MD知识库合并与索引生成工具 v2（边界去重版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础合并（使用拆分清单，自动处理重叠去重）
  python merge_md.py md_files/ -o kb/ --name "民法典评注" --manifest splits/split_manifest.json

  # 法律模式（提取法条索引）
  python merge_md.py md_files/ -o kb/ --name "民法典评注" --legal --manifest splits/split_manifest.json

  # 自定义卷大小
  python merge_md.py md_files/ -o kb/ --name "教材" --max-size 800KB
        """
    )
    parser.add_argument('input_path', help='Markdown 文件或包含 Markdown 文件的目录')
    parser.add_argument('-o', '--output', default='knowledge_base', help='输出知识库目录')
    parser.add_argument('--name', required=True, help='知识库名称')
    parser.add_argument('--manifest', help='split_manifest.json路径（保持原始顺序+重叠去重）')
    parser.add_argument('--max-size', default='500KB',
                        help='单个知识库文件最大大小（如 500KB, 1MB, 默认: 500KB）')
    parser.add_argument('--legal', action='store_true',
                        help='启用法律文本模式（提取条文编号建立索引）')
    parser.add_argument('--clean', action='store_true',
                        help='启用文本清洗（OCR纠错、页码移除、空格规范化等）')

    args = parser.parse_args()

    # 解析文件大小
    size_str = args.max_size.upper()
    if size_str.endswith('MB'):
        max_bytes = int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('KB'):
        max_bytes = int(float(size_str[:-2]) * 1024)
    else:
        max_bytes = int(size_str)

    # 读取manifest
    manifest = None
    if args.manifest:
        manifest = read_manifest(args.manifest)
        version = manifest.get('split_version', 'v1')
        overlap = manifest.get('overlap_pages', 0)
        print(f"已加载清单：{args.manifest}（{manifest.get('files_count', '?')} 个文件, "
              f"版本 {version}, 重叠 {overlap} 页/边界）")

    # 查找MD文件
    md_files_info = find_md_files(args.input_path, manifest)
    if not md_files_info:
        print("错误：未找到任何MD文件", file=sys.stderr)
        sys.exit(1)
    print(f"找到 {len(md_files_info)} 个MD文件")

    # 合并（v2自动处理重叠去重）
    print(f"正在合并（单卷上限：{args.max_size}）...")
    volumes = merge_into_volumes(md_files_info, max_bytes, args.name, args.legal, args.clean)
    print(f"合并为 {len(volumes)} 个知识库文件")

    # 输出
    kb_dir = Path(args.output)
    kb_dir.mkdir(parents=True, exist_ok=True)

    for vol_title, vol_content, _ in volumes:
        safe_title = "".join(c for c in vol_title if c.isalnum() or c in "._-～至（）() ").strip()
        filename = f"{safe_title}.md"
        filepath = kb_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(vol_content)
        size_kb = filepath.stat().st_size / 1024
        print(f"  {filename}  ({size_kb:.0f}KB)")

    # 生成索引
    index = generate_index(volumes, kb_dir, args.name, args.legal)
    index_path = kb_dir / '_知识库索引.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"索引已保存：{index_path}")

    # 生成目录
    toc = generate_toc(volumes, args.name)
    toc_path = kb_dir / '_目录.md'
    with open(toc_path, 'w', encoding='utf-8') as f:
        f.write(toc)
    print(f"目录已保存：{toc_path}")

    print("\n" + "=" * 60)
    print("知识库生成完成！")
    print(f"目录：{kb_dir.resolve()}")
    print(f"索引：{index_path.resolve()}")
    print(f"文件数：{len(volumes)} + 索引 + 目录")
    if args.legal:
        article_count = len(index.get('article_index', {}))
        print(f"法条索引：{article_count} 条")
    print("=" * 60)


if __name__ == '__main__':
    main()
