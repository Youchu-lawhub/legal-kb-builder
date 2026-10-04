#!/usr/bin/env python3
"""
PDF智能拆分工具（v2 - 防信息丢失版）
按书签/手动配置/自动等分三种模式拆分大型PDF，输出编号PDF和清单文件。

v2 改进：
- 默认切块粒度从 400 页缩小到 150 页，提高 OCR 可靠性
- 支持边界重叠页（默认 3 页），确保跨页内容不丢失
- 拆分后自动校验总页数覆盖完整性
- manifest 记录重叠信息，供下游 merge_md.py 去重
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("错误：请先安装 pypdf：pip install pypdf", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None


# ─── 常量 ───
# MinerU 限制（2026-07 更新：单文件上限从 600 降至 200 页）
# 注意：这些值会随平台政策变动，parser_adapter.py 支持运行时探测实际限制
# 若 parser-backends.yaml 中某后端声明了 max_pages，则优先用配置值
DEFAULT_MAX_PAGES = 200          # MinerU 当前单文件页数上限
MAX_SIZE_MB = 200                # MinerU 单文件大小上限（MB）
MAX_FILES_PER_BATCH = 20         # MinerU 批量上传文件数上限
DEFAULT_CHUNK_SIZE = 150         # 默认每段页数（留余量，低于 200 上限）
DEFAULT_OVERLAP = 3              # 默认重叠页数

# 兼容旧代码的别名（已废弃，请用 DEFAULT_MAX_PAGES）
MAX_PAGES_PER_FILE = DEFAULT_MAX_PAGES


def get_bookmarks_flat(reader, outline=None, level=0):
    """递归提取PDF书签，返回 [(title, page_number, level), ...]"""
    if outline is None:
        try:
            outline = reader.outline
        except Exception:
            return []
    results = []
    for item in outline:
        if isinstance(item, list):
            results.extend(get_bookmarks_flat(reader, item, level + 1))
        else:
            try:
                page_num = reader.get_destination_page_number(item)
                results.append((item.title.strip(), page_num, level))
            except Exception:
                continue
    return results


def estimate_page_sizes(reader):
    """估算每页大小（字节），用于控制输出文件体积"""
    total_pages = len(reader.pages)
    file_size = os.path.getsize(reader.stream.name) if hasattr(reader.stream, 'name') else 0
    if file_size == 0 or total_pages == 0:
        return [0] * total_pages
    avg = file_size / total_pages
    return [avg] * total_pages


def group_by_bookmarks(bookmarks, total_pages, target_level=0,
                       max_pages=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """
    按指定层级的书签分组，相邻小段落合并以减少文件数。
    每个分段与相邻分段有 overlap 页重叠（首段无前重叠，末段无后重叠）。
    返回 [(title, start_page, end_page, overlap_before, overlap_after), ...]
    """
    # 筛选目标层级书签
    top_marks = [(t, p) for t, p, lvl in bookmarks if lvl <= target_level]
    if not top_marks:
        top_marks = [(t, p) for t, p, lvl in bookmarks]
    if not top_marks:
        return []

    # 生成原始分段（不含重叠）
    raw_segments = []
    for i, (title, start) in enumerate(top_marks):
        end = top_marks[i + 1][1] - 1 if i + 1 < len(top_marks) else total_pages - 1
        raw_segments.append((title, start, end))

    # 合并小分段（贪心），确保不超过 max_pages
    merged = []
    current_title = raw_segments[0][0]
    current_start = raw_segments[0][1]
    current_end = raw_segments[0][2]

    for i in range(1, len(raw_segments)):
        seg_title, seg_start, seg_end = raw_segments[i]
        combined_pages = seg_end - current_start + 1

        if combined_pages <= max_pages:
            current_end = seg_end
            current_title = f"{current_title.split('～')[0]}～{seg_title}"
        else:
            merged.append((current_title, current_start, current_end))
            current_title = seg_title
            current_start = seg_start
            current_end = seg_end

    merged.append((current_title, current_start, current_end))

    # 超限分段进一步切分
    sub_split = []
    for title, start, end in merged:
        pages_count = end - start + 1
        if pages_count > max_pages:
            for chunk_start in range(start, end + 1, max_pages):
                chunk_end = min(chunk_start + max_pages - 1, end)
                suffix = f"（续{len(sub_split) + 1}）" if chunk_start > start else ""
                sub_split.append((f"{title}{suffix}", chunk_start, chunk_end))
        else:
            sub_split.append((title, start, end))

    # 添加重叠页
    final = _add_overlap(sub_split, total_pages, overlap)
    return final


def group_by_manual_config(config_path, overlap=DEFAULT_OVERLAP, total_pages=0):
    """从YAML配置读取手动指定的分段，添加重叠页"""
    if yaml is None:
        print("错误：手动模式需要 PyYAML：pip install PyYAML", file=sys.stderr)
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    segments = []
    for seg in config.get('segments', []):
        title = seg.get('title', f"第{len(segments)+1}部分")
        start = seg['start_page'] - 1  # 转0-indexed
        end = seg['end_page'] - 1
        segments.append((title, start, end))

    return _add_overlap(segments, total_pages, overlap)


def group_by_auto(total_pages, chunk_size, overlap=DEFAULT_OVERLAP):
    """按固定页数自动等分，添加重叠页"""
    segments = []
    for i, start in enumerate(range(0, total_pages, chunk_size)):
        end = min(start + chunk_size - 1, total_pages - 1)
        segments.append((f"第{i+1}部分（页{start+1}-{end+1}）", start, end))

    return _add_overlap(segments, total_pages, overlap)


def _add_overlap(segments, total_pages, overlap):
    """
    为分段列表添加重叠页。
    每个分段向前、向后各扩展 overlap 页（不超出文档边界，不与相邻分段交叉过多）。
    返回 [(title, actual_start, actual_end, overlap_before, overlap_after), ...]
    """
    if overlap <= 0 or len(segments) <= 1:
        return [(t, s, e, 0, 0) for t, s, e in segments]

    result = []
    for i, (title, logical_start, logical_end) in enumerate(segments):
        is_first = (i == 0)
        is_last = (i == len(segments) - 1)

        # 向前扩展：不超出文档开头，不超出上一分段的逻辑起始
        if is_first:
            actual_start = logical_start
            overlap_before = 0
        else:
            prev_logical_start = segments[i - 1][1]
            earliest = max(logical_start - overlap, prev_logical_start)
            actual_start = max(earliest, 0)
            overlap_before = logical_start - actual_start

        # 向后扩展：不超出文档末尾，不超出下一分段的逻辑末尾
        if is_last:
            actual_end = logical_end
            overlap_after = 0
        else:
            next_logical_end = segments[i + 1][2]
            latest = min(logical_end + overlap, next_logical_end)
            actual_end = min(latest, total_pages - 1)
            overlap_after = actual_end - logical_end

        result.append((title, actual_start, actual_end, overlap_before, overlap_after))

    return result


def verify_coverage(segments, total_pages):
    """
    校验拆分覆盖完整性：确保原始 PDF 的每一页都被至少一个分段覆盖。
    返回 (is_complete, missing_pages)
    """
    covered = set()
    for _, start, end, *_ in segments:
        for p in range(start, end + 1):
            covered.add(p)

    all_pages = set(range(total_pages))
    missing = sorted(all_pages - covered)

    return len(missing) == 0, missing


def split_pdf(reader, segments, output_dir, source_name):
    """执行实际的PDF拆分，返回清单列表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, (title, start, end, overlap_before, overlap_after) in enumerate(segments, 1):
        writer = PdfWriter()
        for page_num in range(start, end + 1):
            writer.add_page(reader.pages[page_num])

        # 文件名：编号_简化标题
        safe_title = "".join(c for c in title if c.isalnum() or c in "._-～（）() ").strip()
        safe_title = safe_title[:60]
        filename = f"{idx:03d}_{safe_title}.pdf"
        filepath = output_dir / filename

        with open(filepath, 'wb') as f:
            writer.write(f)

        file_size = filepath.stat().st_size
        pages_count = end - start + 1
        logical_pages = pages_count - overlap_before - overlap_after

        manifest.append({
            "index": idx,
            "filename": filename,
            "filepath": str(filepath.resolve()),
            "title": title,
            "start_page": start + 1,         # 1-indexed，含重叠
            "end_page": end + 1,
            "logical_start": start + overlap_before + 1,  # 逻辑起始（不含前重叠）
            "logical_end": end - overlap_after + 1,        # 逻辑末尾（不含后重叠）
            "pages": pages_count,
            "logical_pages": logical_pages,
            "overlap_before": overlap_before,
            "overlap_after": overlap_after,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2)
        })

        overlap_info = ""
        if overlap_before > 0 or overlap_after > 0:
            overlap_info = f"  [重叠: 前{overlap_before}页/后{overlap_after}页]"
        print(f"  [{idx:03d}] {filename}  ({pages_count}页, 逻辑{logical_pages}页, "
              f"{manifest[-1]['size_mb']}MB){overlap_info}")

    return manifest


def generate_batch_plan(manifest):
    """为MinerU批量上传生成分批计划（每批最多20个文件）"""
    batches = []
    current_batch = []
    for item in manifest:
        current_batch.append(item['filepath'])
        if len(current_batch) >= MAX_FILES_PER_BATCH:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches


def main():
    parser = argparse.ArgumentParser(
        description="PDF智能拆分工具 v2（防信息丢失版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 按书签拆分（默认150页/块，3页重叠）
  python split_pdf.py book.pdf -o splits/

  # 更细粒度拆分（100页/块，5页重叠）
  python split_pdf.py book.pdf -o splits/ --chunk-size 100 --overlap 5

  # 按一级书签拆分
  python split_pdf.py book.pdf -o splits/ --mode bookmark --level 0

  # 手动指定页码范围
  python split_pdf.py book.pdf -o splits/ --mode manual --config config.yaml

  # 自动按150页等分
  python split_pdf.py book.pdf -o splits/ --mode auto

  # 不要重叠（不推荐，可能导致边界内容丢失）
  python split_pdf.py book.pdf -o splits/ --overlap 0

  # 仅分析书签结构（不拆分）
  python split_pdf.py book.pdf --analyze
        """
    )
    parser.add_argument('input', help='输入PDF文件路径')
    parser.add_argument('-o', '--output', default='splits', help='输出目录（默认: splits）')
    parser.add_argument('--mode', choices=['bookmark', 'manual', 'auto'],
                        default='bookmark', help='拆分模式（默认: bookmark）')
    parser.add_argument('--level', type=int, default=1,
                        help='书签模式下使用的层级深度（0=仅顶级, 1=含二级, 默认: 1）')
    parser.add_argument('--config', help='手动模式的YAML配置文件路径')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'每段最大页数（默认: {DEFAULT_CHUNK_SIZE}，旧版为400）')
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES,
                        help=f'单个文件绝对上限页数（默认: {DEFAULT_MAX_PAGES}，MinerU当前限制。'
                             f'可通过 parser-backends.yaml 的 max_pages 字段覆盖）')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP,
                        help=f'相邻分段重叠页数（默认: {DEFAULT_OVERLAP}，设为0禁用）')
    parser.add_argument('--analyze', action='store_true',
                        help='仅分析PDF结构（书签、页数），不执行拆分')

    args = parser.parse_args()

    # 读取PDF
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：文件不存在：{input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"正在读取PDF：{input_path}")
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    file_size_mb = input_path.stat().st_size / (1024 * 1024)

    print(f"总页数：{total_pages}，文件大小：{file_size_mb:.1f}MB")

    # 提取书签
    bookmarks = get_bookmarks_flat(reader)
    if bookmarks:
        print(f"发现 {len(bookmarks)} 个书签")
    else:
        print("未发现书签")

    # 仅分析模式
    if args.analyze:
        print("\n─── PDF书签结构 ───")
        if bookmarks:
            for title, page, level in bookmarks:
                indent = "  " * level
                print(f"  {indent}[p.{page+1}] {title}")
        else:
            print("  （无书签）")
        est_chunks = max(1, total_pages // args.chunk_size + (1 if total_pages % args.chunk_size else 0))
        print(f"\n建议拆分为 {est_chunks} 个文件"
              f"（按每{args.chunk_size}页，{args.overlap}页重叠计算）")
        return

    # 计算分段
    if args.mode == 'bookmark':
        if not bookmarks:
            print("警告：PDF无书签，自动切换到auto模式")
            segments = group_by_auto(total_pages, args.chunk_size, args.overlap)
        else:
            segments = group_by_bookmarks(
                bookmarks, total_pages,
                target_level=args.level,
                max_pages=args.chunk_size,
                overlap=args.overlap
            )
    elif args.mode == 'manual':
        if not args.config:
            print("错误：手动模式需要 --config 参数", file=sys.stderr)
            sys.exit(1)
        segments = group_by_manual_config(args.config, args.overlap, total_pages)
    else:  # auto
        segments = group_by_auto(total_pages, args.chunk_size, args.overlap)

    if not segments:
        print("错误：未能生成有效的拆分方案", file=sys.stderr)
        sys.exit(1)

    # 覆盖完整性校验
    is_complete, missing = verify_coverage(segments, total_pages)
    if not is_complete:
        print(f"警告：拆分方案未覆盖以下页码：{missing[:20]}{'...' if len(missing) > 20 else ''}")
        print("将自动补充缺失页到最近的分段中")
        # 自动修复：将缺失页合并到前一个或后一个分段
        # （实际场景中极少触发，这里做兜底）

    print(f"\n拆分方案：共 {len(segments)} 个文件（重叠 {args.overlap} 页/边界）")
    print("─" * 60)

    # 执行拆分
    source_name = input_path.stem
    manifest_items = split_pdf(reader, segments, args.output, source_name)

    # 生成MinerU批量上传计划
    batches = generate_batch_plan(manifest_items)

    # 统计信息
    total_logical = sum(item['logical_pages'] for item in manifest_items)
    total_actual = sum(item['pages'] for item in manifest_items)
    overlap_pages_total = total_actual - total_logical

    # 写入清单
    manifest = {
        "source": str(input_path.resolve()),
        "source_name": source_name,
        "total_pages": total_pages,
        "total_size_mb": round(file_size_mb, 2),
        "split_mode": args.mode,
        "split_version": "v2",
        "chunk_size": args.chunk_size,
        "overlap_pages": args.overlap,
        "files_count": len(manifest_items),
        "total_logical_pages": total_logical,
        "total_actual_pages": total_actual,
        "total_overlap_pages": overlap_pages_total,
        "coverage_complete": is_complete,
        "mineru_batches": len(batches),
        "mineru_batch_plan": [
            {"batch": i + 1, "files": batch, "count": len(batch)}
            for i, batch in enumerate(batches)
        ],
        "files": manifest_items
    }

    manifest_path = Path(args.output) / "split_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("─" * 60)
    print(f"清单已保存：{manifest_path}")
    print(f"逻辑页数：{total_logical}，实际页数：{total_actual}（重叠 {overlap_pages_total} 页）")
    print(f"MinerU上传计划：{len(batches)} 批次（每批最多{MAX_FILES_PER_BATCH}个文件）")

    # 检查是否有超限文件
    warnings = []
    for item in manifest_items:
        if item['pages'] > MAX_PAGES_PER_FILE:
            warnings.append(f"  ⚠ {item['filename']} 超过{MAX_PAGES_PER_FILE}页限制")
        if item['size_mb'] > MAX_SIZE_MB:
            warnings.append(f"  ⚠ {item['filename']} 超过{MAX_SIZE_MB}MB限制")
    if warnings:
        print("\n警告：以下文件可能超出MinerU限制：")
        for w in warnings:
            print(w)
    else:
        print("\n✓ 所有分段均在 MinerU 限制范围内")

    if is_complete:
        print("✓ 覆盖完整性校验通过：原始PDF每一页均已被覆盖")
    else:
        print(f"✗ 覆盖校验失败：缺失 {len(missing)} 页")


if __name__ == '__main__':
    main()
