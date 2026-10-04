#!/usr/bin/env python3
"""
格式探测器（Format Detector）

在 legal-kb-builder 每次运行时的第一道关卡：判断用户提供的输入文件是否已经是
Markdown。如果是 .md，就直接进入知识库构建流水线；否则，把控制权交给 parser_adapter，
按用户在 assets/parser-backends.yaml 中配置的顺序调用一个可用的文档解析后端。

支持格式（可扩展，见 --list-formats）：
  - Markdown        : .md / .markdown         → 直通
  - PDF             : .pdf                    → 需解析
  - Word            : .doc / .docx            → 需解析
  - PowerPoint      : .ppt / .pptx            → 需解析
  - 图片 / 扫描件    : .png / .jpg / .jpeg /
                     .webp / .tif / .tiff / .bmp → 需解析（OCR）
  - 纯文本          : .txt                    → 视为 .md 直通（可 --strict 关闭）

用法：
    python format_detector.py <input>              # 单个文件
    python format_detector.py <dir> --recursive     # 目录
    python format_detector.py <input> --json        # 机器可读输出

退出码：
    0 = 全部为 markdown，可直通
    1 = 存在需要解析的文件（非错误，只是提示上游走 parser_adapter）
    2 = 参数错误
    3 = 输入路径不存在
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List


# 后缀 → 分类
MARKDOWN_EXTS = {".md", ".markdown", ".mdown", ".mkd"}
TEXT_EXTS = {".txt"}
PDF_EXTS = {".pdf"}
WORD_EXTS = {".doc", ".docx"}
PPT_EXTS = {".ppt", ".pptx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
HTML_EXTS = {".html", ".htm"}


@dataclass
class FormatDecision:
    path: str
    ext: str
    category: str                 # 'markdown' | 'text' | 'pdf' | 'word' | 'ppt' | 'image' | 'html' | 'unknown'
    need_parsing: bool
    suggested_backends: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def classify(path: Path, strict_text: bool = False) -> FormatDecision:
    """
    根据后缀 + mimetype 判断文件类别，并给出建议的解析后端顺序。
    """
    if not path.exists():
        return FormatDecision(
            path=str(path),
            ext=path.suffix.lower(),
            category="unknown",
            need_parsing=False,
            reason="文件不存在",
        )

    if path.is_dir():
        return FormatDecision(
            path=str(path),
            ext="",
            category="unknown",
            need_parsing=False,
            reason="路径为目录，请配合 --recursive 使用",
        )

    ext = path.suffix.lower()

    # 直通类
    if ext in MARKDOWN_EXTS:
        return FormatDecision(
            path=str(path), ext=ext, category="markdown",
            need_parsing=False, reason="已是 Markdown，直通构建流水线",
        )
    if ext in TEXT_EXTS and not strict_text:
        return FormatDecision(
            path=str(path), ext=ext, category="text",
            need_parsing=False,
            reason="纯文本视同 Markdown（如需强制解析，请传 --strict）",
        )

    # 需要解析的类
    if ext in PDF_EXTS:
        return FormatDecision(
            path=str(path), ext=ext, category="pdf", need_parsing=True,
            suggested_backends=["mineru", "aliyun-bailian", "hehe-textin",
                                "paddleocr", "doc-parse-fallback"],
            reason="PDF，建议按后端配置文件顺序尝试",
        )
    if ext in WORD_EXTS:
        return FormatDecision(
            path=str(path), ext=ext, category="word", need_parsing=True,
            suggested_backends=["local-libreoffice", "mineru",
                                "aliyun-bailian", "hehe-textin"],
            reason="Word 文档，优先本地 LibreOffice/pandoc；其次 OCR 类后端",
        )
    if ext in PPT_EXTS:
        return FormatDecision(
            path=str(path), ext=ext, category="ppt", need_parsing=True,
            suggested_backends=["local-libreoffice", "mineru",
                                "aliyun-bailian"],
            reason="演示文稿，先转 PDF 再走 PDF 后端",
        )
    if ext in IMAGE_EXTS:
        return FormatDecision(
            path=str(path), ext=ext, category="image", need_parsing=True,
            suggested_backends=["paddleocr", "hehe-textin",
                                "aliyun-bailian", "mineru"],
            reason="图片/扫描件，优先 OCR 后端",
        )
    if ext in HTML_EXTS:
        return FormatDecision(
            path=str(path), ext=ext, category="html", need_parsing=True,
            suggested_backends=["local-html2md"],
            reason="HTML，走本地 html→md（可用 markdownify/pandoc）",
        )

    # 兜底：用 mimetype
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("text/"):
        return FormatDecision(
            path=str(path), ext=ext, category="text",
            need_parsing=False, reason=f"mimetype={guessed}，按纯文本直通",
        )

    return FormatDecision(
        path=str(path), ext=ext, category="unknown",
        need_parsing=True,
        suggested_backends=["mineru", "aliyun-bailian", "hehe-textin"],
        reason=f"未知后缀 {ext or '(空)'}，尝试通用后端",
    )


def walk_paths(root: Path, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not recursive:
        return
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Legal KB Builder — 格式探测器")
    ap.add_argument("input", nargs="?", help="文件或目录路径")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="输入为目录时递归遍历")
    ap.add_argument("--strict", action="store_true",
                    help="严格模式：.txt 也视为需要解析")
    ap.add_argument("--json", action="store_true",
                    help="以 JSON 输出全部决策，供上游脚本消费")
    ap.add_argument("--list-formats", action="store_true",
                    help="打印支持的文件类型清单后退出")
    args = ap.parse_args()

    if args.list_formats:
        print("markdown:", sorted(MARKDOWN_EXTS))
        print("text    :", sorted(TEXT_EXTS))
        print("pdf     :", sorted(PDF_EXTS))
        print("word    :", sorted(WORD_EXTS))
        print("ppt     :", sorted(PPT_EXTS))
        print("image   :", sorted(IMAGE_EXTS))
        print("html    :", sorted(HTML_EXTS))
        return 0

    if not args.input:
        ap.error("需要提供输入文件或目录（或使用 --list-formats）")

    root = Path(args.input).expanduser()
    if not root.exists():
        print(f"[format-detector] 输入不存在: {root}", file=sys.stderr)
        return 3

    decisions: List[FormatDecision] = []
    for p in walk_paths(root, args.recursive):
        decisions.append(classify(p, strict_text=args.strict))
    if not decisions and root.is_file():
        decisions.append(classify(root, strict_text=args.strict))

    if args.json:
        json.dump(
            {"decisions": [d.to_dict() for d in decisions]},
            sys.stdout, ensure_ascii=False, indent=2,
        )
        sys.stdout.write("\n")
    else:
        for d in decisions:
            flag = "MD-READY" if not d.need_parsing else "NEED-PARSE"
            print(f"[{flag}] {d.category:<8} {d.path}")
            if d.need_parsing and d.suggested_backends:
                print(f"           backends: {', '.join(d.suggested_backends)}")

    # 只要有一个需要解析，就返回 1 提示上游继续走 parser_adapter
    has_parse = any(d.need_parsing for d in decisions)
    return 1 if has_parse else 0


if __name__ == "__main__":
    sys.exit(main())
