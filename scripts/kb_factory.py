#!/usr/bin/env python3
"""
法律知识库工厂主入口（KB Factory）

把 legal-kb-builder 的双层路由 + 四条流水线（book_kb / case_kb / qa_kb / agentic）
编排成一个统一的"丢进来就建好库"的入口：

    任意语料（PDF/DOCX/MD/TXT/JSON/图片）
       │
    格式路由（format_detector + parser_adapter）  → 全部转成 md
       │
    数据源路由（material_router）                  → book_kb / case_kb / qa_kb / agentic / direct
       │
    对应流水线自动构建
       │
    可检索的知识库

使用方式：
    # 一键构建
    python3 scripts/kb_factory.py build ~/legal_materials/合同法FAQ.pdf \\
        --output ~/kbs/合同法问答库 --name "合同法FAQ"

    # 只看路由决策（不实际构建）
    python3 scripts/kb_factory.py route ~/legal_materials/合同法FAQ.pdf

    # 批量构建（整个目录）
    python3 scripts/kb_factory.py build ~/legal_materials/ --output ~/kbs/ --batch
"""

import argparse
import tempfile
import hashlib
import uuid
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from material_router import MaterialRouter, RoutingDecision
from shared_utils import file_lock, atomic_write_json, vector_index_exists


class KBFactory:
    """
    法律知识库工厂

    编排格式路由 → 数据源路由 → 流水线构建的完整流程。
    """

    def __init__(self):
        self.router = MaterialRouter()

    # ─── 路由（不构建） ───

    def route(self, input_path: str) -> RoutingDecision:
        """
        对输入文件执行双层路由，返回决策结果（不实际构建）。

        如果输入是非 md 格式，会先检测是否需要解析，并在 suggested_params 中标注。
        """
        path = Path(input_path)
        if not path.exists():
            return RoutingDecision(
                target_skill="unknown",
                reasoning=f"文件不存在: {input_path}",
            )
        return self.router.route(input_path)

    # ─── 构建（路由 + 构建） ───

    def build(self, input_path, output_dir, name="", default_category="", skip_parse=False,
              kb_type=None, book_type=None):
        """逐文件准备并校验输入；失败不发布构建成功状态。"""
        path, out = Path(input_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
        try:
            if not path.exists():
                raise FileNotFoundError(f"输入不存在: {path}")
            if out == path or (path.is_dir() and (out == path or path in out.parents)):
                raise ValueError("输出目录必须位于输入目录之外，避免再次导入生成物")
            with tempfile.TemporaryDirectory(prefix="legal-kb-parse-") as temporary:
                prepared = self._ensure_markdown(path, skip_parse, Path(temporary))
                if prepared is None:
                    raise ValueError(f"没有可处理的材料: {path}")
                if prepared.is_dir():
                    decisions = [self.router.route(str(f)) for f in sorted(prepared.glob("*.md"))]
                    if not kb_type and len({d.target_skill for d in decisions}) != 1:
                        raise ValueError("目录包含不同材料类型，请使用 --batch 或显式 --type")
                    decision = decisions[0]
                else:
                    decision = self.router.route(str(prepared))
                target = kb_type or decision.target_skill
                print(f"路由决策: {target}；{decision.reasoning}")
                if target == "qa_kb":
                    result = self._build_qa_kb(prepared, out, name, default_category)
                elif target == "case_kb":
                    result = self._build_case_kb(prepared, out, name)
                elif target == "book_kb":
                    result = self._build_book_kb(prepared, out, name or path.stem, book_type or decision.book_type)
                elif target == "agentic":
                    # 复用已经解析的产物，绝不再次解析原始 PDF。
                    result = self._build_agentic(prepared, out, name or path.stem)
                elif target == "direct":
                    result = self._build_direct(prepared, out)
                else:
                    raise ValueError(f"不支持的路由: {target}")
                result["status"] = "failed" if "error" in result else (
                    "degraded" if result.get("warnings") else "ok")
                result["inputs"] = self._prepared_inputs
                return result
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "input_file": str(path)}

    def build_batch(
        self,
        input_dir: str,
        output_dir: str,
        skip_parse: bool = False,
    ) -> List[Dict[str, Any]]:
        """批量构建：对目录下每个文件分别路由和构建"""
        dir_path = Path(input_dir).expanduser().resolve()
        batch_output = Path(output_dir).expanduser().resolve()
        if batch_output == dir_path or dir_path in batch_output.parents:
            return [{"status": "failed", "error": "批量输出目录必须位于输入目录之外"}]
        if not dir_path.is_dir():
            print(f"输入不是目录: {input_dir}")
            return [{"status": "failed", "error": f"输入不是目录: {input_dir}"}]

        supported = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt", ".json",
                     ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
                     ".ppt", ".pptx", ".html", ".htm"}
        files = sorted(f for f in dir_path.rglob("*") if f.is_file()
                       and not f.is_symlink() and f.suffix.lower() in supported
                       and not any(part.startswith(".") or part == "_parsed_md" or part.startswith("_parse_temp_")
                                   for part in f.relative_to(dir_path).parts))
        print(f"发现 {len(files)} 个文件")
        if not files:
            return [{"status": "failed", "error": "输入目录为空"}]
        results = []
        for f in files:
            print(f"\n{'='*60}")
            print(f"处理: {f.name}")
            print(f"{'='*60}")
            name = f.stem
            relative = f.relative_to(dir_path)
            # 包含扩展名，避免同名 PDF/MD 与子目录覆盖。
            out = Path(output_dir) / relative.parent / f.name
            result = self.build(str(f), str(out), name=name, skip_parse=skip_parse)
            result["input_file"] = str(f)
            results.append(result)

        return results

    # ─── 格式路由 + 解析 ───

    def _ensure_markdown(self, path, skip_parse=False, prepared_dir=None):
        """转换每个输入；任何失败都会停止本次合并构建。源目录只读。"""
        if not path.exists():
            raise FileNotFoundError(str(path))
        # 供内部/测试调用；build 始终提供生命周期受控的临时目录。
        if prepared_dir is None:
            if path.is_file() and path.suffix.lower() == ".md":
                return path
            raise ValueError("转换输入需要 prepared_dir；请使用 build()")
        files = sorted(f for f in path.rglob("*") if f.is_file()
                       and not f.is_symlink()
                       and not any(p.startswith(".") or p == "_parsed_md" or p.startswith("_parse_temp_")
                                   for p in f.relative_to(path).parts)) if path.is_dir() else [path]
        if not files:
            return None
        prepared_dir.mkdir(parents=True, exist_ok=True)
        self._prepared_inputs = []
        for source in files:
            relative = source.relative_to(path) if path.is_dir() else Path(source.name)
            identity = hashlib.sha256(str(relative).encode()).hexdigest()[:12]
            stem = re.sub(r"[^\w.-]", "_", source.stem)
            destination = prepared_dir / f"{stem}-{identity}.md"
            suffix = source.suffix.lower()
            if suffix in {".md", ".txt", ".markdown"}:
                destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                state = "copied"
            elif suffix == ".json":
                # JSON 问答无需解析后端，转换为 QAParser 支持的标题格式。
                data = json.loads(source.read_text(encoding="utf-8"))
                records = data if isinstance(data, list) else data.get("qa_pairs", data.get("items", [data]))
                if not isinstance(records, list) or not records:
                    raise ValueError(f"JSON 不是非空问答列表: {source}")
                sections = []
                for record in records:
                    q, answer = record.get("question"), record.get("answer")
                    if not isinstance(q, str) or not isinstance(answer, str) or not q.strip() or not answer.strip():
                        raise ValueError(f"JSON 问答必须有非空 question/answer: {source}")
                    sections.append(f"Q: {q}\nA: {answer}")
                destination.write_text("\n\n".join(sections), encoding="utf-8")
                state = "converted_json"
            else:
                if skip_parse:
                    raise ValueError(f"--skip-parse 不允许未解析输入: {source}")
                if suffix not in {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
                                  ".ppt", ".pptx", ".html", ".htm"}:
                    raise ValueError(f"不支持的输入类型: {source}")
                parse_dir = prepared_dir / ("parse-" + identity)
                parse_dir.mkdir()
                process = subprocess.run([sys.executable, str(_SCRIPT_DIR / "parser_adapter.py"),
                                          str(source), "-o", str(parse_dir)],
                                         capture_output=True, text=True, timeout=600)
                if process.returncode:
                    raise RuntimeError(f"解析失败 {source}: {(process.stderr or process.stdout)[:1000]}")
                produced = sorted(f for f in parse_dir.rglob("*.md") if not f.name.startswith("_"))
                if not produced:
                    raise RuntimeError(f"解析无 Markdown 产物: {source}")
                destination.write_text("\n\n".join(f.read_text(encoding="utf-8") for f in produced), encoding="utf-8")
                shutil.rmtree(parse_dir)
                state = "parsed"
            self._prepared_inputs.append({"source": str(source), "prepared": destination.name, "status": state})
        produced = sorted(prepared_dir.glob("*.md"))
        return produced[0] if len(produced) == 1 else prepared_dir

    # ─── 各流水线构建 ───

    def _build_qa_kb(
        self, md_path: Path, out_dir: Path, name: str, default_category: str
    ) -> Dict[str, Any]:
        """构建问答知识库"""
        from qa_kb import QAKnowledgeBase

        kb = QAKnowledgeBase(str(out_dir))
        if not (out_dir / "config.yaml").exists():
            kb.init(name or "问答知识库")

        if md_path.is_dir():
            count = kb.add_batch(str(md_path), default_category=default_category)
        else:
            count = kb.add_file(str(md_path), default_category=default_category)
            kb.rebuild_indices()

        stats = kb.stats()
        if not stats.get("total_qa"):
            raise ValueError("没有解析出有效问答对")
        return {
            "warnings": getattr(kb, "build_warnings", []),
            "kb_type": "qa_kb",
            "output_dir": str(out_dir),
            "qa_count": stats.get("total_qa", 0),
            "has_indices": stats.get("has_indices", False),
            "categories": stats.get("categories", {}),
        }

    def _build_case_kb(self, md_path: Path, out_dir: Path, name: str) -> Dict[str, Any]:
        """构建裁判文书知识库"""
        from judgment_kb import JudgmentKnowledgeBase

        kb = JudgmentKnowledgeBase(str(out_dir))
        if not (out_dir / "config.yaml").exists():
            kb.init(name or "裁判文书知识库")

        if md_path.is_dir():
            kb.add_batch(str(md_path))
        else:
            kb.add(str(md_path))
            kb.rebuild_indices()

        stats = kb.stats()
        if not stats.get("total_judgments"):
            raise ValueError("没有导入有效裁判文书")
        return {
            "warnings": getattr(kb, "build_warnings", []),
            "kb_type": "case_kb",
            "output_dir": str(out_dir),
            "judgment_count": stats.get("total_judgments", 0),
            "courts": stats.get("court_count", 0),
            "causes": stats.get("cause_count", 0),
        }

    def _build_book_kb(
        self, md_path: Path, out_dir: Path, name: str, book_type: str
    ) -> Dict[str, Any]:
        """构建书籍知识库，统一使用 <kb>/books/<book>/ 目录契约。"""
        from legal_kb import LegalKnowledgeBase

        display_name = name or md_path.stem or "法律书籍知识库"
        book_name = "".join(c for c in display_name if c.isalnum() or c in "._-（）() ").strip()
        book_name = book_name or "book"
        if book_name in {".", ".."}:
            raise ValueError("无效书籍名称")
        book_dir = out_dir / "books" / book_name
        books_dir = book_dir.parent
        books_dir.mkdir(parents=True, exist_ok=True)
        kb = LegalKnowledgeBase(str(out_dir))
        if not (out_dir / "config.yaml").exists():
            kb.init("法律知识库")

        generation = uuid.uuid4().hex
        staging = books_dir / (".staging-" + generation)
        backup = books_dir / (".backup-" + generation)
        try:
            with file_lock(out_dir / ".book-build.lock", timeout=3600):
                cmd = [sys.executable, str(_SCRIPT_DIR / "merge_md.py"), str(md_path),
                       "-o", str(staging), "--name", display_name, "--legal", "--clean"]
                merged = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if merged.returncode != 0:
                    raise RuntimeError(f"合并入库失败: {(merged.stderr or merged.stdout)[:1000]}")
                self._write_heading_index(staging, display_name, book_type or "monograph")
                # Validate every index before replacing the published book.
                index_report = kb.build_search_index_for_directory(staging)
                if book_dir.exists():
                    os.replace(book_dir, backup)
                try:
                    os.replace(staging, book_dir)
                    kb.register_merged_book(book_name, book_type or "monograph")
                except Exception:
                    if book_dir.exists():
                        shutil.rmtree(book_dir, ignore_errors=True)
                    if backup.exists():
                        os.replace(backup, book_dir)
                    raise
                shutil.rmtree(backup, ignore_errors=True)
            return {"kb_type": "book_kb", "warnings": index_report["warnings"],
                    "book_type": book_type, "output_dir": str(out_dir),
                    "book_dir": str(book_dir), "book": book_name,
                    "has_index": vector_index_exists(book_dir) or (book_dir / "_bm25_corpus.json").exists()}
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if backup.exists() and not book_dir.exists():
                os.replace(backup, book_dir)
            return {"error": str(exc), "output_dir": str(out_dir),
                    "book_dir": str(book_dir)}

    @staticmethod
    def _write_heading_index(book_dir: Path, book_name: str, book_type: str) -> None:
        """为章节型书籍生成可定位的 Markdown 标题索引。

        合并器只记录来源文件，不会把原始 ``#`` 标题变成检索锚点。对
        教材/专著，保存标题行号能让“标题型查询”优先定位到实际章节，而
        非被全文高频词淹没。条文驱动书籍仍交由 EnhancedIndex 的目录索引。
        """
        if book_type not in {"textbook", "monograph"}:
            return

        entries = {}
        for md_file in sorted(book_dir.glob("*.md")):
            if md_file.name.startswith("_"):
                continue
            for line_no, line in enumerate(md_file.read_text(encoding="utf-8").splitlines(), 1):
                match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
                if not match:
                    continue
                title = match.group(1).strip()
                if not title:
                    continue
                key = title
                if key in entries:
                    key = f"{title} ({md_file.name}:{line_no})"
                entries[key] = {"file": md_file.name, "line": line_no, "title": title}

        index = {
            "book_info": {"name": book_name},
            "book_type": book_type,
            "index_version": "v2_heading_index",
            "chapters": {
                "description": "由 Markdown 标题生成的章节锚点",
                "total_count": len(entries),
                "entries": entries,
            },
        }
        atomic_write_json(book_dir / "_book_index.json", index)

    def _build_agentic(self, source_path: Path, out_dir: Path, name: str) -> Dict[str, Any]:
        """Agentic 模式：转换到缓存目录"""
        cache_dir = out_dir / ".agentic_cache" / (name or source_path.stem)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_file() and source_path.suffix.lower() not in (".md", ".txt"):
            process = subprocess.run(
                [sys.executable, str(_SCRIPT_DIR / "parser_adapter.py"),
                 str(source_path), "-o", str(cache_dir)],
                capture_output=True, text=True, timeout=300,
            )
            if process.returncode:
                error = f"解析失败 {source_path}: {(process.stderr or process.stdout)[:1000]}"
                print(f"⚠️  agentic 解析失败: {error}", file=sys.stderr)
                return {
                    "kb_type": "agentic",
                    "error": error,
                    "cache_dir": str(cache_dir),
                }
        elif source_path.is_file():
            shutil.copy2(str(source_path), str(cache_dir / source_path.name))
        elif source_path.is_dir():
            for md_file in source_path.rglob("*.md"):
                shutil.copy2(str(md_file), str(cache_dir / md_file.name))

        md_count = len(list(cache_dir.glob("*.md")))
        return {
            "kb_type": "agentic",
            "cache_dir": str(cache_dir),
            "md_count": md_count,
        }

    def _build_direct(self, md_path: Path, out_dir: Path) -> Dict[str, Any]:
        """方法论/参考文档：直接放置"""
        out_dir.mkdir(parents=True, exist_ok=True)
        if md_path.is_dir():
            copied = []
            for source in md_path.rglob("*.md"):
                dest = out_dir / source.name
                shutil.copy2(str(source), str(dest))
                copied.append(str(dest))
            return {"kb_type": "direct", "output_dir": str(out_dir), "files": copied}
        dest = out_dir / md_path.name
        shutil.copy2(str(md_path), str(dest))
        return {
            "kb_type": "direct",
            "output_dir": str(out_dir),
            "file": str(dest),
        }


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="法律知识库工厂 — 一键路由 + 构建"
    )
    subparsers = parser.add_subparsers(dest="command")

    # route 子命令
    route_p = subparsers.add_parser("route", help="只路由不构建")
    route_p.add_argument("input", help="输入文件或目录")

    # build 子命令
    build_p = subparsers.add_parser("build", help="一键构建知识库")
    build_p.add_argument("input", help="输入文件或目录")
    build_p.add_argument("--output", "-o", required=True, help="知识库输出目录")
    build_p.add_argument("--name", default="", help="知识库名称")
    build_p.add_argument("--category", default="", help="问答库默认分类")
    build_p.add_argument("--skip-parse", action="store_true", help="跳过格式解析")
    build_p.add_argument("--batch", action="store_true", help="批量构建（输入为目录）")
    build_p.add_argument("--type", choices=["book_kb", "case_kb", "qa_kb", "agentic", "direct"], help="显式指定流水线")
    build_p.add_argument("--book-type", default="", help="书籍类型，如 textbook 或 monograph")
    build_p.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    factory = KBFactory()

    if args.command == "route":
        decision = factory.route(args.input)
        print(json.dumps({
            "target_skill": decision.target_skill,
            "book_type": decision.book_type,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "suggested_params": decision.suggested_params,
        }, ensure_ascii=False, indent=2))

    elif args.command == "build":
        # --json 是下游自动化接口：将构建诊断转到 stderr，保证 stdout
        # 只包含最终 JSON。
        output_redirect = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
        with output_redirect:
            if args.batch:
                results = factory.build_batch(args.input, args.output, skip_parse=args.skip_parse)
            else:
                result = factory.build(
                    args.input, args.output,
                    name=args.name,
                    default_category=args.category,
                    skip_parse=args.skip_parse,
                    kb_type=args.type, book_type=args.book_type,
                )
                results = [result]

        failed = any("error" in item or item.get("status") == "failed" for item in results)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"\n{'='*60}")
            print("构建失败或部分失败" if failed else "构建完成")
            print(f"{'='*60}")
            for r in results:
                kb_type = r.get("kb_type", "unknown")
                out = r.get("output_dir", r.get("cache_dir", ""))
                print(f"  [{kb_type}] {out}")
                if "qa_count" in r:
                    print(f"    问答对: {r['qa_count']}")
                if "judgment_count" in r:
                    print(f"    裁判文书: {r['judgment_count']}")

        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
