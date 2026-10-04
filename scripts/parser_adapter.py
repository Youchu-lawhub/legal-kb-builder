#!/usr/bin/env python3
"""
解析后端适配器（Parser Adapter）

作用：把"要解析的原始文件"→"Markdown 文本 / .md 文件"这一步抽象为可插拔的后端调用。
后端在 assets/parser-backends.yaml 中声明（enabled / api_key / endpoint / cmd），
用户自行填写；本适配器只负责按优先级顺序调用。

内置后端（按典型 QPS/离线可用性排序，用户可全部关掉只保留自己想用的）：

  ID                  说明                                           所需配置
  ------------------- ---------------------------------------------- ---------------------------------
  mineru              MinerU 官方 API（可精度/闪速模式）                       api_token, endpoint
  aliyun-bailian      阿里云百炼平台文档理解                                    api_key, endpoint, model
  hehe-textin         合合信息 TextIn 通用文档解析                             app_id, app_secret, endpoint
  paddleocr           PaddleOCR 本地/自建服务                                cmd 或 endpoint
  local-libreoffice   本地 LibreOffice/pandoc（doc/docx/ppt/pptx 转 md）      cmd
  local-html2md       本地 html→md（markdownify/pandoc）                    cmd
  doc-parse-fallback  QoderWork 环境内置的 doc-parse（如可用）                 cmd 指向 doc_parser.py
  raw-pass-through    仅当输入是文本时把内容直读出来当 md                       enabled: true

命令行：
    python parser_adapter.py <input> [-o outdir] [--backends id1,id2] [--dry-run]

程序化调用：
    from parser_adapter import ParserAdapter
    adapter = ParserAdapter()
    md_path = adapter.parse_to_markdown("/path/to/scan.pdf", outdir="/tmp")
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _SCRIPT_DIR.parent / "assets"
_DEFAULT_CONFIG = _ASSETS_DIR / "parser-backends.yaml"


@dataclass
class ParseResult:
    ok: bool
    backend: str
    output_path: Optional[Path] = None
    stderr: str = ""
    elapsed_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


class ParserAdapter:
    """
    按用户配置的后端顺序尝试将非 md 文件转为 Markdown。
    任意一个后端成功即返回；全部失败时抛出 RuntimeError。

    支持大文件自动分片：若输入 PDF 页数超过后端的 max_pages 限制，
    自动调用 split_pdf.py 拆分后逐片解析，最后合并。
    """

    # 各后端的默认页数限制（可被 parser-backends.yaml 中的 max_pages 字段覆盖）
    # MinerU 2026-07 更新：单文件上限从 600 降至 200 页
    DEFAULT_BACKEND_LIMITS = {
        "mineru": 200,
        "aliyun-bailian": 200,
        "hehe-textin": 200,
        "paddleocr": 500,       # PaddleOCR 本地部署，限制较宽松
        "local-libreoffice": 10000,  # 本地工具，无硬限制
        "local-html2md": 10000,
    }

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"未找到后端配置: {cfg_path}\n"
                f"请先复制 assets/parser-backends.example.yaml 到 parser-backends.yaml 并填写。"
            )
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}
        self.backends: List[Dict[str, Any]] = self.config.get("backends", [])
        self.default_order: List[str] = self.config.get("default_order", [])

    # ---------- 公开接口 ----------

    def parse_to_markdown(
        self,
        input_path: str,
        outdir: Optional[str] = None,
        prefer: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> ParseResult:
        src = Path(input_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)

        outdir_path = Path(outdir).expanduser().resolve() if outdir else src.parent
        outdir_path.mkdir(parents=True, exist_ok=True)

        # ── 大文件检测：PDF 页数超过后端限制时自动分片 ──
        if src.suffix.lower() == ".pdf":
            page_count = self._count_pdf_pages(src)
            max_pages = self._get_backend_max_pages(prefer)
            if page_count and page_count > max_pages:
                print(f"[ParserAdapter] PDF {page_count} 页超过单次解析上限 {max_pages} 页，自动分片...")
                return self._parse_large_pdf(src, outdir_path, max_pages, prefer, dry_run)

        order = self._resolve_order(prefer)
        errors: List[str] = []
        for backend_id in order:
            spec = self._find_backend(backend_id)
            if not spec:
                errors.append(f"[skip] 未在配置中定义: {backend_id}")
                continue
            if not spec.get("enabled", False):
                errors.append(f"[skip] 未启用: {backend_id}")
                continue

            handler = _BACKEND_HANDLERS.get(spec.get("type") or backend_id)
            if handler is None:
                errors.append(f"[skip] 未知后端 type: {spec}")
                continue

            if dry_run:
                print(f"[dry-run] would call backend={backend_id} on {src}")
                return ParseResult(ok=True, backend=backend_id,
                                   output_path=None, meta={"dry_run": True})

            t0 = time.time()
            try:
                result = handler(spec, src, outdir_path)
                result.elapsed_ms = int((time.time() - t0) * 1000)
                if result.ok:
                    return result
                errors.append(f"[fail] {backend_id}: {result.stderr[:200]}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"[error] {backend_id}: {exc}")

        raise RuntimeError(
            "所有已启用后端都未能解析该文件。诊断信息：\n  " + "\n  ".join(errors)
        )

    # ---------- 内部工具 ----------

    def _resolve_order(self, prefer: Optional[List[str]]) -> List[str]:
        if prefer:
            return list(prefer)
        if self.default_order:
            return list(self.default_order)
        return [b.get("id") for b in self.backends if b.get("id")]

    def _find_backend(self, backend_id: str) -> Optional[Dict[str, Any]]:
        for b in self.backends:
            if b.get("id") == backend_id:
                return b
        return None

    # ---------- 大文件分片策略 ----------

    @staticmethod
    def _count_pdf_pages(pdf_path: Path) -> int:
        """获取 PDF 页数（用 pypdf，不可用时返回 0）"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            return len(reader.pages)
        except Exception:
            return 0

    def _get_backend_max_pages(self, prefer: Optional[List[str]] = None) -> int:
        """
        获取将要使用的后端的页数限制。

        优先级：
          1. parser-backends.yaml 中后端声明的 max_pages 字段
          2. DEFAULT_BACKEND_LIMITS 中的默认值
          3. 兜底 200（最保守）
        """
        order = self._resolve_order(prefer)
        for backend_id in order:
            spec = self._find_backend(backend_id)
            if not spec or not spec.get("enabled", False):
                continue
            # 配置文件中的 max_pages 优先
            if spec.get("max_pages"):
                return int(spec["max_pages"])
            # 默认限制表
            return self.DEFAULT_BACKEND_LIMITS.get(backend_id, 200)
        return 200  # 最保守兜底

    def _parse_large_pdf(
        self,
        src: Path,
        outdir: Path,
        max_pages: int,
        prefer: Optional[List[str]],
        dry_run: bool,
    ) -> ParseResult:
        """
        大 PDF 自动分片解析策略。

        流程：
          1. 调用 split_pdf.py 按 max_pages 拆分（带 3 页重叠）
          2. 逐片调用 parse_to_markdown（递归，但已分片不会再次触发分片）
          3. 合并所有片的 md 输出为一个文件
          4. 记录分片信息到 meta

        降级策略：
          - 首次分片用 max_pages（如 200）
          - 若某片仍失败（可能限制进一步收紧），自动缩小到 max_pages//2 重试
          - 缩小 2 次后仍失败则跳过该片并记录错误
        """
        import subprocess as sp

        chunk_size = max_pages
        overlap = 3
        max_retries = 2  # 最多缩小 2 次

        for attempt in range(max_retries + 1):
            if dry_run:
                print(f"[dry-run] 将按 {chunk_size} 页/片拆分 {src.name} 并逐片解析")
                return ParseResult(ok=True, backend="auto-split",
                                   output_path=None, meta={"dry_run": True, "chunk_size": chunk_size})

            # Step 1: 拆分
            split_dir = outdir / f"_split_{src.stem}"
            split_dir.mkdir(parents=True, exist_ok=True)

            print(f"[分片] 拆分 {src.name} → {split_dir}/ (每片 {chunk_size} 页，重叠 {overlap} 页)")
            split_result = sp.run(
                [sys.executable, str(_SCRIPT_DIR / "split_pdf.py"),
                 str(src), "-o", str(split_dir),
                 "--mode", "auto",
                 "--chunk-size", str(chunk_size),
                 "--overlap", str(overlap),
                 "--max-pages", str(chunk_size)],
                capture_output=True, text=True, timeout=60,
            )
            if split_result.returncode != 0:
                print(f"[分片] 拆分失败: {split_result.stderr[:300]}")
                if attempt < max_retries:
                    chunk_size = max(50, chunk_size // 2)
                    print(f"[降级] 缩小分片到 {chunk_size} 页重试...")
                    continue
                return ParseResult(ok=False, backend="auto-split",
                                   stderr=f"PDF 拆分失败: {split_result.stderr[:300]}")

            # Step 2: 逐片解析
            split_pdfs = sorted(split_dir.glob("*.pdf"))
            if not split_pdfs:
                return ParseResult(ok=False, backend="auto-split",
                                   stderr="拆分后未找到 PDF 文件")

            print(f"[分片] 共 {len(split_pdfs)} 片，逐片解析...")
            md_parts = []
            errors = []
            for i, pdf_chunk in enumerate(split_pdfs, 1):
                print(f"  [{i}/{len(split_pdfs)}] {pdf_chunk.name}...")
                try:
                    result = self.parse_to_markdown(str(pdf_chunk), str(outdir))
                    if result.ok and result.output_path:
                        md_parts.append(result.output_path.read_text(encoding="utf-8"))
                    else:
                        errors.append(f"{pdf_chunk.name}: {result.stderr[:100]}")
                except Exception as e:
                    errors.append(f"{pdf_chunk.name}: {e}")

            if not md_parts:
                if attempt < max_retries:
                    chunk_size = max(50, chunk_size // 2)
                    print(f"[降级] 所有分片解析失败，缩小到 {chunk_size} 页重试...")
                    continue
                return ParseResult(ok=False, backend="auto-split",
                                   stderr=f"所有分片解析失败: {'; '.join(errors[:3])}")

            # Step 3: 合并
            output_md = outdir / (src.stem + ".md")
            combined = "\n\n---\n\n".join(md_parts)
            output_md.write_text(combined, encoding="utf-8")

            print(f"[分片] 合并完成: {output_md.name} ({len(md_parts)}/{len(split_pdfs)} 片成功)")
            if errors:
                print(f"[分片] {len(errors)} 片失败（已跳过）")

            # 清理临时分片目录
            try:
                import shutil
                shutil.rmtree(split_dir)
            except Exception:
                pass

            return ParseResult(
                ok=True, backend="auto-split",
                output_path=output_md,
                meta={
                    "total_chunks": len(split_pdfs),
                    "successful_chunks": len(md_parts),
                    "failed_chunks": len(errors),
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "errors": errors[:5],
                },
            )

        return ParseResult(ok=False, backend="auto-split",
                           stderr=f"分片解析在 {max_retries + 1} 次降级后仍失败")


# ---------- 后端处理函数 ----------
# 每个 handler(spec, src, outdir) -> ParseResult。签名统一，方便扩展。

import signal


class _SafeCmdDict(dict):
    """给 str.format_map 用的容错字典：未知占位符原样保留，避免 KeyError；
    这样命令模板里 shell 语法自然的 ${var}、awk '{print $1}' 等大括号内容不会崩。"""

    def __missing__(self, key: str) -> str:  # noqa: D401
        return "{" + key + "}"


_ERROR_MARKERS = (
    '"error"', "'error'", '"code":', "'code':",
    "error_code", "traceback", "ERROR:", "Exception:",
)


def _looks_like_error_payload(text: str) -> bool:
    """粗判 backend 返回的内容是否是错误 JSON/日志，而不是真正的 markdown。"""
    if not text:
        return True
    head = text.lstrip()[:400].lower()
    # 明确的错误 JSON
    if head.startswith("{") and any(m.lower() in head for m in ('"error"', "'error'", "error_code", '"code":')):
        return True
    # HTML 错误页
    if "<html" in head and ("error" in head or "not found" in head or "403" in head or "401" in head):
        return True
    return False


def _validate_markdown_output(path: Path, min_bytes: int = 20) -> Optional[str]:
    """返回错误消息；返回 None 表示通过。"""
    if not path.exists():
        return "输出文件未生成"
    if path.stat().st_size < min_bytes:
        return f"输出文件过小（{path.stat().st_size} 字节），疑似空/失败响应"
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception as e:
        return f"无法读取输出文件: {e}"
    if _looks_like_error_payload(head):
        return f"输出内容疑似错误响应: {head[:200]}"
    return None


def _handler_command(spec: Dict[str, Any], src: Path, outdir: Path) -> ParseResult:
    """
    通用"命令行后端"：spec 中 cmd 是模板，支持 {input} {outdir} {output_md} 占位符。

    v2 修复要点：
    - format_map + SafeDict：命令模板里的 awk '{...}'、${var} 等 shell 大括号不会被误解析。
    - 输出校验：不再"文件存在且非空"就当成功，进一步检查文件是否是 JSON 错误响应。
    - POSIX 下用 os.setsid + os.killpg 保证超时时能把整个子进程组杀掉。
    """
    cmd_tpl = spec.get("cmd")
    if not cmd_tpl:
        return ParseResult(ok=False, backend=spec.get("id", "command"),
                           stderr="cmd 字段为空")
    output_md = outdir / (src.stem + ".md")

    placeholders = _SafeCmdDict(
        input=shlex.quote(str(src)),
        outdir=shlex.quote(str(outdir)),
        output_md=shlex.quote(str(output_md)),
    )
    try:
        rendered = cmd_tpl.format_map(placeholders)
    except (IndexError, ValueError) as e:
        return ParseResult(ok=False, backend=spec.get("id", "command"),
                           stderr=f"cmd 模板渲染失败: {e}. 提示：shell 中的 {{...}} 若非占位符请用 {{{{...}}}} 转义")

    timeout = spec.get("timeout", 600)
    popen_kwargs = dict(
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if os.name == "posix":
        popen_kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(rendered, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # 杀掉整个进程组，避免 shell 死掉但真正的子进程还在跑
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        proc.communicate()
        return ParseResult(ok=False, backend=spec.get("id", "command"),
                           stderr=f"timeout after {timeout}s")
    except UnicodeDecodeError as e:
        proc.kill()
        return ParseResult(ok=False, backend=spec.get("id", "command"),
                           stderr=f"输出编码解析失败: {e}")

    # 优先：命令直接写了 output_md 文件
    if output_md.exists():
        err = _validate_markdown_output(output_md)
        if err is None:
            return ParseResult(ok=True, backend=spec.get("id", "command"),
                               output_path=output_md,
                               meta={"returncode": proc.returncode,
                                     "stderr_head": (stderr or "")[:200]})
        # 内容不合格：删除以免后续误用
        try:
            output_md.unlink()
        except Exception:
            pass
        return ParseResult(ok=False, backend=spec.get("id", "command"),
                           stderr=f"输出校验失败: {err}. stderr={stderr[:400]}")

    # 兜底：命令把 md 打到 stdout
    if proc.returncode == 0 and stdout and stdout.strip() and not _looks_like_error_payload(stdout):
        output_md.write_text(stdout, encoding="utf-8")
        return ParseResult(ok=True, backend=spec.get("id", "command"),
                           output_path=output_md,
                           meta={"note": "stdout captured as markdown"})

    return ParseResult(
        ok=False, backend=spec.get("id", "command"),
        stderr=(stderr or stdout or f"exit={proc.returncode}")[:800],
    )


def _handler_http_placeholder(spec: Dict[str, Any], src: Path, outdir: Path) -> ParseResult:
    """
    HTTP 类后端的占位实现：
    - 校验必填字段（api_key/endpoint 等）
    - 校验 requests 是否可用
    - 真正的 HTTP 调用由用户在 spec['cmd'] 里配置具体命令（例如 curl 或自写脚本），
      或者将 spec['type'] 改为 'command' 直接走 shell 调用。

    这样保持"零外部依赖 + 用户自由配置"的原则，避免把 API SDK 硬绑死在本工具中。
    """
    required = spec.get("required_env") or []
    missing = [k for k in required if not (os.environ.get(k) or spec.get(k))]
    if missing:
        return ParseResult(
            ok=False, backend=spec.get("id"),
            stderr=f"缺少必填配置: {missing}，请在 parser-backends.yaml 或环境变量中填写。",
        )
    # 若用户已配了 cmd，就复用命令后端；否则报错提示改配置。
    if spec.get("cmd"):
        return _handler_command(spec, src, outdir)
    return ParseResult(
        ok=False, backend=spec.get("id"),
        stderr=(
            f"{spec.get('id')} 后端需要用户在 parser-backends.yaml 中提供 cmd 字段"
            f"（HTTP 调用命令），或改写 type=command。"
        ),
    )


def _handler_passthrough(spec: Dict[str, Any], src: Path, outdir: Path) -> ParseResult:
    if src.suffix.lower() in {".md", ".markdown", ".txt"}:
        target = outdir / (src.stem + ".md")
        if target.resolve() != src.resolve():
            shutil.copyfile(src, target)
        return ParseResult(ok=True, backend=spec.get("id"),
                           output_path=target,
                           meta={"note": "pass-through"})
    return ParseResult(ok=False, backend=spec.get("id"),
                       stderr="raw-pass-through 仅支持 .md/.txt 输入")


_BACKEND_HANDLERS: Dict[str, Callable[[Dict[str, Any], Path, Path], ParseResult]] = {
    "command": _handler_command,
    "http": _handler_http_placeholder,
    "passthrough": _handler_passthrough,
    # 别名（type 字段未填时按 id 尝试）
    "mineru": _handler_http_placeholder,
    "aliyun-bailian": _handler_http_placeholder,
    "hehe-textin": _handler_http_placeholder,
    "paddleocr": _handler_command,
    "local-libreoffice": _handler_command,
    "local-html2md": _handler_command,
    "doc-parse-fallback": _handler_command,
    "raw-pass-through": _handler_passthrough,
}


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="Legal KB Builder — 解析后端适配器")
    ap.add_argument("input", help="待解析文件路径")
    ap.add_argument("-o", "--outdir", help="输出目录（默认与输入同目录）")
    ap.add_argument("--backends", help="逗号分隔，覆盖默认顺序，如 mineru,paddleocr")
    ap.add_argument("--config", help="自定义 parser-backends.yaml 路径")
    ap.add_argument("--dry-run", action="store_true", help="只显示将调用的后端，不执行")
    args = ap.parse_args()

    adapter = ParserAdapter(config_path=args.config)
    prefer = [x.strip() for x in args.backends.split(",")] if args.backends else None
    try:
        result = adapter.parse_to_markdown(
            args.input, outdir=args.outdir, prefer=prefer, dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[parser-adapter] 解析失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "ok": result.ok,
        "backend": result.backend,
        "output": str(result.output_path) if result.output_path else None,
        "elapsed_ms": result.elapsed_ms,
        "meta": result.meta,
    }, ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
