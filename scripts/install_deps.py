#!/usr/bin/env python3
"""
依赖安装辅助脚本

根据用户的使用场景，推荐并安装对应的依赖组合。

用法：
    python3 install_deps.py                    # 交互式选择场景
    python3 install_deps.py --scene book       # 书籍知识库场景
    python3 install_deps.py --scene all        # 全部安装
    python3 install_deps.py --list             # 列出所有场景
    python3 install_deps.py --check            # 检查已安装依赖状态
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


# ─── 场景定义 ───

SCENES = {
    "minimal": {
        "desc": "最小安装（仅必需依赖，只能做格式路由 + Grep）",
        "requirements": ["requirements.txt"],
        "size": "~10MB",
        "capabilities": [
            "格式路由（format_detector）",
            "解析后端调用（parser_adapter）",
            "数据源路由（material_router）",
            "Grep 关键词检索（无 BM25）",
        ],
        "missing": [
            "BM25 关键词检索",
            "向量语义检索",
            "Cross-Encoder 重排",
            "GraphRAG 图谱",
            "HTTP API / MCP 服务",
        ],
    },
    "book": {
        "desc": "书籍知识库（民法典评注/司法解释/学术专著）",
        "requirements": ["requirements.txt", "requirements-vector.txt"],
        "size": "~2.5GB",
        "capabilities": [
            "BM25 + 向量 + 结构索引 三路融合检索",
            "Cross-Encoder 重排（+20% 精度）",
            "父子分块（Small-to-Big）",
            "查询改写（同义词+指代消解）",
            "bge-small/large/m3 模型可选",
        ],
        "missing": [
            "GraphRAG 法条关联图谱",
            "HTTP API / MCP 对外服务",
            "RAGAS 评估",
        ],
    },
    "book-full": {
        "desc": "书籍知识库 + GraphRAG + 评估（推荐生产环境）",
        "requirements": [
            "requirements.txt",
            "requirements-vector.txt",
            "requirements-graph.txt",
            "requirements-eval.txt",
        ],
        "size": "~2.6GB",
        "capabilities": [
            "四路融合检索（BM25+向量+索引+GraphRAG）",
            "Cross-Encoder 重排",
            "父子分块 + 查询改写",
            "RAGAS 量化评估闭环",
            "法条关联图谱多跳查询",
        ],
        "missing": [
            "HTTP API / MCP 对外服务",
        ],
    },
    "case": {
        "desc": "裁判文书知识库",
        "requirements": ["requirements.txt", "requirements-vector.txt"],
        "size": "~2.5GB",
        "capabilities": [
            "裁判文书要素抽取（12字段）",
            "字段级嵌入 + BM25 检索",
            "案号/法院/案由精确检索",
        ],
        "missing": [
            "GraphRAG 图谱",
            "HTTP API / MCP",
            "RAGAS 评估",
        ],
    },
    "qa": {
        "desc": "问答集/FAQ 知识库",
        "requirements": ["requirements.txt", "requirements-vector.txt"],
        "size": "~2.5GB",
        "capabilities": [
            "六种问答格式自动解析",
            "双索引（问题索引+答案索引）",
            "问题匹配为主、答案召回为辅",
        ],
        "missing": [
            "GraphRAG 图谱",
            "HTTP API / MCP",
            "RAGAS 评估",
        ],
    },
    "agentic": {
        "desc": "Agentic 实时检索（无需向量库，靠 Grep + 蒙特卡洛采样）",
        "requirements": ["requirements.txt"],
        "size": "~10MB",
        "capabilities": [
            "FAST 模式（<5秒，单轮 Grep）",
            "DEEP 模式（三阶段蒙特卡洛采样）",
            "跨会话知识集群缓存",
        ],
        "missing": [
            "向量语义检索",
            "BM25（可选加装）",
            "HTTP API / MCP",
        ],
    },
    "api": {
        "desc": "对外提供 HTTP API 服务（前端/钉钉/飞书）",
        "requirements": [
            "requirements.txt",
            "requirements-vector.txt",
            "requirements-api.txt",
        ],
        "size": "~2.6GB",
        "capabilities": [
            "全部检索能力",
            "/ask 通用咨询接口",
            "钉钉/飞书 webhook",
            "Swagger API 文档",
        ],
        "missing": [
            "GraphRAG（可选加装）",
            "MCP 服务",
            "RAGAS 评估",
        ],
    },
    "mcp": {
        "desc": "MCP 服务（暴露给 Claude Desktop/Cursor）",
        "requirements": [
            "requirements.txt",
            "requirements-vector.txt",
            "requirements-mcp.txt",
        ],
        "size": "~2.6GB",
        "capabilities": [
            "全部检索能力",
            "consult / qa_search 等 MCP 工具",
            "Claude Desktop / Cursor 集成",
        ],
        "missing": [
            "GraphRAG（可选加装）",
            "HTTP API",
            "RAGAS 评估",
        ],
    },
    "all": {
        "desc": "全部安装（一次性装齐所有功能）",
        "requirements": ["requirements-all.txt"],
        "size": "~3GB",
        "capabilities": ["所有功能"],
        "missing": [],
    },
}


# ─── 依赖检查 ───

DEP_CHECKS = {
    "PyYAML": ("yaml", "PyYAML"),
    "pypdf": ("pypdf", "pypdf"),
    "jieba": ("jieba", "jieba"),
    "rank-bm25": ("rank_bm25", "rank-bm25"),
    "numpy": ("numpy", "numpy"),
    "sentence-transformers": ("sentence_transformers", "sentence-transformers"),
    "faiss-cpu": ("faiss", "faiss-cpu"),
    "torch": ("torch", "torch"),
    "networkx": ("networkx", "networkx"),
    "fastapi": ("fastapi", "fastapi"),
    "uvicorn": ("uvicorn", "uvicorn"),
    "pydantic": ("pydantic", "pydantic"),
    "mcp": ("mcp", "mcp"),
    "ragas": ("ragas", "ragas"),
    "datasets": ("datasets", "datasets"),
}


def check_installed() -> dict:
    """检查各依赖的安装状态"""
    status = {}
    for name, (import_name, pip_name) in DEP_CHECKS.items():
        try:
            importlib.import_module(import_name)
            status[name] = "installed"
        except (ImportError, OSError, RuntimeError):
            status[name] = "missing"
    return status


# 依赖的最低 Python 版本（不满足时 pip 安装会失败）
_PKG_PYTHON_MIN = {
    "mcp": (3, 10),
}


def _is_pkg_installable(pkg_name: str) -> bool:
    """检查当前 Python 版本是否满足该包的安装要求"""
    min_ver = _PKG_PYTHON_MIN.get(pkg_name)
    if min_ver is None:
        return True
    return sys.version_info >= min_ver


# ─── 安装逻辑 ───

def install_scene(scene: str, dry_run: bool = False) -> int:
    """安装指定场景的依赖"""
    if scene not in SCENES:
        print(f"未知场景: {scene}")
        print(f"可用场景: {', '.join(SCENES.keys())}")
        return 1

    config = SCENES[scene]
    print(f"\n场景: {scene}")
    print(f"说明: {config['desc']}")
    print(f"预计大小: {config['size']}")
    print(f"安装后能力:")
    for cap in config["capabilities"]:
        print(f"  ✓ {cap}")
    if config["missing"]:
        print(f"不具备:")
        for miss in config["missing"]:
            print(f"  ✗ {miss}")

    print(f"\n将安装以下依赖文件:")
    for req in config["requirements"]:
        req_path = _PROJECT_ROOT / req
        print(f"  - {req}")
        if not req_path.exists():
            print(f"    ⚠️ 文件不存在: {req_path}")
            return 1

    if dry_run:
        print("\n[dry-run] 未实际安装")
        return 0

    print()
    for req in config["requirements"]:
        req_path = _PROJECT_ROOT / req
        print(f"安装 {req} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"❌ 安装失败: {req}")
            return result.returncode
        print(f"✓ {req} 安装完成")

    print(f"\n✅ 场景 '{scene}' 依赖安装完成")
    return 0


def list_scenes():
    """列出所有场景"""
    print("可用安装场景:\n")
    for name, config in SCENES.items():
        print(f"  {name:15s}  {config['desc']}")
        print(f"  {'':15s}  大小: {config['size']}, 依赖文件: {', '.join(config['requirements'])}")
        print()


def interactive_select() -> str:
    """交互式选择场景"""
    print("=" * 60)
    print("legal-kb-builder 依赖安装助手")
    print("=" * 60)
    print("\n请选择你的使用场景:\n")
    scenes = list(SCENES.keys())
    for i, name in enumerate(scenes, 1):
        config = SCENES[name]
        print(f"  {i}. {name:15s}  {config['desc']}")
        print(f"     {'':15s}  大小: {config['size']}")
    print(f"\n  0. 退出")

    while True:
        try:
            choice = input(f"\n请输入编号 (0-{len(scenes)}): ").strip()
            if choice == "0":
                return ""
            idx = int(choice) - 1
            if 0 <= idx < len(scenes):
                return scenes[idx]
            print("无效编号，请重试")
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\n已取消")
            return ""


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="legal-kb-builder 依赖安装助手"
    )
    parser.add_argument(
        "--scene", "-s",
        help="直接指定安装场景",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有安装场景",
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="检查已安装依赖状态",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将安装什么，不实际安装",
    )
    args = parser.parse_args()

    if args.list:
        list_scenes()
        return

    if args.check:
        # Python 版本警告
        if sys.version_info < (3, 10):
            print(f"⚠️  当前 Python {sys.version.split()[0]}：mcp 包需要 Python >= 3.10，MCP 场景不可用。\n")

        print("依赖安装状态检查:\n")
        status = check_installed()
        installed = [k for k, v in status.items() if v == "installed"]
        missing = [k for k, v in status.items() if v == "missing"]
        if installed:
            print(f"已安装 ({len(installed)}):")
            for pkg in installed:
                print(f"  ✓ {pkg}")
        if missing:
            print(f"\n未安装 ({len(missing)}):")
            for pkg in missing:
                if not _is_pkg_installable(pkg):
                    min_v = _PKG_PYTHON_MIN[pkg]
                    print(f"  ✗ {pkg}  (需 Python >={min_v[0]}.{min_v[1]}，当前不满足)")
                else:
                    print(f"  ✗ {pkg}")
        print(f"\n总计: {len(installed)} 已安装, {len(missing)} 未安装")

        # 推荐场景：仅推荐当前 Python 版本可安装且仍有缺失依赖的场景
        installable_missing = [pkg for pkg in missing if _is_pkg_installable(pkg)]
        if installable_missing:
            recommended = []
            for name, config in SCENES.items():
                # 该场景所需的 pip 包列表（通过 requirements 文件推断）
                scene_pkgs = set()
                for req_file in config["requirements"]:
                    req_path = _PROJECT_ROOT / req_file
                    if req_path.exists():
                        for line in req_path.read_text().splitlines():
                            line = line.strip()
                            if line and not line.startswith("#") and not line.startswith("-"):
                                pkg = line.split(">=")[0].split("==")[0].split("<")[0].strip()
                                scene_pkgs.add(pkg)
                # 场景中所有包要么已装，要么可装
                scene_missing = scene_pkgs & set(missing)
                scene_blocked = scene_pkgs & set(m for m in missing if not _is_pkg_installable(m))
                if scene_missing and not scene_blocked:
                    recommended.append((name, config["desc"]))
            if recommended:
                print("\n可安装的推荐场景:")
                for name, desc in recommended:
                    print(f"  {name:15s}  {desc}")
        return

    if args.scene:
        rc = install_scene(args.scene, dry_run=args.dry_run)
        sys.exit(rc)
    else:
        scene = interactive_select()
        if scene:
            rc = install_scene(scene, dry_run=args.dry_run)
            sys.exit(rc)


if __name__ == "__main__":
    main()
