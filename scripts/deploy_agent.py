#!/usr/bin/env python3
"""
知识库智能体云端部署助手

把本地知识库智能体托管到云端平台，获取公网可访问的 API。

两条路径：
  1. 阿里云百炼 — 通过百炼 CLI (bl) 创建智能体应用
  2. Qoder Cloud Agents — 通过 Qoder API 创建云端 Agent

使用方式：
    # 交互式部署（推荐）
    python3 scripts/deploy_agent.py

    # 指定平台
    python3 scripts/deploy_agent.py --platform qoder --kb-path ~/kbs/民法典评注
    python3 scripts/deploy_agent.py --platform bailian --kb-path ~/kbs/民法典评注

    # 只检查前置条件
    python3 scripts/deploy_agent.py --check

    # 生成部署配置（不实际部署）
    python3 scripts/deploy_agent.py --generate-config --kb-path ~/kbs/民法典评注

部署流程见 references/deploy-guide.md
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# 尝试加载 YAML
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ─── 平台检测 ───

def check_bailian() -> dict:
    """检查阿里云百炼 CLI 前置条件"""
    result = {
        "platform": "bailian",
        "cli_installed": False,
        "cli_version": "",
        "auth_configured": False,
        "api_key": "",
        "issues": [],
    }

    # 检查 bl 命令
    bl_path = shutil.which("bl") or shutil.which("bailian")
    if bl_path:
        result["cli_installed"] = True
        try:
            ver = subprocess.run(
                [bl_path, "--version"],
                capture_output=True, text=True, timeout=10
            )
            result["cli_version"] = ver.stdout.strip()
        except Exception:
            pass
    else:
        result["issues"].append(
            "百炼 CLI 未安装。安装命令：npm install -g bailian-cli"
        )
        return result

    # 检查认证状态
    try:
        auth = subprocess.run(
            [bl_path, "auth", "status", "--output", "json"],
            capture_output=True, text=True, timeout=10
        )
        if auth.returncode == 0:
            auth_data = json.loads(auth.stdout)
            if auth_data.get("configured") or auth_data.get("api_key"):
                result["auth_configured"] = True
                result["api_key"] = auth_data.get("api_key_masked", "****")
    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if not result["auth_configured"]:
        result["issues"].append(
            "百炼 CLI 未认证。运行：bl auth login --console（浏览器登录）\n"
            "  或：bl auth login --api-key YOUR_API_KEY\n"
            "  API Key 获取：https://bailian.console.aliyun.com/cn-beijing/?tab=app#/api-key"
        )

    return result


def check_qoder() -> dict:
    """检查 Qoder Cloud Agents 前置条件"""
    result = {
        "platform": "qoder",
        "pat_configured": False,
        "api_base": "https://api.qoder.com/api/v1/cloud",
        "issues": [],
    }

    pat = os.environ.get("QODER_PAT", "")
    if pat:
        result["pat_configured"] = True
    else:
        result["issues"].append(
            "QODER_PAT 环境变量未设置。\n"
            "  获取令牌：https://qoder.com → 设置 → 个人访问令牌 → 创建令牌\n"
            "  设置环境变量：export QODER_PAT=\"your-personal-access-token\""
        )

    return result


# ─── Qoder 部署 ───

def deploy_qoder(kb_path: str, config: dict, dry_run: bool = False) -> dict:
    """
    部署到 Qoder Cloud Agents。

    流程：
      1. 验证 PAT
      2. 获取/创建 Environment
      3. 上传知识库文件到 Files
      4. 创建 Agent（含 system prompt + tools）
      5. 输出公网 API endpoint

    Returns:
        部署结果（含 agent_id, api_endpoint 等）
    """
    import urllib.request

    pat = os.environ.get("QODER_PAT", "")
    if not pat:
        return {"error": "QODER_PAT 未设置"}

    api_base = config.get("api_base", "https://api.qoder.com/api/v1/cloud")
    headers = {
        "Authorization": f"Bearer {pat}",
        "Content-Type": "application/json",
    }

    result = {"platform": "qoder", "steps": []}

    def api_call(method, path, data=None):
        url = f"{api_base}{path}"
        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            return {"error": f"HTTP {e.code}", "detail": error_body}
        except Exception as e:
            return {"error": str(e)}

    # Step 1: 获取或创建 Environment
    print("\n[Step 1/4] 获取运行环境...")
    if dry_run:
        print("  [dry-run] 跳过实际 API 调用")
        result["steps"].append({"step": "environment", "status": "dry_run"})
    else:
        env_resp = api_call("GET", "/environments")
        if "error" in env_resp:
            return {"error": f"获取环境失败: {env_resp}"}

        envs = env_resp.get("data", [])
        if envs:
            env_id = envs[0]["id"]
            print(f"  使用现有环境: {env_id}")
        else:
            print("  创建默认环境...")
            env_resp = api_call("POST", "/environments", {
                "name": "legal-kb-env",
                "config": {"type": "cloud", "networking": {"type": "unrestricted"}},
            })
            if "error" in env_resp:
                return {"error": f"创建环境失败: {env_resp}"}
            env_id = env_resp["id"]
            print(f"  新建环境: {env_id}")

        result["environment_id"] = env_id
        result["steps"].append({"step": "environment", "status": "ok", "id": env_id})

    # Step 2: 创建 Agent
    print("\n[Step 2/4] 创建知识库智能体...")
    agent_config = config.get("agent", {})
    agent_name = agent_config.get("name", "legal-kb-agent")
    agent_model = agent_config.get("model", "ultimate")
    agent_system = agent_config.get("system", "你是一个法律知识库咨询智能体。")
    tools_config = agent_config.get("tools", {})

    if dry_run:
        print(f"  [dry-run] 将创建 Agent: {agent_name}")
        result["steps"].append({"step": "agent", "status": "dry_run"})
    else:
        agent_payload = {
            "name": agent_name,
            "model": agent_model,
            "system": agent_system,
            "tools": [tools_config] if tools_config else [],
        }
        agent_resp = api_call("POST", "/agents", agent_payload)
        if "error" in agent_resp:
            return {"error": f"创建 Agent 失败: {agent_resp}"}

        agent_id = agent_resp["id"]
        print(f"  Agent ID: {agent_id}")
        result["agent_id"] = agent_id
        result["steps"].append({"step": "agent", "status": "ok", "id": agent_id})

    # Step 3: 提示上传知识库文件
    print("\n[Step 3/4] 知识库文件上传...")
    kb_upload = agent_config.get("kb_upload", {})
    files_config = kb_upload.get("files", [])

    if files_config:
        print("  需要上传的文件:")
        for fc in files_config:
            p = fc.get("path", "")
            pattern = fc.get("pattern", "*")
            full_path = Path(p) / pattern if p else Path(pattern)
            print(f"    {full_path}")

    if dry_run:
        print("  [dry-run] 跳过文件上传")
    else:
        # 上传文件（通过 /files API）
        env_id = result.get("environment_id", "")
        agent_id = result.get("agent_id", "")
        if env_id and agent_id:
            print("  上传知识库文件到 Qoder Files...")
            uploaded = _upload_files_to_qoder(
                api_base, pat, kb_path, files_config, env_id
            )
            result["uploaded_files"] = uploaded
            print(f"  已上传 {len(uploaded)} 个文件")

    result["steps"].append({"step": "files", "status": "ok" if not dry_run else "dry_run"})

    # Step 4: 输出 API endpoint
    print("\n[Step 4/4] 生成公网 API endpoint...")

    if not dry_run and "agent_id" in result:
        agent_id = result["agent_id"]
        env_id = result.get("environment_id", "")

        # 公网 API 调用方式
        result["api_endpoint"] = f"{api_base}/agents/{agent_id}"
        result["session_endpoint"] = f"{api_base}/sessions"
        result["events_endpoint"] = f"{api_base}/sessions/{{session_id}}/events"
        result["stream_endpoint"] = f"{api_base}/sessions/{{session_id}}/events/stream"

        print(f"\n  ✅ 部署成功！公网 API 信息：")
        print(f"  Agent ID: {agent_id}")
        print(f"  Environment ID: {env_id}")
        print(f"  API Base: {api_base}")
        print(f"\n  调用示例：")
        print(f"  curl -s -X POST {api_base}/sessions \\")
        print(f'    -H "Authorization: Bearer $QODER_PAT" \\')
        print(f'    -H "Content-Type: application/json" \\')
        print(f'    -d \'{{"agent": "{agent_id}", "environment_id": "{env_id}"}}\'')

    result["deployed_at"] = datetime.now().isoformat()
    return result


def _upload_files_to_qoder(
    api_base: str, pat: str, kb_path: str, files_config: list, env_id: str
) -> list:
    """上传文件到 Qoder Files API"""
    import urllib.request

    uploaded = []
    headers = {"Authorization": f"Bearer {pat}"}

    for fc in files_config:
        base_path = Path(fc.get("path", "").replace("kb_dir", kb_path))
        pattern = fc.get("pattern", "*")

        if not base_path.exists():
            print(f"  ⚠️ 路径不存在: {base_path}")
            continue

        for file_path in base_path.glob(pattern):
            if not file_path.is_file():
                continue
            try:
                # 读取文件内容
                content = file_path.read_bytes()

                # 构造 multipart 请求
                boundary = "----LegalKBBoundary"
                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                    f"Content-Type: application/octet-stream\r\n\r\n"
                ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

                req_headers = {
                    **headers,
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                }

                req = urllib.request.Request(
                    f"{api_base}/files",
                    data=body,
                    headers=req_headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    uploaded.append({
                        "file": str(file_path),
                        "file_id": resp_data.get("id", ""),
                    })
            except Exception as e:
                print(f"  ⚠️ 上传失败 {file_path.name}: {e}")

    return uploaded


# ─── 百炼部署 ───

def deploy_bailian(kb_path: str, config: dict, dry_run: bool = False) -> dict:
    """
    部署到阿里云百炼。

    流程：
      1. 验证 bl CLI 和认证
      2. 根据 kb_mode 选择部署模式
         - upload: 上传文档到百炼知识库
         - mcp:    配置 MCP 指向本地服务
         - api:    配置 HTTP API 指向已部署的 api_server
      3. 创建智能体应用
      4. 输出公网 API endpoint
    """
    bl_path = shutil.which("bl") or shutil.which("bailian")
    if not bl_path:
        return {"error": "百炼 CLI 未安装。运行：npm install -g bailian-cli"}

    result = {"platform": "bailian", "steps": []}
    app_config = config.get("app", {})
    kb_mode = app_config.get("kb_mode", "mcp")

    print(f"\n部署模式: {kb_mode}")

    # Step 1: 验证认证
    print("\n[Step 1/3] 验证百炼 CLI 认证...")
    if dry_run:
        print("  [dry-run] 跳过认证检查")
    else:
        auth = subprocess.run(
            [bl_path, "auth", "status", "--output", "json"],
            capture_output=True, text=True, timeout=10
        )
        if auth.returncode != 0:
            return {"error": "百炼 CLI 未认证。运行：bl auth login --console"}
        print("  ✓ 认证正常")
    result["steps"].append({"step": "auth", "status": "ok"})

    # Step 2: 按模式部署
    print(f"\n[Step 2/3] 配置知识库接入（{kb_mode} 模式）...")

    if kb_mode == "upload":
        # 上传文档到百炼知识库
        if dry_run:
            print("  [dry-run] 将上传以下文件到百炼知识库:")
            for pattern in app_config.get("upload", {}).get("kb_files", []):
                print(f"    {pattern}")
        else:
            print("  上传知识库文件到百炼...")
            kb_files = app_config.get("upload", {}).get("kb_files", [])
            uploaded = []
            for pattern in kb_files:
                full_pattern = pattern.replace("kb_dir", kb_path)
                for fp in Path("/").glob(full_pattern.lstrip("/")) if full_pattern.startswith("/") else Path(".").glob(full_pattern):
                    if fp.is_file():
                        try:
                            subprocess.run(
                                [bl_path, "kb", "upload", "--file", str(fp)],
                                capture_output=True, text=True, timeout=120
                            )
                            uploaded.append(str(fp))
                        except Exception as e:
                            print(f"  ⚠️ 上传失败 {fp.name}: {e}")
            result["uploaded_files"] = uploaded
            print(f"  ✓ 已上传 {len(uploaded)} 个文件")

    elif kb_mode == "mcp":
        mcp_endpoint = app_config.get("mcp", {}).get("endpoint", "")
        if not mcp_endpoint:
            print("  ⚠️ MCP 模式需要先在公网服务器部署 mcp_server.py")
            print("  部署方法见 references/deploy-guide.md")
            print("  配置好公网地址后填入 deploy-backends.yaml 的 bailian.app.mcp.endpoint")
        else:
            print(f"  MCP endpoint: {mcp_endpoint}")

    elif kb_mode == "api":
        api_endpoint = app_config.get("api", {}).get("endpoint", "")
        if not api_endpoint:
            print("  ⚠️ API 模式需要先在公网服务器部署 api_server.py")
        else:
            print(f"  API endpoint: {api_endpoint}")

    result["steps"].append({"step": "kb_config", "status": "ok", "mode": kb_mode})

    # Step 3: 创建智能体应用
    print("\n[Step 3/3] 创建百炼智能体应用...")
    app_name = app_config.get("name", "legal-kb-agent")
    app_desc = app_config.get("description", "法律知识库咨询智能体")
    model = app_config.get("model", "qwen-plus")

    if dry_run:
        print(f"  [dry-run] 将创建应用: {app_name} (model={model})")
    else:
        # 用 bl CLI 创建应用（具体命令取决于百炼 CLI 版本）
        print(f"  应用名: {app_name}")
        print(f"  模型: {model}")
        print(f"  描述: {app_desc}")
        print(f"  知识库模式: {kb_mode}")

        # 提示用户在百炼控制台完成应用创建
        print(f"\n  请在百炼控制台完成应用创建和发布：")
        print(f"  https://bailian.console.aliyun.com/cn-beijing/?tab=app")
        print(f"  创建应用后，在「应用调用」页面获取 App ID 和 API Key")

    result["console_url"] = "https://bailian.console.aliyun.com/cn-beijing/?tab=app"
    result["api_docs"] = "https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api"
    result["deployed_at"] = datetime.now().isoformat()

    print(f"\n  ✅ 百炼部署引导完成！")
    print(f"  控制台: {result['console_url']}")
    print(f"  API 文档: {result['api_docs']}")
    print(f"  发布后可通过百炼 API 调用，公网可访问。")

    return result


# ─── 配置管理 ───

def load_config(config_path: str = None) -> dict:
    """加载部署配置"""
    if not HAS_YAML:
        print("⚠️ PyYAML 未安装，使用默认配置")
        return _default_config()

    path = Path(config_path) if config_path else _PROJECT_ROOT / "assets" / "deploy-backends.yaml"
    if not path.exists():
        # 从 example 复制
        example = _PROJECT_ROOT / "assets" / "deploy-backends.example.yaml"
        if example.exists():
            print(f"配置文件不存在，从模板创建: {path}")
            shutil.copy2(example, path)
        else:
            return _default_config()

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def _default_config() -> dict:
    """默认配置"""
    return {
        "platforms": {
            "bailian": {
                "enabled": False,
                "api_key": "",
                "region": "cn-beijing",
                "cli_command": "bl",
                "app": {
                    "name": "legal-kb-agent",
                    "model": "qwen-plus",
                    "kb_mode": "mcp",
                },
            },
            "qoder": {
                "enabled": False,
                "api_base": "https://api.qoder.com/api/v1/cloud",
                "agent": {
                    "name": "legal-kb-agent",
                    "model": "ultimate",
                },
            },
        }
    }


def save_deploy_result(result: dict, output_dir: str = None):
    """保存部署结果"""
    out = Path(output_dir) if output_dir else _PROJECT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "deploy_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n部署结果已保存: {result_path}")


# ─── 交互式部署 ───

def interactive_deploy(kb_path: str = None):
    """交互式部署流程"""
    print("=" * 60)
    print("legal-kb-builder 云端部署助手")
    print("=" * 60)

    # 选择平台
    print("\n选择部署平台：")
    print("  1. Qoder Cloud Agents — 全托管，Agent 在云端 sandbox 运行")
    print("     公网 API：https://api.qoder.com/api/v1/cloud")
    print("     特点：上传 KB 文件 + 创建 Agent 即可获得公网 API")
    print()
    print("  2. 阿里云百炼 — 通过百炼 CLI 创建智能体应用")
    print("     公网 API：百炼应用发布后可通过 API 调用")
    print("     特点：支持知识库上传 / MCP / HTTP API 三种接入模式")
    print()

    while True:
        try:
            choice = input("请选择 (1/2，0 退出): ").strip()
            if choice == "0":
                return
            if choice == "1":
                platform = "qoder"
                break
            elif choice == "2":
                platform = "bailian"
                break
            print("无效选择")
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return

    # 输入知识库路径
    if not kb_path:
        kb_path = input("\n请输入知识库目录路径: ").strip()
        if not kb_path or not Path(kb_path).exists():
            print(f"路径不存在: {kb_path}")
            return

    # 检查前置条件
    print(f"\n--- 检查 {platform} 前置条件 ---")
    if platform == "qoder":
        check = check_qoder()
    else:
        check = check_bailian()

    if check.get("issues"):
        print("\n⚠️ 以下问题需要解决：")
        for issue in check["issues"]:
            print(f"  • {issue}")
        print("\n请解决以上问题后重新运行。")
        return

    print("  ✓ 前置条件检查通过")

    # 加载配置
    config = load_config()
    platform_config = config.get("platforms", {}).get(platform, {})

    # 确认部署
    print(f"\n即将部署到 {platform}，知识库路径: {kb_path}")
    confirm = input("确认部署？(y/n): ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    # 执行部署
    if platform == "qoder":
        result = deploy_qoder(kb_path, platform_config)
    else:
        result = deploy_bailian(kb_path, platform_config)

    if "error" in result:
        print(f"\n❌ 部署失败: {result['error']}")
    else:
        print("\n✅ 部署完成！")
        save_deploy_result(result)


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="知识库智能体云端部署助手"
    )
    parser.add_argument("--platform", "-p", choices=["qoder", "bailian"], help="部署平台")
    parser.add_argument("--kb-path", "-k", help="知识库目录路径")
    parser.add_argument("--check", "-c", action="store_true", help="检查前置条件")
    parser.add_argument("--generate-config", "-g", action="store_true", help="生成部署配置")
    parser.add_argument("--dry-run", action="store_true", help="只演示不实际部署")
    args = parser.parse_args()

    if args.check:
        print("=== 前置条件检查 ===\n")
        print("--- Qoder Cloud Agents ---")
        qoder_check = check_qoder()
        print(f"  PAT 配置: {'✓' if qoder_check['pat_configured'] else '✗'}")
        if qoder_check.get("issues"):
            for issue in qoder_check["issues"]:
                print(f"  • {issue}")

        print("\n--- 阿里云百炼 ---")
        bailian_check = check_bailian()
        print(f"  CLI 安装: {'✓' if bailian_check['cli_installed'] else '✗'}")
        print(f"  CLI 版本: {bailian_check.get('cli_version', 'N/A')}")
        print(f"  认证状态: {'✓' if bailian_check['auth_configured'] else '✗'}")
        if bailian_check.get("issues"):
            for issue in bailian_check["issues"]:
                print(f"  • {issue}")
        return

    if args.generate_config:
        config_path = _PROJECT_ROOT / "assets" / "deploy-backends.yaml"
        if config_path.exists():
            print(f"配置文件已存在: {config_path}")
            return
        example = _PROJECT_ROOT / "assets" / "deploy-backends.example.yaml"
        if example.exists():
            shutil.copy2(example, config_path)
            print(f"已从模板创建配置文件: {config_path}")
            print("请编辑此文件填写你的平台配置。")
        return

    if args.platform and args.kb_path:
        config = load_config()
        platform_config = config.get("platforms", {}).get(args.platform, {})

        if args.platform == "qoder":
            result = deploy_qoder(args.kb_path, platform_config, dry_run=args.dry_run)
        else:
            result = deploy_bailian(args.kb_path, platform_config, dry_run=args.dry_run)

        if "error" in result:
            print(f"\n❌ 部署失败: {result['error']}")
            sys.exit(1)
        else:
            save_deploy_result(result)
    else:
        interactive_deploy(args.kb_path)


if __name__ == "__main__":
    main()
