# ============================================================
#  config.py —— 公共配置
#
#  ★ 这个文件里【不含任何密钥】，可以安全地提交到 git、打进 Docker 镜像。
#    （之前它被 gitignore 排除，导致服务器上根本没有这个文件，
#      容器一启动就报 "No module named 'config'" —— 这就是那次部署失败的原因）
#
#  密钥从哪来？按优先级：
#    ① 环境变量（Docker 里由 docker-compose.yml 的 environment 注入）
#    ② 同目录的 .env 文件（本地开发和服务器手动部署用；.env 被 gitignore 排除）
#    ③ 都没有 → 值为空，并在终端给出明确提示
#
#  ★ 这里手写了一个极简的 .env 解析，为的是不依赖 python-dotenv（少一个依赖）
# ============================================================

import os
from pathlib import Path


def _read_env_file(path):
    """极简 .env 解析：一行一个 KEY=VALUE，支持 # 注释和空行。"""
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


_ENV_FILE = _read_env_file(Path(__file__).parent / ".env")


def _get(key, default=""):
    """取值顺序：环境变量 → .env 文件 → 默认值"""
    return os.environ.get(key) or _ENV_FILE.get(key) or default


# 大模型密钥
#   DeepSeek 官方：https://platform.deepseek.com → API Keys
#   本地：写在 .env 里；Docker：写在 docker-compose.yml 的 environment 里
API_KEY = _get("API_KEY")

# 模型和接口地址
#   deepseek-v4-flash-vision-exp  ← 支持图片（多模态），文字也照样能用
#   deepseek-v4-flash            ← 纯文字，更快更便宜
#   deepseek-v4-pro              ← 纯文字，能力更强
MODEL_NAME = _get("MODEL_NAME", "deepseek-v4-flash-vision-exp")
BASE_URL = _get("BASE_URL", "https://api.deepseek.com")

# ⚠️ 注意：上面这几个模型都是"推理模型"（会先思考再回答）
#    所以调用时 max_tokens 要给够（建议 2000 以上），
#    否则思考过程会把 token 吃光，导致回答是空的。

# 博查搜索的密钥（联网搜索用）
# 去 https://open.bochaai.com 微信扫码登录 → API Key 管理 → 创建密钥
# 没填之前，搜索工具会提示"还没配置"，其他功能不受影响
BOCHA_API_KEY = _get("BOCHA_API_KEY", "")

# 启动自检：密钥没读到就给一句人话提示（不抛异常，让服务还能起来）
if not API_KEY:
    print("[提示] 没读到 API_KEY —— 本地请检查 .env 文件；Docker 请检查 docker-compose.yml 的 environment")
