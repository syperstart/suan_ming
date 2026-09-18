# ============================================================
#  config.example.py —— 配置模板（这个文件可以上传到 git）
#
#  用法：
#    1. 把本文件复制一份，改名为 config.py
#    2. 把下面引号里的内容换成你自己的真实值
#
#  ★ config.py 已经被 .gitignore 排除，不会上传——
#    所以密钥是安全的，但换电脑/部署到服务器时要重新填一次。
# ============================================================

# 大模型的密钥
# DeepSeek 官方：https://platform.deepseek.com  →  API Keys
API_KEY = "在这里填你的大模型 API Key"

# 模型和接口地址
#   deepseek-v4-flash-vision-exp  ← 支持图片（多模态）
#   deepseek-v4-flash            ← 纯文字，更快更便宜
#   deepseek-v4-pro              ← 纯文字，能力更强
MODEL_NAME = "deepseek-v4-flash-vision-exp"
BASE_URL = "https://api.deepseek.com"

# ⚠️ 上面这几个 DeepSeek 模型都是"推理模型"（先思考再回答）
#    调用时 max_tokens 要给够（建议 2000 以上），否则回答会是空的。

# 博查搜索的密钥（联网搜索用）
# 去 https://open.bochaai.com 微信扫码登录 → API Key 管理 → 创建密钥
# 没填之前，搜索工具会提示"还没配置"，其他功能不受影响
BOCHA_API_KEY = ""
