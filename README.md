# 陈大师 · 命理测算智能体

> 一个会"算命"的 AI 智能体。人设是《鬼吹灯》里的陈玉楼（人称陈大师），
> 能聊天、能排八字、能联网查资料，还能记住你说过的话。

基于 **FastAPI + LangChain Agent + DeepSeek** 构建，配一个手机端 H5 聊天页，
自带本地知识库（RAG）和三层记忆系统，一条命令即可用 Docker 部署。

---

## 目录

- [它能做什么](#它能做什么)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
  - [方式一：本地直接跑](#方式一本地直接跑)
  - [方式二：Docker 部署（推荐）](#方式二docker-部署推荐)
- [配置说明](#配置说明)
- [接口一览](#接口一览)
- [项目结构](#项目结构)
- [核心设计](#核心设计)
- [常见问题](#常见问题)
- [安全提醒](#安全提醒)

---

## 它能做什么

| 能力 | 说明 |
| --- | --- |
| 🧙 **角色扮演** | 完整的陈玉楼人设：精通阴阳五行、紫微斗数、姓名测算、八字排盘，60 岁上下、双目失明的湘西老匪，自称"老夫""老朽"，自带口头禅（"命里有时终须有，命里无时莫强求"）。 |
| 😊 **情绪感知** | 每轮对话先判断用户情绪（`angry` / `depressed` / `cheerful` / `upbeat` / `friendly` / `default`），再据此切换回答语气——你难过时他会安慰你，你兴奋时他会提醒你别乐极生悲。 |
| 🧠 **三层记忆** | 档案（姓名、生辰等固定信息）+ 早前对话摘要 + 最近 3 轮原文。原文超过 10 条自动触发压缩，长期聊天也不会把上下文撑爆。记忆落盘 JSON，重启不丢。 |
| 👥 **多会话隔离** | 按 `session_id` 分开存记忆，不同人、不同窗口聊天互不串台。 |
| 🔮 **八字排盘** | 调用缘份居八字接口，从自然语言里自动解析姓名、性别、出生年月日时，给出四柱八字。 |
| 📚 **本地知识库（RAG）** | Qdrant 本地向量库 + 中文嵌入模型 `BAAI/bge-small-zh-v1.5`，用 MMR 检索（结果既相关又不重复）。 |
| 🌐 **联网搜索 + 自动学习** | 集成博查（Bocha）搜索，搜到的网页会自动切片存进知识库，下次直接能查到。 |
| 🔗 **发链接即学** | 你丢一个网址给它，它抓取正文、切块入库，然后基于内容回答。 |
| 🤐 **不外泄技术细节** | 工具报错时不会把密钥、状态码、报错原文甩给用户，而是用角色内的话术带过去（"今日天机未通……"）。 |

---

## 技术栈

| 层次 | 选型 |
| --- | --- |
| Web 框架 | FastAPI 0.115 + Uvicorn |
| Agent 编排 | LangChain（`create_openai_tools_agent` + `AgentExecutor`） |
| 大模型 | DeepSeek（`deepseek-v4-flash-vision-exp` 默认，兼容 OpenAI 协议） |
| 向量数据库 | Qdrant（本地文件模式，无需起服务） |
| 嵌入模型 | sentence-transformers + `BAAI/bge-small-zh-v1.5`（本地免费） |
| 联网搜索 | 博查 Bocha Web Search API |
| 前端 | 单文件 H5 页面（原生 JS，无构建步骤） |
| 部署 | Docker + Docker Compose |

---

## 快速开始

### 准备工作

去 [DeepSeek 开放平台](https://platform.deepseek.com) 申请一个 API Key（必填）。
联网搜索功能还需要一个[博查](https://open.bochaai.com)的 Key（选填，不填不影响算命/八字/知识库）。

### 方式一：本地直接跑

```bash
# 1. 安装依赖（torch 体积大，建议单独装 CPU 版）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. 配置密钥：复制模板，改名为 config.py，把 API_KEY 换成你自己的
copy config.example.py config.py

# 3. 启动
python server.py
```

也可以直接双击 **`启动服务.bat`**（会自动清理 8000 端口残留进程再启动）。

启动成功后浏览器打开：

- 电脑：<http://127.0.0.1:8000/h5>
- 手机：`http://<电脑的局域网IP>:8000/h5`（手机和电脑连同一个 WiFi）

> 首次启动需要加载中文嵌入模型，约 15–30 秒，看到 `Application startup complete` 就是好了。

### 方式二：Docker 部署（推荐）

```bash
# 1. 配置密钥
copy .env.example .env
#    编辑 .env，填上 API_KEY

# 2. 一键构建并启动
docker compose up -d --build

# 3. 看日志 / 停止
docker compose logs -f
docker compose down
```

访问地址同上：<http://127.0.0.1:8000/h5>

> **为什么用 Docker 管理的数据卷（named volume）而不是直接挂载本机目录？**
> 本项目主要在 Windows 下开发，把本地目录挂进 Linux 容器要靠文件共享转发，
> 实测会出现"程序以为写进去了、外面看不到"的情况，重启容器数据就没了。
> `named volume` 存在 Docker 自己的存储里，最稳，换台机器部署也一样。

---

## 配置说明

配置有两处入口，**环境变量优先**，不设环境变量时用默认值：

### 1. `config.py`（本地运行用）

从 `config.example.py` 复制而来，**已被 `.gitignore` 排除，不会上传**。

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `API_KEY` | DeepSeek 大模型密钥，**必填** | — |
| `MODEL_NAME` | 使用的模型 | `deepseek-v4-flash-vision-exp` |
| `BASE_URL` | 接口地址 | `https://api.deepseek.com` |
| `BOCHA_API_KEY` | 博查搜索密钥，选填 | 空 |

可选模型：

- `deepseek-v4-flash-vision-exp` —— 支持图片（多模态），文字也照样能用
- `deepseek-v4-flash` —— 纯文字，更快更便宜
- `deepseek-v4-pro` —— 纯文字，能力更强

> ⚠️ 这几个都是**推理模型**（先思考再回答），调用时 `max_tokens` 要给够（建议 2000 以上），
> 否则思考过程会把 token 吃光，回答会是空的。

### 2. `.env`（Docker 部署用）

`docker compose` 会自动读取同目录的 `.env` 来填充 `docker-compose.yml` 里的 `${变量}`。
同样已被 `.gitignore` 排除。

额外可用的环境变量：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `MEMORY_DIR` | 聊天记忆存放目录 | `server.py` 所在目录 |
| `DB_PATH` | 知识库（Qdrant）目录 | `./local_qdrant` |
| `HF_ENDPOINT` | 模型下载镜像站 | `https://hf-mirror.com` |

---

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 健康检查，返回 `{"Hello":"World"}` |
| `GET` | `/h5` | 移动端聊天页面 |
| `POST` | `/chat` | 聊天。参数 `query`（问题）、`session_id`（会话 ID），返回 `emotion` + `response` |
| `POST` | `/qingxu` | 只测情绪识别，返回 `emotion` |
| `GET` | `/memory` | 列出所有会话；带 `?session_id=xxx` 查看某个会话的记忆 |
| `POST` | `/memory/clear` | 清空指定会话记忆，默认清 `default` |
| `POST` | `/memory/summarize` | 手动触发一次记忆压缩（想马上看效果时用） |
| `POST` | `/add_urls` | 把指定 URL 的正文抓取、切片、存入知识库 |
| `GET` | `/kb/count` | 查看知识库当前条数（确认资料进没进库） |
| `POST` | `/add_pdfs` | 占位接口，待实现 |
| `POST` | `/add_texts` | 占位接口，待实现 |
| `WS` | `/ws` | WebSocket，目前是"复读机"占位 |

**调用示例：**

```bash
# 聊天
curl -X POST "http://127.0.0.1:8000/chat?query=老夫帮我算算八字&session_id=user1"

# 只测情绪
curl -X POST "http://127.0.0.1:8000/qingxu?query=气死我了"

# 看某个会话的记忆
curl "http://127.0.0.1:8000/memory?session_id=user1"

# 看知识库里存了多少段资料
curl "http://127.0.0.1:8000/kb/count"
```

---

## 项目结构

```
.
├── server.py                  # 主服务：FastAPI 应用 + Agent 主体 + 记忆系统 + 全部接口
├── Mytools.py                 # 工具集合：知识库检索 / 抓网页 / 联网搜索 / 八字排盘
├── config.py                  # 真实密钥（★ 不上传，需自己创建）
├── config.example.py          # 配置模板
├── h5.html                    # 手机端聊天页面（单文件，无构建）
├── test_kb.py                 # 知识库冒烟测试：先存资料再检索
├── requirements.txt           # 依赖清单（只列本项目真正 import 的）
├── Dockerfile                 # 镜像构建（torch 单独走 CPU 版源）
├── docker-compose.yml         # 一键启动，含数据卷与健康检查
├── .env.example               # 环境变量模板
├── .gitignore                 # 排除密钥、聊天记录、向量库、模型缓存
├── dify_tool_schema.json      # 接入 Dify 的工具 schema
├── 启动服务.bat               # Windows 一键启动脚本
└── docs/                      # 学习指南 / 开发顺序 / Dify 聊天嵌入示例
```

---

## 核心设计

### 三层记忆

```
① profile（档案）  —— 姓名、生辰等固定信息，单独抽出，永不压缩
② summary（摘要）  —— 更早的对话压成一段 300 字以内的话，满 10 条压一次
③ history（原文）  —— 最近 3 轮原样保留，保证"刚才说到哪"不丢
```

拼成一段文字后作为 `memory_text` 塞进 system 提示词。压缩失败不会影响聊天——
记忆原样留着，下一轮再试。

### 情绪 → 语气

一轮对话的完整链路：

```
用户提问
   ↓
qingxu_chain()  判断情绪（白名单校验，不在名单里一律按 default）
   ↓
get_role_set()  用情绪查出对应的语气描述
   ↓
create_executor()  基础人设 + 本次语气 → 拼成新的 system 提示词 → 造一个执行器
   ↓
AgentExecutor.invoke(  带上"这个会话的"历史记忆  )
   ↓
存进记忆（超过 10 条自动压缩）
```

优先级：`angry` > `depressed` > `cheerful` > `upbeat` > `friendly` > `default`

### 工具出错为什么不报错？

工具直接把异常原文返回给模型，模型就会原样转述给用户——用户就看到了"密钥""状态码"这些技术词。
所以定死两条规矩：

1. 技术细节（异常信息、状态码、原始响应体）**只能在 `print` 里出现**，绝不放进 `return`；
2. `return` 出去的不是"错误"，而是**给模型看的指令**：用角色内的话把这一节带过去，
   并明确禁止提技术词、禁止编造精确的排盘结果。

### CORS 为什么只放行 `/chat`？

双击打开 `h5.html` 时地址栏是 `file:///...`，而接口在 `http://127.0.0.1:8000`，
浏览器会按跨域拦掉请求。所以只给 `/chat` 加 CORS 头——
`/memory` 这类含聊天记录的接口不放行，否则随便哪个网页都能跨域读走你的聊天记录。

---

## 常见问题

**Q：启动后提示"连不上老夫的卦摊"？**
服务没起来，或者端口不对。确认终端里有 `Application startup complete`，
再检查 `h5.html` 里的 `API` 地址是否指向正确的 IP 和端口。

**Q：端口 8000 被占用？**
双击 `启动服务.bat` 会自动清理占用 8000 端口的进程。
或在 `docker-compose.yml` 里把 `"8000:8000"` 改成 `"8080:8000"`。

**Q：`docker compose up` 报 `project name must not be empty`？**
目录名带中文导致的。`docker-compose.yml` 里已经写了 `name: chen-master` 来避开这个坑，
如果你把文件挪走了，记得保留这一行。

**Q：回答是空的？**
DeepSeek 的推理模型会先思考，`max_tokens` 太小的话思考过程就把额度吃光了。
调到 2000 以上。

**Q：联网搜索没反应？**
没配 `BOCHA_API_KEY`。去 <https://open.bochaai.com> 微信扫码登录 → API Key 管理 → 创建密钥。
不配也不影响算命、八字、知识库这些功能。

**Q：知识库检索报文件锁错误？**
Qdrant 本地模式靠文件锁独占，用完后必须 `client.close()` 释放。代码里已经用 `try/finally` 包住了。

---

## 安全提醒

- 🔑 **`config.py` 和 `.env` 里是你的真实密钥，永远不会上传到 git。**
  换电脑或部署到服务器时，需要重新填一次。
- 🚫 千万别为了"图方便"把密钥硬编码进 `server.py` 或提交进 `config.py`——
  密钥一旦上 GitHub，几分钟内就会被爬虫扫到并盗刷。
- 💬 `memory.json` / `memory_*.json` 里存着**用户的聊天记录**，同样已被 `.gitignore` 排除。
- 🔒 `/memory` 系列接口会暴露聊天记录，目前没有鉴权，**请勿直接暴露到公网**。
  要对外提供服务，建议加一层反向代理 + 访问控制。
- 📦 `data/`、`local_qdrant/` 和模型缓存（约 90MB+）都不上传，容器里由数据卷接管。

---

## 开源协议

本项目仅供学习交流使用。涉及命理测算的内容仅供参考娱乐，不构成任何决策依据。

`Mytools.py` 中八字接口的 `api_key` 是教程演示密钥，**已失效**，
请自行到 <https://doc.yuanfenju.com> 注册申请。
