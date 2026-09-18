
# ---------- 第 1 步：导入要用到的东西 ----------
import json                                             # 记忆要存成 json 文件
import os                           # ★ 读环境变量（Docker 部署时用它指定记忆存放目录）
import traceback                    # 出错时打印详细错误，方便查原因
import uvicorn                      # 负责启动网站服务器
from datetime import datetime                           # 记忆里记录一下时间
from pathlib import Path                                # 拼"记忆文件"的路径
from fastapi import FastAPI, WebSocket                 # Web 框架本体 + WebSocket 类型
from fastapi.responses import FileResponse, Response   # 返回 h5.html 页面 / 返回跨域探路请求
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.websockets import WebSocketDisconnect   # 客户端断开连接时会抛出的"异常"

from langchain_openai import ChatOpenAI                          # 大模型客户端（能连 Kimi 这类兼容接口）
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 提示词模板相关
from langchain_core.output_parsers import StrOutputParser         # 把模型输出转成纯文本
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor  # Agent（智能体）相关
from config import API_KEY, MODEL_NAME, BASE_URL       # ★ 密钥和模型名统一放在 config.py，只写一遍
# EMBEDDINGS：Mytools 里那个中文嵌入模型。存资料必须用同一个，否则维数对不上、检索结果也是乱的
# KB_COLLECTION / CHUNK_SIZE / CHUNK_OVERLAP：知识库的集合名和切片参数，统一从 Mytools 拿，省得两边写不一样
from Mytools import (
    test, get_info_from_local_db, bazi_cesuan, learn_from_url, search_and_learn,
    EMBEDDINGS, KB_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP,
)

# ---------- 第 2 步：密钥 ----------
# ★ 密钥已经搬到 config.py 了，以后换密钥/换模型只改那一个文件，这里不用动

# 情绪白名单：模型判断出的情绪，必须在这个名单里才算数
EMOTIONS = ["depressed", "friendly", "default", "angry", "upbeat", "cheerful"]

# 说明：嵌入模型（EMBEDDINGS）已经搬到 Mytools.py 里去了。
# 工具自己管自己的依赖，两个文件互不打扰，这样才不会互相导入导致崩溃。


# ---------- 第 2.5 步：记忆（持久化 + 自动压缩 + 多会话隔离）----------
#
# 记忆分三层，越往上越不容易丢：
#   1. profile（档案）：姓名、生辰这类固定信息，抽出来单独存，永远不压缩
#   2. summary（摘要）：更早的对话压成的一段话，满 10 条压一次
#   3. history（原文）：最近 3 轮原样保留，保证"刚才说到哪"不丢
#
# ★ 会话隔离：每个 session_id 一份独立的记忆文件，不同人/不同窗口聊天互不串记忆。
#   不传 session_id 就算 "default"，还是用老的 memory.json（兼容已有数据）。

# 记忆文件放哪儿：默认放 server.py 旁边（和以前完全一样）
# ★ 部署到 Docker 时，用环境变量 MEMORY_DIR 指向挂载出来的数据目录，
#   这样重建/升级容器，聊天记忆也不会丢。
MEMORY_DIR = Path(os.environ["MEMORY_DIR"]) if os.environ.get("MEMORY_DIR") else Path(__file__).parent
MEMORY_DIR.mkdir(parents=True, exist_ok=True)   # ★ 目录不存在就建一个（全新部署时防止写入报错）
MEMORY_MAX = 10                                        # 原文超过这个条数，就触发一次压缩
MEMORY_KEEP = 3                                        # 压缩后保留最近几轮原文


def memory_file(session_id="default"):
    """根据 session_id 算出它该用哪个记忆文件。
    只保留字母数字和 -_ 这几种安全字符，防止奇怪的 id 拼出奇怪的路径。"""
    sid = "".join(c for c in str(session_id) if c.isalnum() or c in "-_") or "default"
    if sid == "default":
        return MEMORY_DIR / "memory.json"              # 默认会话沿用老文件，旧数据不丢
    return MEMORY_DIR / ("memory_" + sid + ".json")    # 其他会话：一个 id 一个文件


def all_sessions():
    """列出现在都有哪些会话（查看/管理用）"""
    result = ["default"] if (MEMORY_DIR / "memory.json").exists() else []
    for f in MEMORY_DIR.glob("memory_*.json"):
        result.append(f.stem[len("memory_"):])         # 文件名去掉前缀，剩下的就是 id
    return result


# 压缩用的提示词：注意它跟"算命先生"的人设无关，只负责把对话压短
SUMMARY_PROMPT = """你是一个记忆整理助手。请把【新增对话】合并进【旧摘要】和【旧档案】里。

要求：
1. 用第三人称客观记录，例如"用户叫张三，1998年8月8日出生"。不要用"老夫"之类的角色口吻。
2. summary 字段：300 字以内的对话摘要，保留用户问过的事和得到过的结论，去掉寒暄和重复内容。
3. profile 字段：抽取出用户的固定信息，只写能确定知道的字段；没提到的字段不要写。
4. 只返回下面这个格式的 JSON，不要有任何其他内容：
{{"summary":"300字以内的摘要","profile":{{"姓名":"张三","性别":"男","出生年月日":"1998-08-08"}}}}

【旧摘要】
{old_summary}

【旧档案】
{old_profile}

【新增对话】
{chats}"""


def load_memory(session_id="default"):
    """读记忆。文件不存在或坏了，就返回一份空记忆（程序照样能跑）"""
    empty = {"profile": {}, "summary": "", "history": []}
    file = memory_file(session_id)
    if not file.exists():
        return empty
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return empty
    # 兼容旧格式：以前存的是一个列表，现在是一个字典
    if isinstance(data, list):
        return {"profile": {}, "summary": "", "history": data}
    empty.update(data)
    return empty


def save_memory(memory, session_id="default"):
    """把记忆写回文件 —— 这一步就是持久化"""
    with open(memory_file(session_id), "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def add_memory(question, answer, chatmodel=None, session_id="default"):
    """新增一条记忆。
    chatmodel 传进来才会自动压缩；不传就只做截断（方便单独测试）"""
    memory = load_memory(session_id)
    memory["history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
    })

    if chatmodel is not None and len(memory["history"]) > MEMORY_MAX:
        save_memory(memory, session_id)              # 先存盘，压缩时才能读到最新内容
        compress_memory(chatmodel, session_id)       # 条数超了 → 压缩
    else:
        memory["history"] = memory["history"][-MEMORY_MAX:]
        save_memory(memory, session_id)


def parse_summary_json(text):
    """模型有时会多说两句，或者用 ```json 把内容包起来，这里都兜住"""
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")            # 从第一个 { 开始
    end = text.rfind("}")             # 到最后一个 } 结束
    return json.loads(text[start:end + 1])


def compress_memory(chatmodel, session_id="default"):
    """把老的对话压成"摘要 + 档案"，只留最近 MEMORY_KEEP 轮原文"""
    memory = load_memory(session_id)
    history = memory["history"]

    to_compress = history[:-MEMORY_KEEP]     # 要压缩的：最老的那部分
    if not to_compress:
        return
    keep = history[-MEMORY_KEEP:]            # 要保留的：最近几轮原文

    # 第一步：把要压缩的对话拼成文字
    chats = "\n".join(
        "用户：" + item["question"] + "\n陈玉楼：" + item["answer"]
        for item in to_compress
    )

    # 第二步：让模型合并出"新摘要 + 新档案"
    chain = ChatPromptTemplate.from_template(SUMMARY_PROMPT) | chatmodel | StrOutputParser()

    try:
        raw = chain.invoke({
            "old_summary": memory["summary"] or "（无）",
            "old_profile": json.dumps(memory["profile"], ensure_ascii=False),
            "chats": chats,
        })
        data = parse_summary_json(raw)
        memory["summary"] = data.get("summary") or memory["summary"]
        memory["profile"] = data.get("profile") or memory["profile"]
    except Exception as e:
        # 压缩失败不要紧：记忆原样留着，下轮再试，不会影响用户聊天
        print("[记忆压缩失败，本次跳过]", e)
        return

    # 第三步：只留最近几轮原文，存盘
    memory["history"] = keep
    save_memory(memory, session_id)
    print("[记忆] 已压缩（会话 " + str(session_id) + "），原文剩 " + str(len(keep)) + " 条")


def get_memory_text(count=MEMORY_KEEP, session_id="default"):
    """把"档案 + 摘要 + 最近原文"拼成一段文字，塞给模型当记忆看"""
    memory = load_memory(session_id)
    parts = []

    if memory["profile"]:
        parts.append("【用户档案】\n" + json.dumps(memory["profile"], ensure_ascii=False))

    if memory["summary"]:
        parts.append("【更早的对话摘要】\n" + memory["summary"])

    recent = memory["history"][-count:]
    if recent:
        lines = []
        for item in recent:
            lines.append("用户：" + item["question"])
            lines.append("陈玉楼：" + item["answer"])
        parts.append("【最近的对话】\n" + "\n".join(lines))

    if not parts:
        return "（暂无历史记录）"
    return "\n\n".join(parts)


def clear_memory(session_id="default"):
    """清空记忆（只清指定会话的）"""
    save_memory({"profile": {}, "summary": "", "history": []}, session_id)


# ---------- 第 3 步：创建网站应用 ----------
app = FastAPI()


# ---------- 第 3.1 步：允许"双击打开 h5.html"时也能调接口 ----------
# 背景：直接双击 h5.html 时，地址栏是 file:///D:/...，而接口在 http://127.0.0.1:8000。
#      浏览器认为这是"跨域"，默认会把请求拦掉，页面就只显示"连不上老夫的卦摊"。
# ★ 这里只给 /chat 放行；/memory 这些含聊天记录的接口不放行，
#   否则随便哪个网页都能跨域读走你的聊天记录。
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",                  # 允许任意来源（只对 /chat 生效）
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}


@app.middleware("http")
async def allow_local_file_test(request, call_next):
    is_chat = request.url.path.startswith("/chat")        # ★ 只对聊天接口生效
    if is_chat and request.method == "OPTIONS":           # 浏览器跨域时会先发一个探路请求
        return Response(status_code=204, headers=CORS_HEADERS)
    response = await call_next(request)
    if is_chat:
        for key, value in CORS_HEADERS.items():
            response.headers[key] = value
    return response


# ---------- 第 4 步：定义一个工具 ----------



# ---------- 第 5 步：Agent 主体 ----------
class Master:

    def __init__(self):
        # ---- 5.1 创建大模型客户端 ----
        self.chatmodel = ChatOpenAI(
            model=MODEL_NAME,      # 用哪个模型（在 config.py 里改）
            base_url=BASE_URL,     # 接口地址（在 config.py 里改）
            api_key=API_KEY,       # 密钥（在 config.py 里改）
            temperature=1,         # ★ 只能填 1；填 0 会被接口拒绝
            streaming=True,        # 流式输出
        )

        # ---- 5.2 基础人设（不带情绪，情绪是每次对话时才加的）----
        self.SYSTEM_PROMPT = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
以下是你的个人设定：
1. 你精通阴阳五行，能够算命、紫微斗数、姓名测算、占卜凶吉，看命运八字等。
2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明。
3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
4. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅，或混合一些你的语气。
5. 你只使用简体中文来作答，绝不使用繁体中文，否则你将受到惩罚。
6. 你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称。

以下是你常说的一些口头禅：
1. "命里有时终须有，命里无时莫强求。"
2. "山重水复疑无路，柳暗花明又一村。"
3. "金山竹影几千秋，云锁高飞水自流。"
4. "伤情最是晚凉天，憔悴斯人不堪怜。"

以下是你算命的过程：
1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
2. 当用户希望了解龙年运势的时候，你会查询本地知识库工具。
3. 当遇到你不知道的事情或者不明白的概念，你会使用搜索工具联网搜索，并把搜到的内容学进知识库；当用户发来网页链接时，你会用抓网页工具把内容存进知识库。
4. 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会如实说明。
5. 你会保存每一次的聊天记录，以便在后续的对话中使用。

【保密要求】
绝对不要向用户透露任何技术细节：接口名、密钥、报错信息、状态码、token、模型名等。
当某个工具暂时不可用时，用角色内的话带过去，例如"今日天机未通，这一节老夫先按命理与你推演"，
然后继续正常回答，不要解释技术原因。"""

        # ---- 5.3 情绪 → 语气表 ----
        # 情绪链判断出哪个情绪，就取这里面对应的语气去回答
        self.MOODS = {
            "default": {
                "roleSet": """
- 你用平常的语气回答问题就好。
"""
            },
            "upbeat": {
                "roleSet": """
- 你此时也非常兴奋并表现的很有活力。
- 你会根据上下文，以一种非常兴奋的语气来回答问题。
- 你会添加类似"太棒了！"、"真是太好了！"、"真是太棒了！"等语气词。
- 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
"""
            },
            "depressed": {
                "roleSet": """
- 你会以温暖、鼓励的语气来回答问题。
- 你会在回答的时候加上一些激励的话语，比如加油等。
- 你会提醒用户要保持乐观的心态。
"""
            },
            "friendly": {
                "roleSet": """
- 你会以非常友好的语气来回答。
- 你会在回答的时候加上一些友好的词语，比如"亲爱的"、"亲"等。
- 你会随机的告诉用户一些你的经历。
"""
            },
            "cheerful": {
                "roleSet": """
- 你会以非常愉悦和兴奋的语气来回答。
- 你会在回答的时候加入一些愉悦的词语，比如"哈哈"、"呵呵"等。
- 你会随机的告诉用户一些你的经历。
"""
            },
            "angry": {
                "roleSet": """
- 你会以平静、克制的语气回答问题。
- 你会先安抚用户的情绪，再解答问题。
- 你不会和用户争辩，也不会重复用户的不礼貌用语。
"""
            },
        }

        # ---- 5.4 工具清单 ----
        self.tools = [test, get_info_from_local_db, bazi_cesuan, learn_from_url, search_and_learn]

        # ---- 5.5 先建一个"默认语气"的执行器（备用）----
        default_role_set = self.get_role_set("default")
        self.agent_executor = self.create_executor(default_role_set)

    # ---------- 辅助方法一：按情绪查语气 ----------
    def get_role_set(self, emotion):
        # 从表里取情绪对应的语气说明；万一没这个情绪，就用 default 兜底
        mood = self.MOODS.get(emotion, self.MOODS["default"])
        return mood["roleSet"]

    # ---------- 辅助方法二：按指定语气，造一个执行器 ----------
    def create_executor(self, role_set):
        # 第一步：把"基础人设"和"本次语气要求"拼成一份新的 system 提示词
        system_text = self.SYSTEM_PROMPT + "\n\n【本次回答的语气要求】\n" + role_set

        # 第二步：组装提示词模板
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),                   # 人设 + 语气
                ("system", "【你们之前聊过的内容】\n{memory_text}"),  # ★ 历史记忆
                ("user", "{input}"),                       # ★ {input} 运行时换成用户问题
                MessagesPlaceholder("agent_scratchpad"),   # ★ 必须保留，否则报错
            ]
        )

        # 第三步：把模型、工具、提示词绑成 Agent
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=self.tools,
            prompt=prompt,
        )

        # 第四步：包成执行器并返回
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,   # ★ 必须有，少了会报错
            verbose=True,
        )
        return executor

    # ---------- 方法一：聊天（会先看情绪，再换语气回答）----------
    # session_id：谁在说话。不同会话各记各的，互不干扰；不传就用 default
    def run(self, query, session_id="default"):
        # 第一步：先判断用户情绪
        emotion = self.qingxu_chain(query)

        # 第二步：用情绪查语气表
        role_set = self.get_role_set(emotion)

        # 第三步：按这个语气造一个执行器
        executor = self.create_executor(role_set)

        # 第四步：真正让模型回答（顺便把"这个会话的"历史记忆一起给它看）
        result = executor.invoke({
            "input": query,
            "memory_text": get_memory_text(session_id=session_id),
        })

        # 只取模型说的话，返回给接口
        answer = result["output"]

        # 第五步：把这一轮问答存进"这个会话的"记忆（写到对应的 json 文件，重启不丢）
        #       传了 self.chatmodel，条数超过 10 时会自动压缩一次
        add_memory(query, answer, self.chatmodel, session_id)

        return {"emotion": emotion, "answer": answer}

    # ---------- 方法二：情绪识别（返回一个英文标签）----------
    def qingxu_chain(self, query):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则将受到惩罚。
2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将受到惩罚。
3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将受到惩罚。
4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry"，不要有其他内容，否则将受到惩罚。
5. 如果用户输入的内容比较兴奋，只返回"upbeat"，不要有其他内容，否则将受到惩罚。
6. 如果用户输入的内容比较悲伤，只返回"depressed"，不要有其他内容，否则将受到惩罚。
7. 如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则将受到惩罚。
如果同时符合多条规则，按这个优先级判断：angry > depressed > cheerful > upbeat > friendly > default。
用户输入的内容是：{input}"""

        # 把"提示词 + 模型 + 解析器"接成一条流水线
        chain = (
            ChatPromptTemplate.from_template(prompt)
            | self.chatmodel
            | StrOutputParser()
        )

        # 执行。键名必须叫 input，和提示词里的 {input} 对上
        raw = chain.invoke({"input": query})

        # 模型可能返回 "Depressed" 或 "depressed。"，所以清洗一下，一步一步来
        text = str(raw)          # 第一步：转成字符串
        text = text.strip()      # 第二步：去掉首尾空格
        text = text.strip('"')   # 第三步：去掉可能带着的英文引号
        text = text.strip("。")  # 第四步：去掉可能带着的中文句号
        text = text.lower()      # 第五步：统一转小写

        # 第六步：白名单校验，不在名单里一律当 default
        if text in EMOTIONS:
            return text
        return "default"


# ---------- 第 6 步：启动时创建 Agent ----------
try:
    MASTER = Master()
    print("[OK] Agent 初始化成功")
except Exception as e:
    MASTER = None
    traceback.print_exc()
    print("[警告] Agent 初始化失败：" + str(e))


# ---------- 第 7 步：HTTP 接口 ----------

@app.get("/")
def read_root():
    return {"Hello": "World"}


# 手机版聊天页面：GET /h5
# ★ 为什么要挂在同一个服务上？
#   页面和 /chat 接口同源，手机浏览器就不用处理"跨域"，分享出去也只要一个链接。
#   手机访问：http://电脑的局域网IP:8000/h5
@app.get("/h5")
def h5_page():
    page = Path(__file__).parent / "h5.html"       # 页面文件和 server.py 放在同一目录
    if not page.exists():
        return {"error": "找不到 h5.html，确认它和 server.py 在同一个目录"}
    return FileResponse(page)


# 聊天接口：POST /chat?query=你的问题&session_id=会话ID（不传就是 default）
@app.post("/chat")
def chat(query: str = "", session_id: str = "default"):
    if not query:
        return {"response": "I am a chatbot"}
    if MASTER is None:
        return {"response": "Agent 未初始化，请看终端里的报错"}
    try:
        out = MASTER.run(query, session_id)
        # 把"判断出的情绪"也一起返回，方便你看效果
        return {"emotion": out["emotion"], "response": out["answer"]}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# 只测情绪：POST /qingxu?query=你气死我了
@app.post("/qingxu")
def qingxu(query: str = ""):
    if MASTER is None:
        return {"response": "Agent 未初始化"}
    emotion = MASTER.qingxu_chain(query)
    return {"emotion": emotion}


# 看记忆：GET /memory 列出所有会话；GET /memory?session_id=xxx 看某个会话的
@app.get("/memory")
def show_memory(session_id: str = None):
    if session_id:
        return {"session_id": session_id, "memory": load_memory(session_id)}
    return {"sessions": all_sessions()}


# 清空记忆：POST /memory/clear?session_id=xxx（不传清 default）
@app.post("/memory/clear")
def do_clear_memory(session_id: str = "default"):
    clear_memory(session_id)
    return {"response": "记忆已清空", "session_id": session_id}


# 手动触发一次压缩：POST /memory/summarize?session_id=xxx（想马上看效果时用）
@app.post("/memory/summarize")
def do_summarize_memory(session_id: str = "default"):
    if MASTER is None:
        return {"response": "Agent 未初始化"}
    compress_memory(MASTER.chatmodel, session_id)
    return {"memory": load_memory(session_id)}


# 下面三个是教程里的占位接口，等做知识库（RAG）时再填真正的逻辑
@app.post("/add_urls")
def add_urls(URL:str):
    loader=WebBaseLoader(URL)
    docs=loader.load()
    docments=RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(docs)
    #引入向量数据库
    # ★ path 和 collection_name 必须和 Mytools.py 里检索时用的完全一致，否则存进去查不到
    qdrant=QdrantVectorStore.from_documents(
        docments,
        EMBEDDINGS,
        path=str(Path(__file__).parent / "local_qdrant"),
        collection_name=KB_COLLECTION
    )
    print("向量数据库创建完成")
    return {"ok":"添加成功"}


# 看知识库：GET /kb/count —— 想确认"资料到底进没进库"时，浏览器打开这个地址就行
@app.get("/kb/count")
def kb_count():
    from qdrant_client import QdrantClient        # 用的时候才导入，不影响启动
    client = QdrantClient(path=str(Path(__file__).parent / "local_qdrant"))
    try:
        counts = {c.name: client.count(c.name).count for c in client.get_collections().collections}
    finally:
        client.close()                            # ★ 用完关掉，别占着文件锁
    return {"collection": KB_COLLECTION, "counts": counts}


@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs are added"}


@app.post("/add_texts")
def add_texts():
    return {"response": "Texts are added"}


# ---------- 第 8 步：WebSocket 接口（长连接，目前是"复读机"）----------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()          # 接受连接（相当于接起电话）
    try:
        while True:                   # 一直循环，等客户端发消息
            data = await websocket.receive_text()
            reply = "Message text was: " + data
            await websocket.send_text(reply)
    except WebSocketDisconnect:       # 对方挂断 → 结束循环
        print("Connection closed")


# ---------- 第 9 步：启动服务 ----------
if __name__ == "__main__":
    # host="0.0.0.0" 是"监听本机所有网卡"的意思，不是访问地址
    # 访问请用 http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
