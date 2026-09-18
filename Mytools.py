# ============================================================
#  Mytools.py —— 我自己写的工具集合
#
#  记住两条规矩，以后加工具就不会错：
#    1. 一个工具 = 一个普通函数 + 上面加 @tool
#    2. 这个文件必须能"自己站住"：用到的模块都在这里 import，
#       ★ 绝对不要去 import server.py（否则和 server 互相导入 → 程序崩）
#
#  ★ 加完新工具，记得去 server.py 改两个地方：
#      from Mytools import ...        （导入它）
#      self.tools = [...]             （加进工具清单，模型才用得到）
# ============================================================

import os                                                # ★ 读环境变量（Docker 部署时用它指定知识库目录）
import requests                                          # 用来调用外部的八字接口（注意是 requests，不是 request）
from pathlib import Path                                 # 用来拼"本文件所在的目录"

from config import API_KEY, MODEL_NAME, BASE_URL, BOCHA_API_KEY  # 公共配置（密钥 / 模型名 / 地址）

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool                    # ★ 必须有！少了它 @tool 会报"未定义"
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader   # 抓网页用
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 把长网页切成小块用


# 数据库文件夹：用"本文件所在目录"来拼。
# 好处：不管你在哪个目录运行，都能找对地方（相对路径会跟着工作目录变，容易出错）
# ★ 部署到 Docker 时，用环境变量 DB_PATH 指向挂载出来的数据目录（不设就和以前一样）
DB_PATH = os.environ.get("DB_PATH") or str(Path(__file__).parent / "local_qdrant")

# 集合名：存资料、查资料都用这一个常量，避免两边写成不一样的名字
KB_COLLECTION = "local_documents"

# 切片参数：和 server.py 的 /add_urls 保持一致
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50

# 一次搜索最多把前几条网页学进知识库（太多会拖慢速度）
SEARCH_TOP_N = 3


# ---------- 出错时的"统一话术"（★ 本次改动新增）----------
# 背景：工具出错时，如果直接把接口的报错原文返回给模型，
#       模型会原样转述给用户 —— 用户就看到"密钥""状态码"这些技术词了。
# 所以规矩定死两条：
#   ① 技术细节（异常信息 e、状态码、errmsg、原始响应体）
#      只能在 print 里出现，绝不放进 return
#   ② return 出去的不是"错误"，而是"给模型看的指令"：
#      用角色内的话把这一节带过去，并且明确禁止提技术词
# ★ 按业务分成四份，措辞各自贴合场景（八字 / 搜索 / 网页抓取 / 查知识库）

# 八字测算失败时，返回这句
BAZI_ERROR_MSG = (
    "【工具暂时不可用】请用角色内的话告诉用户：今日天机未通，排盘一时得不到印证，"
    "这一节先按命理与常识做方向性推演，并提醒他稍后再试。"
    "★ 禁止提到接口、密钥、报错、状态码、模型名等技术词；"
    "★ 禁止编造精确的排盘/测算结果。"
)

# 联网搜索失败时，返回这句
SEARCH_ERROR_MSG = (
    "【工具暂时不可用】请用角色内的话告诉用户：今日天机未通，外头的消息一时探听不到，"
    "这一节先按老夫已知的命理与常识做方向性推演，并提醒他稍后再试。"
    "★ 禁止提到接口、密钥、报错、状态码、模型名等技术词；"
    "★ 禁止编造精确的排盘/测算结果。"
)

# 抓网页失败时，返回这句
URL_ERROR_MSG = (
    "【工具暂时不可用】请用角色内的话告诉用户：今日天机未通，他发来的这个网址老夫一时打不开，"
    "这一节先按命理与常识做方向性推演，并提醒他稍后再试。"
    "★ 禁止提到接口、密钥、报错、状态码、模型名等技术词；"
    "★ 禁止编造精确的排盘/测算结果。"
)

# 查本地知识库失败时，返回这句
KB_ERROR_MSG = (
    "【工具暂时不可用】请用角色内的话告诉用户：今日天机未通，老夫手边这本册子一时翻不开，"
    "这一节先按命理与常识做方向性推演，并提醒他稍后再试。"
    "★ 禁止提到接口、密钥、报错、状态码、模型名等技术词；"
    "★ 禁止编造精确的排盘/测算结果。"
)


# 嵌入模型（本地免费的中文模型）。
# ★ 只加载一次，别写在函数里面——否则每次调用工具都要重新加载，要等好几秒
EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 工具里要用的模型。工具文件自带一个，不依赖 server.py（避免循环导入）
TOOL_MODEL = ChatOpenAI(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=1,
)


# ---------- 工具一：测试用的假工具 ----------
@tool
def test():
    """Test tool"""
    # ★ 上面这行文档字符串就是"工具说明书"，模型靠它判断什么时候该用这个工具
    return "test"


# ---------- 工具二：查本地知识库 ----------
@tool
def get_info_from_local_db(query: str):
    """当用户问的是本地知识库资料里的内容（比如命理资料、运势资料）时，用这个工具检索。"""
    # 第一步：打开本地的"向量数据库"
    # ★ 本地 Qdrant 靠文件锁独占，同一个连接不关掉，下次再开就会报 AlreadyLocked
    client = QdrantClient(path=DB_PATH)
    try:
        store = QdrantVectorStore(
            client=client,
            collection_name=KB_COLLECTION,                # 集合名，和"存资料"时用的一致
            embedding=EMBEDDINGS,                         # 用哪个模型把文字变成坐标
        )
        # 第二步：造一个检索器。mmr = 结果既相关、又不重复
        retriever = store.as_retriever(search_type="mmr")
        # 第三步：拿用户的问题去检索，取回最相关的几段资料
        return retriever.invoke(query)
    except Exception as e:
        # ★ 库里出问题（比如文件被别的进程占用）照样只打印在终端，不往外交
        print("【排查信息，仅终端可见】查知识库失败：", e)
        return KB_ERROR_MSG
    finally:
        client.close()                                    # ★ 用完必须关掉，释放文件锁


def learn_one_url(url):
    """抓一个网页 → 切片 → 存进知识库。
    返回 (存了几段, 正文)；出错时返回 (0, 错误说明)。
    这个函数不是工具，是给下面两个工具共用的。"""
    # 第一步：抓网页
    try:
        docs = WebBaseLoader(url).load()
    except Exception as e:
        # ★ 异常原文只打印在终端；返回值只给一句不含技术词的原因
        print("【排查信息，仅终端可见】抓网页失败：", e)
        return 0, "网页抓取失败"

    # 第二步：切成小块（参数和 server.py 的 /add_urls 保持一致）
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(docs)

    # 第三步：存进知识库
    client = QdrantClient(path=DB_PATH)
    try:
        store = QdrantVectorStore(
            client=client,
            collection_name=KB_COLLECTION,
            embedding=EMBEDDINGS,
        )
        store.add_documents(chunks)
    except Exception as e:
        # ★ 同上：细节只留终端
        print("【排查信息，仅终端可见】写入知识库失败：", e)
        return 0, "写入知识库失败"
    finally:
        client.close()                                    # ★ 用完关掉，别占着文件锁

    # 第四步：返回段数和正文
    text = "\n".join(doc.page_content for doc in chunks)
    return len(chunks), text[:3000]


# ---------- 工具三：把网页学进知识库 ----------
@tool
def learn_from_url(url: str):
    """当用户发来一个网址（http/https 开头），想让你看看、记住、学习这个网页的内容时，
    用这个工具。它会抓取网页、切成小块存进本地知识库，并把网页正文返回给你。"""
    count, text = learn_one_url(url)
    if count == 0:
        # ★ 出错原因只在终端看得见；返回给模型的是"角色内话术"指令
        print("【排查信息，仅终端可见】learn_from_url 没抓成，原因：", text)
        return URL_ERROR_MSG
    return "已存入知识库，共 " + str(count) + " 段。网页正文如下：\n" + text


# ---------- 工具四：联网搜索 + 自动学进知识库 ----------
@tool
def search_and_learn(query: str):
    """当遇到你不知道的事、需要上网查资料的时候，用这个工具联网搜索。
    它会搜出最相关的几个网页，自动把它们存进本地知识库，并把搜索结果返回给你。"""
    # 第零步：先看 key 配了没有
    if not BOCHA_API_KEY:
        # ★ 让用户看到的是角色话术；"要配 key"这种运维信息只留终端
        print("【排查信息，仅终端可见】BOCHA_API_KEY 还没配：去 https://open.bochaai.com 拿 key 填进 config.py")
        return SEARCH_ERROR_MSG

    # 第一步：联网搜索
    try:
        resp = requests.post(
            "https://api.bochaai.com/v1/web-search",
            headers={
                "Authorization": "Bearer " + BOCHA_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "summary": True,        # 返回详细摘要，模型更好用
                "count": SEARCH_TOP_N,  # 只取前几条
                "freshness": "noLimit",  # 不限时间
            },
            timeout=30,
        )
    except Exception as e:
        # ★ 连不上的异常原文只打印，不返回
        print("【排查信息，仅终端可见】搜索接口连不上：", e)
        return SEARCH_ERROR_MSG

    if resp.status_code != 200:
        # ★ 状态码 + 原始响应体，以前是直接返回出去的（泄露风险最高的一处）
        #   现在全部只留在终端
        print("【排查信息，仅终端可见】搜索失败，状态码：", resp.status_code)
        print("【排查信息，仅终端可见】原始响应：", resp.text[:500])
        return SEARCH_ERROR_MSG

    # 第二步：把搜到的网页取出来
    try:
        data = resp.json().get("data") or {}
    except Exception as e:
        # 返回的不是 JSON（少见），同样只留终端
        print("【排查信息，仅终端可见】搜索结果不是合法的 JSON：", e)
        return SEARCH_ERROR_MSG
    pages = (data.get("webPages") or {}).get("value") or []
    if not pages:
        return "老夫搜遍了，没搜到跟这个相关的内容。"

    # 第三步：一条条学进知识库，同时把摘要整理出来
    lines = []
    for i, page in enumerate(pages, 1):
        title = page.get("name") or "（无标题）"
        url = page.get("url") or ""
        zhaiyao = (page.get("summary") or page.get("snippet") or "").strip()

        lines.append(str(i) + ". " + title)
        lines.append("   链接：" + url)
        lines.append("   摘要：" + zhaiyao[:300])

        if url:
            count, msg = learn_one_url(url)
            if count:
                lines.append("   → 已学进知识库，共 " + str(count) + " 段")
            else:
                # ★ msg 现在只有"网页抓取失败/写入知识库失败"这类不含技术词的原因
                lines.append("   → 这条暂时没学成（" + msg + "）")
        print("搜索结果 " + str(i) + "：", title)

    return "搜到 " + str(len(pages)) + " 条，已经试着学进知识库：\n" + "\n".join(lines)


# ---------- 工具五：八字测算 ----------
@tool
def bazi_cesuan(query: str):
    """当用户要做八字排盘、算命，而且说出了姓名和出生年月日时，才用这个工具；信息不全时不要用。"""

    # ========== 第一步：让模型把用户的话，解析成 JSON 参数 ==========
    # ★ 注意：下面提示词里的 api_key 是教程里的"演示密钥"，已经失效。
    #   你需要去 https://doc.yuanfenju.com 注册，拿到自己的 key，替换掉这一行里的那个。
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手，根据用户输入内容找出相关的参数并按json格式返回。
JSON字段如下：
- "api_key":"K0I5WcMce7jLMZzTw7v1ixsn0"
- "name":"姓名"
- "sex":"性别 0表示男 1表示女，根据姓名判断"
- "type":"日历类型 0农历 1公历，默认1"
- "year":"出生年份 例：1998"
- "month":"出生月份 例：8"
- "day":"出生日期 例：8"
- "hour":"出生小时 例：14"
- "minute":"0"
如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他的评论。

{format_instructions}

用户输入：{query}"""
    )

    # JSON 解析器：负责把模型返回的文字解析成字典
    parser = JsonOutputParser()

    # 把解析器要求的格式说明，填进模板里的 {format_instructions} 位置
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # 接成一条流水线：提示词 → 模型 → JSON 解析器
    chain = prompt | TOOL_MODEL | parser

    # 执行：把用户的话变成参数字典
    try:
        data = chain.invoke({"query": query})
    except Exception as e:
        # ★ 模型调用失败（密钥无效、模型服务异常等）也只打印在终端
        print("【排查信息，仅终端可见】八字参数解析失败：", e)
        return BAZI_ERROR_MSG
    print("=====解析出的参数=====")
    print(data)

    # ========== 第二步：把参数发给八字接口 ==========
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"

    try:
        result = requests.post(url, data=data, timeout=30)
    except Exception as e:
        # 连都连不上（断网、域名有问题等）
        print("【排查信息，仅终端可见】八字接口连接失败：", e)
        return BAZI_ERROR_MSG

    # ========== 第三步：处理接口返回的数据 ==========
    if result.status_code == 200:
        try:
            resp_json = result.json()                 # ★ 挪进 try 里，万一不是 JSON 也不会漏出去
            print("=====返回数据=====")
            print(resp_json)

            # 先看接口自己有没有报错：errcode 不是 0，说明接口那边有问题（最常见的就是密钥无效）
            if str(resp_json.get("errcode")) != "0":
                # ★★ 问题的根源就在这：以前把 errmsg 原样返回，模型照着说"密钥"，用户就看到了。
                #    现在报错原文只打印在终端，返回给模型的是角色内话术指令。
                print("【排查信息，仅终端可见】接口返回错误：", resp_json.get("errmsg"))
                return BAZI_ERROR_MSG

            # 接口没报错，才把八字结果取出来（★ 成功路径保持原样，一个字没改）
            return "八字为:" + resp_json["data"]["bazi_info"]["bazi"]
        except Exception as e:
            # 请求成功，但返回的数据结构和我们预期的不一样
            print("【排查信息，仅终端可见】解析返回数据出错：", e)
            return BAZI_ERROR_MSG

    else:
        # 请求本身就没成功（key 失效 / 参数不对 / 接口故障）
        # ★ 状态码和原始响应体只打印在终端，绝不返回
        print("【排查信息，仅终端可见】请求失败，状态码：", result.status_code)
        print("【排查信息，仅终端可见】返回内容：", result.text)
        return BAZI_ERROR_MSG
