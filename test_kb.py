# ============================================================
#  test_kb.py —— 测试本地知识库（先"存"资料，再"取"资料）
#
#  运行：在 D:\智能体项目 目录下执行  python test_kb.py
#
#  这个脚本干的事，就是以后 /add_texts、/add_pdfs、/add_urls 要干的事：
#  把资料存进本地向量库（./local_qdrant），然后试着按意思检索出来。
# ============================================================

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

PATH = "./local_qdrant"          # 数据库文件夹（和 server.py 里用的必须一致）
COLLECTION = "local_documents"   # 集合名（也必须一致）


# ---------- 第 1 步：加载本地嵌入模型 ----------
print("正在加载本地嵌入模型（第一次会慢几秒）...")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
print("模型加载完成")


# ---------- 第 2 步：准备资料 ----------
docs = [
    "龙年运势：属龙的人在本命年宜稳不宜动，事业上适合守成，不宜进行大额投资。",
    "八字入门：八字由年柱、月柱、日柱、时柱组成，每柱一个天干加一个地支，一共八个字。",
    "紫微斗数：以命宫为核心，通过十二个宫位来分析一个人的性格特点与运势走向。",
]


# ---------- 第 3 步：创建集合（先清空重建，方便反复测试）----------
client = QdrantClient(path=PATH)
vector_size = len(EMBEDDINGS.embed_query("测试"))   # 问一下模型：它输出的坐标是几维的
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)
print("集合已建好，向量维度：", vector_size)


# ---------- 第 4 步：把资料存进向量库 ----------
store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    embedding=EMBEDDINGS,
)
store.add_texts(docs)
print("已存入资料条数：", len(docs))


# ---------- 第 5 步：检索（和 server.py 里那个工具用的是同一套写法）----------
retriever = store.as_retriever(search_type="mmr")

questions = ["龙年运势怎么样", "八字是什么", "今天天气如何"]
for q in questions:
    results = retriever.invoke(q)
    print("\n问题：", q)
    if len(results) == 0:
        print("    （没有检索到内容）")
    for i, doc in enumerate(results, 1):
        print("    结果", i, "：", doc.page_content[:50])
