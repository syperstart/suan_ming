# ============================================================
#  陈大师服务 —— Docker 镜像
#
#  一句话说明它是干嘛的：
#    把"能跑这套代码的运行环境"打包成一个盒子，换台电脑也能一模一样地跑起来。
#
#  ★ 你一般不用直接碰这个文件，真正要改配置去 docker-compose.yml。
# ============================================================

# 基础镜像：Python 3.12 精简版（体积小）。
# 想跟本机 Python 3.14 完全一致，把下面这行改成 python:3.14-slim 即可。
FROM python:3.12-slim

# 让 print 出来的日志立刻出现在 docker logs 里（不加的话要攒满缓冲区才看得到）
ENV PYTHONUNBUFFERED=1
# 不生成 .pyc 缓存文件，镜像更干净
ENV PYTHONDONTWRITEBYTECODE=1
# 模型缓存目录。挂出来之后，重建容器就不用重新下载那个中文嵌入模型（约 400MB）
ENV HF_HOME=/root/.cache/huggingface
# 国内服务器下载模型走镜像站；如果你在国外或网络通畅，把这行注释掉即可
ENV HF_ENDPOINT=https://hf-mirror.com
# 日志时间按北京时间显示
ENV TZ=Asia/Shanghai

WORKDIR /app

# 装编译工具：精简镜像里默认没有，个别依赖包需要现场编译
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ★★ 先只复制依赖清单（不复制代码）
#    这是 Docker 的小技巧：改代码不会让上面的 pip 重装一遍，重建快很多
COPY requirements.txt .

# ★ torch 单独用「CPU 版源」安装 —— 默认源会拉 2GB 多的 CUDA 版本，慢且用不上
#   若报找不到 2.10.0 这个版本，把 ==2.10.0 删掉再构建
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# 装其余依赖。如果服务器在国内且访问 PyPI 慢，在下面这行末尾加：
#   -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt

# 最后复制代码
COPY . .

# 数据目录（记忆 / 知识库 / 模型缓存）。真正的内容由 docker-compose 挂载进来
RUN mkdir -p /app/data/memory /app/data/local_qdrant /app/data/hf_cache

EXPOSE 8000

# 健康检查：每 60 秒问一次 GET /（这个接口只返回 {"Hello":"World"}，不依赖模型和数据库）
# start-period=90s 是给"加载嵌入模型"留出缓冲时间，避免刚启动就被判定为不健康
HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)"

CMD ["python", "server.py"]
