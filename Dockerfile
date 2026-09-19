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

# ★★ 第一步必须先换 apt 源！！
#    Debian 13 用的是 deb822 格式（源文件在 /etc/apt/sources.list.d/debian.sources）
#    官方源 deb.debian.org 在国内服务器实测只有 82 KB/s，
#    不换源的话 apt-get update 会卡十几分钟甚至超时。
#    换成腾讯云内网镜像后实测 3.6 MB/s，快 45 倍。
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.tencentyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.tencentyun.com|g; s|security.debian.org|mirrors.tencentyun.com|g' /etc/apt/sources.list; \
    fi

# 装编译工具：精简镜像里默认没有，个别依赖包需要现场编译
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ★★ 先只复制依赖清单（不复制代码）
#    这是 Docker 的小技巧：改代码不会让上面的 pip 重装一遍，重建快很多
COPY requirements.txt .

# ★★ torch 必须分两步装，这是国内构建的关键（踩过坑！）
#    第 1 步：torch 的"纯 Python 依赖"（sympy / networkx / jinja2 等）走腾讯云镜像 → 快
#    第 2 步：只把 torch 本体从 pytorch 官方源拉（CPU 版约 196MB，实测 4.5MB/s）
#    如果合成一句装，pip 会跑去 pytorch 源拉那几个小包，
#    实测每个只有 20KB/s，能卡十几分钟不动（这就是一开始构建卡死的原因之一）。
RUN pip install --no-cache-dir -i https://mirrors.tencentyun.com/pypi/simple \
        filelock typing-extensions sympy networkx jinja2 fsspec mpmath MarkupSafe

# torch 用「CPU 版源」—— 默认源会拉 2GB 多的 CUDA 版本，又慢又用不上
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# ★ 其余依赖走腾讯云 PyPI 镜像（国内实测 1.4 MB/s，比官方源快几十倍）
RUN pip install --no-cache-dir -i https://mirrors.tencentyun.com/pypi/simple -r requirements.txt

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
