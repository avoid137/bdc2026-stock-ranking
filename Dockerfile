FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make -j1 \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 复制本地 PyTorch .whl 文件
COPY torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl /tmp/

# 从本地文件安装 PyTorch
RUN pip install --no-cache-dir /tmp/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl

# 配置 pip 使用清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ⭐ 分段安装其他依赖
RUN pip install --no-cache-dir --default-timeout=1000 torchvision torchaudio
RUN pip install --no-cache-dir --default-timeout=1000 torch-geometric
RUN pip install --no-cache-dir --default-timeout=1000 numpy pandas scikit-learn
RUN pip install --no-cache-dir --default-timeout=1000 tqdm joblib tensorboardX TA-Lib

# 复制代码
COPY app /app

RUN chmod +x /app/*.sh

CMD ["/bin/bash"]