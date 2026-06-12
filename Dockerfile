FROM python:3.12-slim

# 安装 curl 用于 Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 注意：API Key 通过 docker-compose.yml 的 environment 注入，不在镜像中硬编码
CMD ["python", "-m", "app.main"]
