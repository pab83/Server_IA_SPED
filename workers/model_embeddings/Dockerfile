FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requisitos específicos del worker de embeddings
COPY workers/model_embeddings/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Código compartido (schemas, reliability, messaging, base worker)
COPY common/schemas ./schemas
COPY common/reliability ./reliability
COPY common/messaging ./messaging
COPY workers/base/base_worker.py ./base_worker.py
COPY workers/base/lifecycle.py ./lifecycle.py

# Código del worker
COPY workers/model_embeddings/app ./app

ENV PYTHONPATH=/app

CMD ["python", "app/main.py"]
