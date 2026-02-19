FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && \
    apt-get install -y git build-essential libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "transformers==4.56.1"

COPY workers/model_moondream/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common/schemas ./schemas
COPY common/reliability ./reliability
COPY common/messaging ./messaging
COPY workers/base/base_worker.py ./base_worker.py
COPY workers/base/lifecycle.py ./lifecycle.py

COPY workers/model_moondream/app ./app

ENV PYTHONPATH=/app

CMD ["python", "app/main.py"]
