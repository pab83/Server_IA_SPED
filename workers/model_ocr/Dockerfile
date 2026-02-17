FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Para evitar problemas con OpenMP / threads en WSL2
ENV OMP_NUM_THREADS=1
ENV MKL_THREADING_LAYER=GNU
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Librerías de sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY workers/model_ocr/requirements.txt ./requirements.txt

# Actualizar pip antes de instalar
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Código compartido
COPY common/schemas ./schemas
COPY common/reliability ./reliability
COPY common/messaging ./messaging
COPY workers/base/base_worker.py ./base_worker.py
COPY workers/base/lifecycle.py ./lifecycle.py

# Código del worker
COPY workers/model_ocr/app ./app

ENV PYTHONPATH=/app

CMD ["python", "app/main.py"]
