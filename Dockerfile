# KubeSage Docker Configuration
# FastAPI backend + Streamlit frontend + PostgreSQL

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=20

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Error while installing torch from requirements.txt hence pip install from docker file
RUN pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    --retries=20 \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    --retries=20 \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]