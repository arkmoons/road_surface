ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG APP_ENV=production
ENV APP_ENV=${APP_ENV}

WORKDIR /app

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY ml/registry/ ./ml/registry/
COPY models/current.pt ./models/current.pt

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/current.pt

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
