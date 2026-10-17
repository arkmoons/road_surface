# Slim production image for the road-surface classifier.
# Switch base image with a build arg (e.g. the course Azure mirror for faster pulls).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Which environment this image was built for (staging / production).
ARG APP_ENV=production
ENV APP_ENV=${APP_ENV}

WORKDIR /app

# Install runtime deps first so this layer is cached when only code changes.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Application code, the production-model registry, and the trained weights.
COPY app/ ./app/
COPY ml/registry/ ./ml/registry/
COPY models/current.pt ./models/current.pt

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/current.pt

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
