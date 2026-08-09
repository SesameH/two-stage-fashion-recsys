# Serving image. Contains the model, the precomputed snapshot and the API — not the raw data
# and not the training code paths' dependencies.
#
# Build:  docker build -t hm-recsys .
# Run:    docker run -p 8000:8000 hm-recsys
#
# data/serving/ must exist before building (`make serve-data`). It is deliberately baked into
# the image rather than mounted: the snapshot and the model are a matched pair, and shipping
# them together is what stops a container from serving a model against a stale snapshot.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Only the serving dependencies. lightgbm needs libgomp at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY sql/ ./sql/
COPY models/ranker.txt models/features.json ./models/
COPY data/serving/ ./data/serving/
COPY data/processed/customer_id_map.parquet ./data/processed/

# Cloud Run and App Runner inject the port to listen on and will fail the deployment if the
# container binds anything else. Fly and plain `docker run` do not set PORT, hence the default.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health')"

# `exec` so uvicorn becomes PID 1 and receives SIGTERM directly; without it the shell swallows
# the signal and the platform waits out its full termination grace period on every deploy.
CMD ["sh", "-c", "exec uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
