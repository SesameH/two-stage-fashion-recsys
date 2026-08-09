#!/usr/bin/env bash
# Deploy the service to Google Cloud Run.
#
#   PROJECT_ID=my-project ./deploy/cloudrun.sh
#
# Prerequisites: gcloud installed and authenticated, billing enabled, and
# `make serve-data` already run so data/serving/ exists (it is baked into the image).
#
# This script does not run itself from CI. Publishing a public URL is a deliberate act.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-northeast1}"
SERVICE="${SERVICE:-hm-recsys}"
REPO="${REPO:-containers}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(date +%Y%m%d-%H%M%S)"

if [[ ! -f data/serving/meta.json ]]; then
  echo "data/serving/ is missing — run 'make serve-data' first" >&2
  exit 1
fi

echo "==> enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  --project "${PROJECT_ID}"

echo "==> ensuring Artifact Registry repo ${REPO} exists"
gcloud artifacts repositories describe "${REPO}" \
  --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location "${REGION}" --project "${PROJECT_ID}"

echo "==> building and pushing ${IMAGE}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
# linux/amd64 explicitly: Cloud Run does not run arm64 images, and building on an Apple Silicon
# machine produces arm64 by default. The failure mode is a container that exits immediately with
# no useful log line, so it is worth pinning.
docker build --platform linux/amd64 -t "${IMAGE}" .
docker push "${IMAGE}"

echo "==> deploying ${SERVICE}"
# --memory 2Gi: the service holds ~834 MB resident in parquet mode, and Cloud Run's /tmp is a
#   tmpfs that counts against the same limit, so any DuckDB spill eats into it. 1Gi is too tight.
# --min-instances 0: scale to zero. Cold start measured at 7 s to first 200 (reports/serving.md).
# --concurrency 8: each request holds a DataFrame of 300 candidates x 63 features; unbounded
#   concurrency on 2 vCPU turns a burst into an OOM rather than into queueing.
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 4 \
  --concurrency 8 \
  --timeout 60s \
  --set-env-vars HM_SERVE_MODE=parquet

URL=$(gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" \
        --region "${REGION}" --format 'value(status.url)')
echo "==> deployed: ${URL}"
echo "    console : ${URL}/"
echo "    health  : ${URL}/health"
echo "    metrics : ${URL}/metrics"
