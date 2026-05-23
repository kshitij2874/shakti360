#!/bin/bash
set -e

PROJECT_ID="deployfest-kv-2026"
REGION="asia-south1"
SERVICE_NAME="shakti-agent"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "═══════════════════════════════════════════════════"
echo "  ShaktiAgent — Deploying to Google Cloud Run"
echo "═══════════════════════════════════════════════════"

echo "[1/2] Building container image..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}"

echo "[2/2] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars "PROJECT_ID=${PROJECT_ID}" \
  --project "${PROJECT_ID}"

echo ""
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)")
echo "🌐 Service URL: ${SERVICE_URL}"
