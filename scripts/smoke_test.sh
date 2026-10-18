#!/usr/bin/env bash
# Verify a running API: ./scripts/smoke_test.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8000}"
echo "== GET $BASE/health";     curl -fsS "$BASE/health"; echo
echo "== GET $BASE/ready";      curl -fsS "$BASE/ready"; echo
echo "== GET $BASE/model-info"; curl -fsS "$BASE/model-info" | head -c 400; echo
echo "== POST $BASE/predict";   curl -fsS -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d @"$(dirname "$0")/sample_request.json"; echo
echo "== GET $BASE/metrics (excerpt)"; curl -fsS "$BASE/metrics" | grep -E '^(api_requests_total|model_predictions_total)' | head
