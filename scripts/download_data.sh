#!/usr/bin/env bash
# Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.
# Primary source: UCI repository. Fallback: GitHub mirror (re-coded back to UCI coding).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m heart.data
