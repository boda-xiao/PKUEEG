#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/pkueeg_trf_prediction.py" --self-test

"${PYTHON_BIN}" "${SCRIPT_DIR}/pkueeg_trf_prediction.py" \
  --subjects all \
  --days all \
  --eeg-sfreq 250 \
  --envelope-sfreq 100 \
  --analysis-sfreq 250 \
  --tmin -0.3 \
  --tmax 0.6 \
  --alphas 1e2,1e3,1e4,1e5,1e6 \
  --folds 5 \
  --n-jobs 8
