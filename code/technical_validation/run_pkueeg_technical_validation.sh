#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/pkueeg_completeness_temporal.py"
bash "${SCRIPT_DIR}/run_pkueeg_trf.sh"
bash "${SCRIPT_DIR}/run_pkueeg_validation_recheck.sh"
"${PYTHON_BIN}" "${SCRIPT_DIR}/pkueeg_low_neural_qc.py"
printf '%s\n' 'Behavior analysis skipped: supply an authorized external TSV explicitly to run that branch.'

if [[ "${RUN_RAW_QUALITY:-0}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/data_quality/run_quality.sh"
fi
