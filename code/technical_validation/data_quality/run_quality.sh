#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$SCRIPT_DIR"
"$PYTHON_BIN" code/analyze_raw.py --config config.yaml "$@"
"$PYTHON_BIN" code/aggregate_quality.py --config config.yaml
