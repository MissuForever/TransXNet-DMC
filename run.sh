#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/train_${TIMESTAMP}.err"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "============================================"
echo "  Start training: $(date)"
echo "  Python: $PYTHON_BIN"
echo "  Stdout: $LOG_FILE"
echo "  Stderr: $ERR_FILE"
echo "============================================"

"$PYTHON_BIN" -u fl.py > >(tee "$LOG_FILE") 2> >(tee "$ERR_FILE" >&2)

echo "============================================"
echo "  Finished: $(date)"
echo "============================================"
