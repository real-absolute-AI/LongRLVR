#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_GEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${INPUT_PATH:-${1:-data/documents.jsonl}}"
OUTPUT_PATH="${OUTPUT_PATH:-${2:-clustered.jsonl}}"
WORLD_SIZE="${WORLD_SIZE:-8}"

for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    CUDA_VISIBLE_DEVICES=$rank python "${DATA_GEN_DIR}/clustering.py" \
        --input_path "${INPUT_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --rank "${rank}" \
        --world_size "${WORLD_SIZE}" &
done
wait
