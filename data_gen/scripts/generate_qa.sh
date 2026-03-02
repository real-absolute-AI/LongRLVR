#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_GEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <clustered.jsonl> [more_clustered.jsonl ...]" >&2
    exit 1
fi

python "${DATA_GEN_DIR}/generate_qa_batch.py" \
    --input_data_paths "$@" \
    --output_data_path generated_questions.jsonl \
    --organized_output_path organized_questions.json
