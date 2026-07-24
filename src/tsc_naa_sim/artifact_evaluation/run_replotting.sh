#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (($#)); then
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        echo "Usage: ./run_replotting.sh"
        echo "Render every figure/table from the supplied raw_data tree."
        exit 0
    fi
    echo "error: run_replotting.sh does not accept arguments" >&2
    exit 2
fi

cd "$SCRIPT_DIR"
export MPLBACKEND=${MPLBACKEND:-Agg}

echo "[replotting] fig10"
uv run python "$SCRIPT_DIR/fig10/analyze.py" \
    --input-dir "$SCRIPT_DIR/raw_data/fig10"

for target in fig11a table3 fig15 fig16 fig17 fig19 fig18 fig20; do
    args=(--input-dir "$SCRIPT_DIR/raw_data/$target")
    echo "[replotting] $target"
    uv run python "$SCRIPT_DIR/$target/analyze.py" "${args[@]}"
done
