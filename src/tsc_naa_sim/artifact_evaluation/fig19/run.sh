#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec uv run python "$SCRIPT_DIR/../common/run_pipeline.py" --target fig19 "$@"
