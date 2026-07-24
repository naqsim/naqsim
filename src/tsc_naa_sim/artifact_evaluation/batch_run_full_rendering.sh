#!/usr/bin/env bash
#SBATCH --job-name=ae_NAQsim_rendering
#SBATCH --partition=FIXME
#SBATCH --time=FIXME
#SBATCH --output=logs/rendering_%A.out
#SBATCH --error=logs/rendering_%A.err


source "../../../.venv/bin/activate"

set -euo pipefail

echo "started_at=$(date --iso-8601=seconds)"

"./run_full_rendering.sh"

echo "finished_at=$(date --iso-8601=seconds)"
