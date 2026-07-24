#!/usr/bin/env bash
#SBATCH --job-name=ae_NAQsim_full
#SBATCH --partition=FIXME
#SBATCH --time=FIXME
#SBATCH --array=1-8
#SBATCH --output=logs/full_sim_%A_%a.out
#SBATCH --error=logs/full_sim_%A_%a.err


source "../../../.venv/bin/activate"

set -euo pipefail

PARAM_FILE=${AE_PARAM_FILE:-"params_full_simulation.txt"}
MEMORY_BUDGET_GIB=${AE_MEMORY_BUDGET_GIB:-600} #FIXME

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "error: submit this file as a Slurm array job" >&2
    exit 2
fi

param_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")
read -r workload num_threads extra <<<"$param_line"
if [[ -z "${workload:-}" || -z "${num_threads:-}" || -n "${extra:-}" ]]; then
    echo "error: invalid parameter line $SLURM_ARRAY_TASK_ID in $PARAM_FILE: $param_line" >&2
    exit 2
fi


echo "===== artifact-evaluation array task ====="
echo "job_id=$SLURM_JOB_ID task_id=$SLURM_ARRAY_TASK_ID"
echo "workload=$workload num_threads=$num_threads memory_budget_gib=$MEMORY_BUDGET_GIB"
echo "parameter_file=$PARAM_FILE"
echo "started_at=$(date --iso-8601=seconds)"

"./run_full_simulation.sh" "$workload" \
    --num-threads "$num_threads" \
    --memory-budget-gib "$MEMORY_BUDGET_GIB"

echo "finished_at=$(date --iso-8601=seconds)"
