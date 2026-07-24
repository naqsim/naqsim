#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TSC_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
ANALYSIS_TARGETS=(fig11a table3 fig15 fig16 fig17 fig19 fig18 fig20)

usage() {
    cat <<'EOF'
Usage: ./run_full_rendering.sh [options]

Render every paper-facing figure and table after all workload-wise simulation
and analysis jobs have completed.

Options:
  --quration-results-dir DIR  Quration prerequisite root (default: quration/results)
  --dry-run                   Show rendering commands without executing them
  -h, --help
EOF
}

quration_results_dir=""
dry_run=false

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --quration-results-dir)
            if (($# < 2)); then
                echo "error: $1 requires a value" >&2
                exit 2
            fi
            quration_results_dir=$2
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$quration_results_dir" ]]; then
    quration_results_dir=${AE_QURATION_RESULTS_DIR:-"$SCRIPT_DIR/quration/results"}
fi

render_command() {
    if [[ "$dry_run" == true ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

cd "$SCRIPT_DIR"
export MPLBACKEND=${MPLBACKEND:-Agg}

echo "[full-rendering] rendering Fig. 10 from its supplied decoding data"
render_command uv run python "$SCRIPT_DIR/fig10/analyze.py" \
    --input-dir "$SCRIPT_DIR/raw_data/fig10"

for target in "${ANALYSIS_TARGETS[@]}"; do
    case "$target" in
        fig11a|table3)
            input_dir="$TSC_DIR/output/comp_opt_evaluation_fixed_d"
            ;;
        fig15)
            input_dir="$TSC_DIR/output/d3rot_opt_evaluation_fixed_d"
            ;;
        fig16|fig17)
            input_dir="$TSC_DIR/output/final_evaluation_fixed_d"
            ;;
        fig18)
            input_dir="$TSC_DIR/output/sensitivity_test_evaluation_fixed_d"
            ;;
        fig19)
            input_dir="$TSC_DIR/output/ms_overhead_fixed_d"
            ;;
        fig20)
            input_dir="$TSC_DIR/output/hs_skip_evaluation_fixed_d"
            ;;
    esac

    command=(uv run python "$SCRIPT_DIR/$target/analyze.py" --input-dir "$input_dir")
    if [[ "$target" == "fig17" || "$target" == "fig19" ]]; then
        command+=(--input-dir "$quration_results_dir")
    fi
    echo "[full-rendering] rendering $target"
    render_command "${command[@]}"
done
