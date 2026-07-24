#!/usr/bin/env python3
"""Run separately cached AE compilation, execution, and analysis stages."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


AE_DIR = Path(__file__).resolve().parents[1]
TSC_DIR = AE_DIR.parent
REPO_DIR = TSC_DIR.parents[1]
WORKLOAD_CSV = AE_DIR / "config" / "workloads.csv"


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    ham_name: str
    distance: int
    num_threads: int
    fig19: bool


TRANSVERSAL_DISTANCE_COLUMN = "transversal_distance"

TARGET_STAGES = {
    "fig11a": ("comp_opt",),
    "table3": ("comp_opt",),
    "fig15": ("d3rot",),
    "fig16": ("comp_opt", "d3rot", "final_analysis"),
    "fig17": ("comp_opt", "d3rot", "final_analysis"),
    "fig18": ("sensitivity",),
    "fig19": ("d3rot", "ms_overhead"),
    "fig20": ("hs_skip",),
}

EVALUATION_SCRIPTS = {
    "comp_opt": "script_comp_opt_evaluation_fixed_d.py",
    "d3rot": "script_d3rot_opt_evaluation_fixed_d.py",
    "sensitivity": "script_sensitivity_test_evaluation_fixed_d.py",
    "hs_skip": "script_hs_skip_evaluation_fixed_d.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run existing simulator stages for one AE figure/table."
    )
    parser.add_argument("--target", required=True, choices=sorted(TARGET_STAGES))
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="Workload key, label, or full ham_name. Repeat to select several.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Override the per-workload worker count in config/workloads.csv.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "compilation", "execution", "analysis"),
        default="all",
        help=(
            "Run only one pipeline stage. 'all' runs compilation, execution, "
            "then analysis in that order."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=AE_DIR / ".cache",
        help="Shared content-addressed compilation/execution cache.",
    )
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        default=None,
        help="Maximum aggregate worker-memory budget for compilation/execution.",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=10,
        help="Monte Carlo shots for Fig. 19 D3 resource-state analysis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun analysis even when its expected CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--quration-results-dir",
        type=Path,
        default=AE_DIR / "quration" / "results",
        help=(
            "Prerequisite Quration CSV root for Fig. 17/19. Generate it first "
            "with artifact_evaluation/quration/run.sh."
        ),
    )
    return parser.parse_args()


def load_workloads(target: str, selectors: list[str], thread_override: int | None) -> list[Workload]:
    rows: list[Workload] = []
    with WORKLOAD_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            candidates = {row["key"], row["label"], row["ham_name"]}
            if target == "fig19" and row["fig19"] != "1":
                continue
            if selectors and not any(selector in candidates for selector in selectors):
                continue
            rows.append(
                Workload(
                    key=row["key"],
                    label=row["label"],
                    ham_name=row["ham_name"],
                    distance=int(row[TRANSVERSAL_DISTANCE_COLUMN]),
                    num_threads=thread_override or int(row["num_threads"]),
                    fig19=row["fig19"] == "1",
                )
            )
    if selectors:
        matched = {value for row in rows for value in (row.key, row.label, row.ham_name)}
        missing = [selector for selector in selectors if selector not in matched]
        if missing:
            raise ValueError(f"Unknown or unavailable workload selector(s) for {target}: {missing}")
    if not rows:
        raise ValueError(f"No workloads selected for {target}")
    return rows


def mapping_dir_name(workload: Workload) -> str:
    return f"naive_mapping_{workload.ham_name}_distance{workload.distance}"


def analysis_markers(stage: str, workload: Workload) -> list[Path]:
    ham = workload.ham_name
    dist = workload.distance
    if stage == "comp_opt":
        name = f"{ham}_case_study_1_analysis_d{dist}.csv"
        return [
            TSC_DIR / "output" / "comp_opt_evaluation_fixed_d" / mapping_dir_name(workload) / name,
            Path("/home/youte/comp_opt_res/comp_opt_csv") / name,
        ]
    if stage == "d3rot":
        name = f"{ham}_d3rot_analysis_d{dist}.csv"
        return [
            TSC_DIR / "output" / "d3rot_opt_evaluation_fixed_d" / mapping_dir_name(workload) / name,
            Path("/home/youte/d3rot_opt_res/d3rot_csv") / name,
        ]
    if stage == "sensitivity":
        name = f"{ham}_sensitivity_test_analysis_normalized_d{dist}.csv"
        return [
            TSC_DIR / "output" / "sensitivity_test_evaluation_fixed_d" / mapping_dir_name(workload) / name,
            TSC_DIR / "output" / "sensitivity_test_evaluation_fixed_d" / "csv" / name,
        ]
    if stage == "hs_skip":
        name = f"{ham}_hs_skip_analysis_d{dist}.csv"
        return [
            TSC_DIR / "output" / "hs_skip_evaluation_fixed_d" / mapping_dir_name(workload) / name,
            TSC_DIR / "output" / "hs_skip_evaluation_fixed_d" / "hs_skip_benchmarks_csv" / name,
        ]
    if stage == "ms_overhead":
        name = f"{ham}_ms_overhead_DIR_TOGL_DEDICATE_CELL2_d{dist}_sweep.csv"
        return [
            TSC_DIR / "output" / "ms_overhead_fixed_d" / mapping_dir_name(workload) / name,
        ]
    if stage == "final_analysis":
        name = f"{ham}_naive_mapping_d_{dist}.csv"
        return [
            TSC_DIR / "output" / "final_evaluation_fixed_d" / mapping_dir_name(workload) / name,
        ]
    raise ValueError(f"Unknown stage: {stage}")


def base_args(workload: Workload) -> list[str]:
    return [
        "--ham-name",
        workload.ham_name,
        "--code-distance",
        str(workload.distance),
        "--use-naive-mapping",
        "--num-threads",
        str(workload.num_threads),
    ]


def evaluation_command(
    pipeline_stage: str,
    stage: str,
    workload: Workload,
    cache_dir: Path,
    memory_budget_gib: float | None,
) -> list[str] | None:
    script = EVALUATION_SCRIPTS.get(stage)
    if script is None:
        return None
    command = [
        "uv", "run", "python", str(AE_DIR / "common" / "stage_runner.py"),
        "--stage", pipeline_stage,
        "--script", str(TSC_DIR / script),
        "--cache-dir", str(cache_dir),
    ]
    if memory_budget_gib is not None:
        command.extend(("--memory-budget-gib", str(memory_budget_gib)))
    command.extend(("--", *base_args(workload)))
    return command


def analysis_commands_for_stage(
    stage: str,
    workload: Workload,
    num_shots: int,
) -> list[list[str]]:
    args = base_args(workload)
    if stage == "comp_opt":
        return [["uv", "run", "python", "script_comp_opt_analysis_fixed_d.py", *args]]
    if stage == "d3rot":
        return [[
            "uv", "run", "python", "script_d3rot_opt_analysis_fixed_d.py", *args,
            "--detail-cfg-name", "DIR_TOGL+DEDICATE_CELL3",
        ]]
    if stage == "sensitivity":
        return [["uv", "run", "python", "script_sensitivity_test_analysis_fixed_d.py", *args]]
    if stage == "hs_skip":
        return [[
            "uv", "run", "python", "script_hs_skip_analysis_fixed_d.py", *args,
            "--detail-cfg-name", "D3ROT_CELL3_S_H",
        ]]
    if stage == "ms_overhead":
        return [[
            "uv", "run", "python", "script_ms_overhead_fixed_d.py", *args,
            "--cfg-name", "DIR_TOGL+DEDICATE_CELL2",
            "--num-shots", str(num_shots),
        ]]
    if stage == "final_analysis":
        return [[
            "uv", "run", "python", "script_analysis_final_evaluation_fixed_d.py", *args,
            "--d3rot-opt2-cfg-name", "DIR_TOGL+DEDICATE_CELL3",
        ]]
    raise ValueError(f"Unknown stage: {stage}")


def run_command(command: list[str], dry_run: bool, env: dict[str, str]) -> None:
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=TSC_DIR, env=env, check=True)


def check_quration_prerequisite(
    target: str,
    workloads: list[Workload],
    results_dir: Path,
    dry_run: bool,
) -> None:
    if target == "fig17":
        filename = "quration_eval_summary.csv"
    elif target == "fig19":
        filename = "quration_trace_ms_overhead_sweep.csv"
    else:
        return
    matches = sorted(results_dir.expanduser().resolve().rglob(filename))
    if not matches:
        message = (
            f"Quration prerequisite {filename} was not found below {results_dir}. "
            "Run artifact_evaluation/quration/run.sh first, or pass "
            "--quration-results-dir."
        )
        if dry_run:
            print(f"[prerequisite] {message}", flush=True)
            return
        raise FileNotFoundError(message)
    marker = matches[0]
    with marker.open(newline="") as handle:
        available = {row.get("ham_name", "") for row in csv.DictReader(handle)}
    missing = [item.ham_name for item in workloads if item.ham_name not in available]
    if missing:
        message = f"Quration prerequisite {marker} is missing workload(s): {missing}"
        if dry_run:
            print(f"[prerequisite] {message}", flush=True)
            return
        raise ValueError(message)
    print(f"[reuse] quration prerequisite: {marker}", flush=True)


def main() -> None:
    args = parse_args()
    workloads = load_workloads(args.target, args.workload, args.num_threads)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    cache_dir = args.cache_dir.expanduser().resolve()
    phases = (
        ("compilation", "execution", "analysis")
        if args.stage == "all"
        else (args.stage,)
    )

    # Fig. 17/19 intentionally treat Quration as an independent prerequisite,
    # but compilation/execution do not consume it.
    if "analysis" in phases:
        check_quration_prerequisite(
            args.target,
            workloads,
            args.quration_results_dir,
            args.dry_run,
        )

    for phase in phases:
        print(f"[phase] {phase}", flush=True)
        for stage in TARGET_STAGES[args.target]:
            for workload in workloads:
                if phase in ("compilation", "execution"):
                    command = evaluation_command(
                        phase,
                        stage,
                        workload,
                        cache_dir,
                        args.memory_budget_gib,
                    )
                    if command is None:
                        continue
                    print(
                        f"[run] {phase}/{stage}: {workload.label}, d={workload.distance}",
                        flush=True,
                    )
                    run_command(command, args.dry_run, env)
                    continue

                markers = analysis_markers(stage, workload)
                existing = next((path for path in markers if path.is_file()), None)
                if existing is not None and not args.force:
                    print(
                        f"[reuse] analysis/{stage}: {workload.label}, "
                        f"d={workload.distance}: {existing}",
                        flush=True,
                    )
                    continue
                print(
                    f"[run] analysis/{stage}: {workload.label}, d={workload.distance}",
                    flush=True,
                )
                for command in analysis_commands_for_stage(
                    stage, workload, args.num_shots
                ):
                    run_command(command, args.dry_run, env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
