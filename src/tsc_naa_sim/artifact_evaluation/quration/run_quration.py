#!/usr/bin/env python3
"""Run the Quration prerequisite for Fig. 17 and Fig. 19.

The caller supplies the external qret executable directly.  This program creates
the same pipeline/topology configuration as the accepted-version notebooks and
stores Quration IR, ML, and profile JSON below quration/results/.  It then invokes
the standalone postprocessor to create the shared current-format CSV files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from common import QURATION_DIR, TSC_DIR, find_qasm, load_workloads, target_workloads


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
PIPELINE_TEMPLATE = TEMPLATE_DIR / "pipeline.yaml"
TOPOLOGY_TEMPLATE = TEMPLATE_DIR / "topology.yaml"


@dataclass(frozen=True)
class EvaluationConfig:
    use_magic_state_cultivation: bool
    magic_period: int
    magic_prob: int = 1
    magic_factory_seed_offset: int = 0
    reaction_time: int = 1
    pbc: bool = False
    physical_error_rate: float = 0.01
    drop_rate: float = 0.1
    code_cycle_time_sec: float = 0.000237


FIG17_CONFIG = EvaluationConfig(
    use_magic_state_cultivation=True,
    magic_period=1,
)
FIG19_CONFIG = EvaluationConfig(
    use_magic_state_cultivation=False,
    magic_period=0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external Quration and generate Fig. 17/19 prerequisite CSVs."
    )
    parser.add_argument(
        "--quration-bin",
        type=Path,
        required=True,
        help="Path to the external qret executable.",
    )
    parser.add_argument(
        "--target", choices=["fig17", "fig19", "both"], default="both"
    )
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="Workload key, label, or full ham_name. Repeat to select several.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=QURATION_DIR / "results"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=10,
        help="Monte Carlo shots for the Fig. 19 trace postprocessing.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun qret even if ML/profile JSON exists."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print qret commands without executing."
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Create Quration JSON only; do not generate the shared CSVs.",
    )
    return parser.parse_args()


def check_executable(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} executable does not exist: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"{label} is not executable: {resolved}")
    return resolved


def rectangle(n: int, max_diff: int = 2) -> tuple[int, int]:
    min_area = float("inf")
    best_dims = None
    for width in range(1, n + 1):
        for height in range(width, width + max_diff + 1):
            area = width * height
            if area >= n and area < min_area:
                min_area = area
                best_dims = (width * 2, height * 2)
            if area >= n:
                break
    if best_dims is None:
        raise ValueError(f"Could not construct a topology for {n} qubits")
    return best_dims


def create_topology(path: Path, num_qubits: int) -> None:
    width, height = rectangle(num_qubits)
    with TOPOLOGY_TEMPLATE.open() as handle:
        data = yaml.safe_load(handle)
    data["grids"][0]["coord"] = [width, height + 1, 0]
    data["grids"][0]["magic_factory"] = [
        {"symbol": index, "coord": [index * 2, 0]}
        for index in range(width // 2)
    ]
    qubits = []
    for row in range(height // 2):
        for column in range(width // 2):
            qubits.append(
                {"symbol": len(qubits), "coord": [column * 2, row * 2 + 2]}
            )
            if len(qubits) >= num_qubits:
                break
        if len(qubits) >= num_qubits:
            break
    data["grids"][0]["qubit"] = qubits
    with path.open("w") as handle:
        yaml.safe_dump(data, handle)


def create_pipeline(
    path: Path,
    ir_path: Path,
    ml_path: Path,
    topology_path: Path,
    config: EvaluationConfig,
) -> None:
    with PIPELINE_TEMPLATE.open() as handle:
        data = yaml.safe_load(handle)
    data["sc_ls_fixed_v0_reaction_time"] = config.reaction_time
    if config.use_magic_state_cultivation:
        data["sc_ls_fixed_v0_use_magic_state_cultivation"] = True
        data["sc_ls_fixed_v0_magic_factory_seed_offset"] = (
            config.magic_factory_seed_offset
        )
    else:
        # Quration enables cultivation based on key presence.
        data.pop("sc_ls_fixed_v0_use_magic_state_cultivation", None)
        data.pop("sc_ls_fixed_v0_magic_factory_seed_offset", None)
    data["sc_ls_fixed_v0_magic_generation_period"] = config.magic_period
    data["sc_ls_fixed_v0_prob_magic_state_creation"] = config.magic_prob
    data["sc_ls_fixed_v0_machine_type"] = "PBC" if config.pbc else "Dim2"
    data["sc_ls_fixed_v0_drop_rate"] = config.drop_rate
    data["sc_ls_fixed_v0_code_cycle_time_sec"] = config.code_cycle_time_sec
    data["sc_ls_fixed_v0_physical_error_rate"] = config.physical_error_rate
    data["sc_ls_fixed_v0_topology"] = str(topology_path.resolve())
    data["function"] = "main"
    data["input"] = str(ir_path.resolve())
    data["output"] = str(ml_path.resolve())
    with path.open("w") as handle:
        yaml.safe_dump(data, handle)


def render(command: list[str | Path]) -> str:
    return " ".join(str(item) for item in command)


def run_command(command: list[str | Path], env: dict[str, str], dry_run: bool) -> None:
    normalized = [str(item) for item in command]
    print(f"+ {render(normalized)}", flush=True)
    if not dry_run:
        subprocess.run(normalized, cwd=TSC_DIR, env=env, check=True)


def parse_num_qubits(ir_path: Path) -> int:
    with ir_path.open() as handle:
        data = json.load(handle)
    return int(data["circuit_list"][0]["argument"]["num_qubits"])


def run_profile(
    qret: Path,
    qasm_path: Path,
    output_root: Path,
    run_name: str,
    config: EvaluationConfig,
    env: dict[str, str],
    force: bool,
    dry_run: bool,
) -> None:
    run_dir = output_root / run_name
    ir_path = run_dir / f"{run_name}_ir.json"
    ml_path = run_dir / f"{run_name}_ml.json"
    pipeline_path = run_dir / f"{run_name}_pipeline.yaml"
    topology_path = run_dir / f"{run_name}_topology.yaml"
    profile_path = run_dir / f"{run_name}_profile.json"
    if ml_path.is_file() and profile_path.is_file() and not force:
        print(f"[reuse] {run_name}: {profile_path}", flush=True)
        return
    print(f"[run] {run_name}", flush=True)
    if dry_run:
        run_command([qret, "parse", "-i", qasm_path, "-o", ir_path], env, True)
        run_command([qret, "compile", "--verbose", "--pipeline", pipeline_path], env, True)
        run_command([qret, "profile", "-i", ml_path, "-o", profile_path], env, True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    run_command([qret, "parse", "-i", qasm_path, "-o", ir_path], env, False)
    create_topology(topology_path, parse_num_qubits(ir_path))
    create_pipeline(pipeline_path, ir_path, ml_path, topology_path, config)
    run_command([qret, "compile", "--verbose", "--pipeline", pipeline_path], env, False)
    run_command([qret, "profile", "-i", ml_path, "-o", profile_path], env, False)
    if not ml_path.is_file() or not profile_path.is_file():
        raise RuntimeError(f"Quration did not create the expected outputs for {run_name}")


def main() -> None:
    args = parse_args()
    qret = check_executable(args.quration_bin, "qret")
    workloads = load_workloads(args.workload)
    targets = ["fig17", "fig19"] if args.target == "both" else [args.target]
    if args.target == "fig19":
        target_workloads(workloads, "fig19")

    env = os.environ.copy()
    env["PATH"] = str(qret.parent) + os.pathsep + env.get("PATH", "")

    for target in targets:
        selected = target_workloads(workloads, target, require_nonempty=False)
        if not selected:
            print(f"[skip] no selected workload is used by {target}", flush=True)
            continue
        config = FIG17_CONFIG if target == "fig17" else FIG19_CONFIG
        raw_root = args.output_dir.resolve() / target / "quration_eval_result"
        for workload in selected:
            run_name = (
                workload.ham_name
                if target == "fig17"
                else f"{workload.ham_name}_ideal"
            )
            run_profile(
                qret,
                find_qasm(workload.ham_name),
                raw_root,
                run_name,
                config,
                env,
                args.force,
                args.dry_run,
            )

    if args.dry_run or args.skip_analysis:
        return
    from analyze import analyze_outputs

    analyze_outputs(
        input_dir=args.output_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        targets=targets,
        workloads=workloads,
        num_shots=args.num_shots,
        allow_partial=False,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
