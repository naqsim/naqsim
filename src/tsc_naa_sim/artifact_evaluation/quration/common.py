"""Shared workload selection for the standalone Quration prerequisite."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


QURATION_DIR = Path(__file__).resolve().parent
AE_DIR = QURATION_DIR.parent
TSC_DIR = AE_DIR.parent
REPO_DIR = TSC_DIR.parents[1]
WORKLOAD_CSV = AE_DIR / "config" / "workloads.csv"


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    ham_name: str
    ls_distance: int
    fig19: bool


def load_workloads(selectors: list[str]) -> list[Workload]:
    workloads = []
    with WORKLOAD_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            candidates = {row["key"], row["label"], row["ham_name"]}
            if selectors and not any(selector in candidates for selector in selectors):
                continue
            workloads.append(
                Workload(
                    key=row["key"],
                    label=row["label"],
                    ham_name=row["ham_name"],
                    ls_distance=int(row["ls_distance"]),
                    fig19=row["fig19"] == "1",
                )
            )
    if selectors:
        matched = {
            value
            for workload in workloads
            for value in (workload.key, workload.label, workload.ham_name)
        }
        missing = [selector for selector in selectors if selector not in matched]
        if missing:
            raise ValueError(f"Unknown workload selector(s): {missing}")
    if not workloads:
        raise ValueError("No workloads selected")
    return workloads


def target_workloads(
    workloads: list[Workload], target: str, *, require_nonempty: bool = True
) -> list[Workload]:
    selected = workloads if target == "fig17" else [item for item in workloads if item.fig19]
    order = {
        key: index
        for index, key in enumerate(
            ["fh", "jellium", "h4", "hei", "adder", "ising1d", "ising2d", "qft"]
        )
    }
    selected = sorted(selected, key=lambda item: order[item.key])
    if require_nonempty and not selected:
        raise ValueError(f"No selected workload is used by {target}")
    return selected


def find_qasm(ham_name: str) -> Path:
    qasm_root = REPO_DIR / "src" / "benchmarks" / "qasm"
    direct = qasm_root / f"{ham_name}.qasm"
    if direct.is_file():
        return direct
    matches = sorted(qasm_root.rglob(f"{ham_name}.qasm"))
    if not matches:
        raise FileNotFoundError(f"Could not find {ham_name}.qasm under {qasm_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous QASM path for {ham_name}: {matches}")
    return matches[0]
