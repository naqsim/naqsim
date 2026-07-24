#!/usr/bin/env python3
"""Convert standalone Quration JSON outputs to the shared paper CSV formats."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from common import QURATION_DIR, TSC_DIR, Workload, load_workloads, target_workloads

sys.path.insert(0, str(TSC_DIR))
from count_occupied_cells import parse_ml_json  # noqa: E402


CODE_CYCLE_TIME_SEC = 0.000237


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate current-format Fig. 17/19 CSVs from Quration JSON."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=QURATION_DIR / "results"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to --input-dir.",
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
        "--num-shots", type=int, default=10,
        help="Monte Carlo shots for the Fig. 19 trace sweep.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Skip workloads whose expected ML/profile JSON is unavailable.",
    )
    return parser.parse_args()


def find_named(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename)) if root.is_dir() else []
    return matches[0] if matches else None


def require_named(root: Path, filename: str) -> Path:
    path = find_named(root, filename)
    if path is None:
        raise FileNotFoundError(f"Could not find {filename} below {root}")
    return path


def available_pairs(
    input_dir: Path,
    workloads: list[Workload],
    target: str,
    allow_partial: bool,
) -> list[tuple[Workload, Path, Path]]:
    pairs = []
    target_root = input_dir / target
    search_root = target_root if target_root.is_dir() else input_dir
    for workload in target_workloads(workloads, target, require_nonempty=False):
        run_name = workload.ham_name if target == "fig17" else f"{workload.ham_name}_ideal"
        try:
            ml_path = require_named(search_root, f"{run_name}_ml.json")
            profile_path = require_named(search_root, f"{run_name}_profile.json")
        except FileNotFoundError as exc:
            if not allow_partial:
                raise
            print(f"warning: skipping {workload.label}: {exc}", file=sys.stderr)
            continue
        pairs.append((workload, ml_path, profile_path))
    if not pairs:
        raise FileNotFoundError(f"No Quration JSON was available for {target}")
    return pairs


def analyze_fig17(
    input_dir: Path,
    output_dir: Path,
    workloads: list[Workload],
    allow_partial: bool,
) -> Path:
    rows = []
    for workload, ml_path, _ in available_pairs(
        input_dir, workloads, "fig17", allow_partial
    ):
        result = parse_ml_json(str(ml_path))
        if result is None:
            raise ValueError(f"No instructions found in {ml_path}")
        result["ham_name"] = workload.ham_name
        rows.append(result)
    target_dir = output_dir / "fig17"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "quration_eval_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote Fig. 17 Quration CSV: {output_path.resolve()}")
    return output_path


@dataclass(frozen=True)
class MagicOverheadConfig:
    code_cycle_time_sec: float
    code_distance: int
    init_plus_cycles: float = 1.0
    move_cx_cycles: float = 1.0
    move_5_cx_cycles: float = 1.0


class QurationResourceFactory:
    """MSC/MSD model used by quration_trace_ms_overhead.ipy."""

    def __init__(self, config: MagicOverheadConfig):
        self.code_dist = config.code_distance
        self.esm_sec = config.code_cycle_time_sec
        self.pq_per_patch = 2 * (self.code_dist**2) - 1
        self.logical_beat_sec = self.code_dist * config.code_cycle_time_sec
        self.init_plus_sec = config.init_plus_cycles * self.esm_sec
        self.move_cx_sec = config.move_cx_cycles * self.esm_sec
        self.move_5_cx_sec = config.move_5_cx_cycles * self.esm_sec

    def msd_cost(self) -> tuple[float, int]:
        per_msd_pq = 11 * self.pq_per_patch
        per_msd_sec = 11 * self.code_dist * self.esm_sec
        return per_msd_sec, per_msd_pq

    def msc_cost(self) -> tuple[float, int]:
        d3_color_pq = 13
        d3_color_sec = 4 * self.esm_sec
        d3_color_pacc = 0.822
        d5_surf_sec = 3 * self.esm_sec
        d5_surf_pacc = 0.77
        d11_surf_pq = 2 * (11**2) - 1
        d11_surf_sec = 15 * self.esm_sec
        d11_surf_pacc = 0.568
        expansion_sec = max(self.code_dist - 11, 0) * self.esm_sec
        per_msc_pq = (4 * d3_color_pq) + d11_surf_pq
        per_msc_sec = (
            (d3_color_sec + d5_surf_sec)
            / (1 - (1 - d3_color_pacc * d5_surf_pacc) ** 4)
            + (d11_surf_sec / d11_surf_pacc)
            + expansion_sec
        )
        return per_msc_sec, per_msc_pq

    def sample_msc_sec(self, rng: random.Random) -> float:
        d3_color_sec = 4 * self.esm_sec
        d3_color_pacc = 0.822
        d5_surf_sec = 3 * self.esm_sec
        d5_surf_pacc = 0.77
        d11_surf_sec = 15 * self.esm_sec
        d11_surf_pacc = 0.568
        expansion_sec = max(self.code_dist - 11, 0) * self.esm_sec
        accepted = False
        elapsed = 0.0
        while not accepted:
            phase_12_end = []
            for _ in range(4):
                latency = 0.0
                phase_2_accepted = False
                while not phase_2_accepted:
                    phase_1_accepted = False
                    while not phase_1_accepted:
                        latency += d3_color_sec
                        phase_1_accepted = rng.random() < d3_color_pacc
                    latency += d5_surf_sec
                    phase_2_accepted = rng.random() < d5_surf_pacc
                phase_12_end.append(latency)
            elapsed += min(phase_12_end)
            elapsed += d11_surf_sec
            accepted = rng.random() < d11_surf_pacc
        return elapsed + expansion_sec

    def estimate_factory_requirement(self, profile: dict) -> tuple[int, int]:
        values = [int(value) for value in profile.get("magic_state_consumption_rate", [])]
        runtime = int(profile["runtime"])
        if len(values) < runtime:
            values += [0] * (runtime - len(values))
        else:
            values = values[:runtime]
        if sum(values) == 0:
            return 0, 0
        last_index = next(
            index for index in range(len(values) - 1, -1, -1) if values[index] != 0
        )
        time_to_last = max(
            last_index * self.logical_beat_sec, self.logical_beat_sec
        )
        throughput = sum(values) / time_to_last
        per_msd_sec, _ = self.msd_cost()
        per_msc_sec, _ = self.msc_cost()
        est_msd = max(round(throughput * per_msd_sec), 1)
        est_msc = max(int(round(throughput * 15 * per_msc_sec) * 1.1), 1)
        return est_msd, est_msc

    @staticmethod
    def num_factories(
        est_msd: int, est_msc: int, msd_factor: float, msc_factor: float
    ) -> tuple[int, int]:
        num_msd = 0 if est_msd == 0 else max(round(est_msd * msd_factor), 1)
        num_msc = 0 if est_msc == 0 else max(round(est_msc * msc_factor), 1)
        return num_msd, num_msc

    def original_pq(self, profile: dict) -> int:
        return int(profile["chip_cell_count"]) * self.pq_per_patch

    def factory_pq(self, num_msd: int, num_msc: int) -> tuple[int, int]:
        _, per_msd_pq = self.msd_cost()
        _, per_msc_pq = self.msc_cost()
        return per_msd_pq * num_msd, per_msc_pq * num_msc


def walk_json_objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json_objects(child)


def load_magic_request_beats(ml_path: Path, profile: dict) -> tuple[list[int], str]:
    with ml_path.open() as handle:
        data = json.load(handle)
    beats = []
    for item in walk_json_objects(data):
        if item.get("type") != "LATTICE_SURGERY_MAGIC":
            continue
        metadata = item.get("metadata", {})
        if "beat" not in metadata:
            raise ValueError(f"LATTICE_SURGERY_MAGIC without metadata.beat in {ml_path}")
        beats.append(int(metadata["beat"]))
    if beats:
        beats.sort()
        expected = int(profile.get("magic_state_consumption_count", len(beats)))
        if len(beats) != expected:
            raise ValueError(
                f"Magic request count mismatch for {ml_path}: ML={len(beats)}, "
                f"profile={expected}"
            )
        return beats, "ml_lattice_surgery_magic"
    for beat, count in enumerate(profile.get("magic_state_consumption_rate", [])):
        beats.extend([beat] * int(count))
    return beats, "profile_magic_state_consumption_rate"


class TraceResourceStateSimulator:
    def __init__(
        self, profile: dict, request_beats: list[int], factory: QurationResourceFactory
    ):
        self.profile = profile
        self.request_beats = sorted(int(beat) for beat in request_beats)
        self.factory = factory
        self.ideal_runtime_sec = int(profile["runtime"]) * factory.logical_beat_sec

    def consume_msc_batch(self, heap: list[float], rng: random.Random) -> float:
        ready = 0.0
        for _ in range(15):
            generated = heapq.heappop(heap)
            ready = max(ready, generated)
            heapq.heappush(heap, generated + self.factory.sample_msc_sec(rng))
        return ready

    def shot(self, num_msd: int, num_msc: int, seed: str) -> float:
        if not self.request_beats:
            return 0.0
        if num_msd <= 0 or num_msc <= 0:
            raise ValueError("Nonzero magic requests require MSD and MSC factories")
        rng = random.Random(seed)
        per_msd_sec, _ = self.factory.msd_cost()
        msc_heap = [self.factory.sample_msc_sec(rng) for _ in range(num_msc)]
        heapq.heapify(msc_heap)
        msd_heap = [0.0] * num_msd
        heapq.heapify(msd_heap)
        total_stall = 0.0
        for request_beat in self.request_beats:
            request_time = request_beat * self.factory.logical_beat_sec + total_stall
            msc_ready = self.consume_msc_batch(msc_heap, rng)
            msd_ready = heapq.heappop(msd_heap)
            finish = max(msc_ready, msd_ready) + per_msd_sec
            heapq.heappush(msd_heap, finish)
            if finish > request_time:
                total_stall += finish - request_time
        return total_stall


def trace_sweep_for_workload(
    workload: Workload,
    ml_path: Path,
    profile_path: Path,
    scale_factors: np.ndarray,
    shots: int,
) -> list[dict]:
    with profile_path.open() as handle:
        profile = json.load(handle)
    request_beats, trace_source = load_magic_request_beats(ml_path, profile)
    factory = QurationResourceFactory(
        MagicOverheadConfig(CODE_CYCLE_TIME_SEC, workload.ls_distance)
    )
    simulator = TraceResourceStateSimulator(profile, request_beats, factory)
    est_msd, est_msc = factory.estimate_factory_requirement(profile)
    original_pq = factory.original_pq(profile)
    ideal_runtime = profile["runtime"]
    runtime_cache = {}
    rows = []
    for msd_factor, msc_factor in product(scale_factors, scale_factors):
        num_msd, num_msc = factory.num_factories(
            est_msd, est_msc, msd_factor, msc_factor
        )
        cache_key = (num_msd, num_msc, shots)
        if cache_key not in runtime_cache:
            stall_samples = []
            stall_beat_samples = []
            runtime_samples = []
            for shot_index in range(shots):
                seed = (
                    f"{workload.ham_name}:{workload.ls_distance}:"
                    f"{num_msd}:{num_msc}:{shot_index}"
                )
                stall = simulator.shot(num_msd, num_msc, seed)
                stall_beat = stall / factory.logical_beat_sec
                stall_samples.append(stall)
                stall_beat_samples.append(stall_beat)
                runtime_samples.append(ideal_runtime + stall_beat)
            runtime_cache[cache_key] = (
                stall_samples, stall_beat_samples, runtime_samples
            )
        stall_samples, stall_beat_samples, runtime_samples = runtime_cache[cache_key]
        mean_runtime = float(np.mean(runtime_samples))
        std_runtime = float(np.std(runtime_samples))
        mean_stall = float(np.mean(stall_samples))
        std_stall = float(np.std(stall_samples))
        mean_stall_beat = float(np.mean(stall_beat_samples))
        std_stall_beat = float(np.std(stall_beat_samples))
        time_cost = float(round(mean_runtime / ideal_runtime, 2))
        time_cost_std = float(std_runtime / ideal_runtime)
        msd_pq, msc_pq = factory.factory_pq(num_msd, num_msc)
        space_cost = float(round((original_pq + msd_pq + msc_pq) / original_pq, 2))
        spacetime_cost = float(round(space_cost * time_cost, 2))
        rows.append(
            {
                "ham_name": workload.ham_name,
                "est_num_msd": est_msd,
                "est_num_msc": est_msc,
                "num_msd_sf": msd_factor,
                "num_msc_sf": msc_factor,
                "num_msd": num_msd,
                "num_msc": num_msc,
                "magic_period": None,
                "shots": len(runtime_samples),
                "quration_magic_factory_count": profile.get("magic_factory_count"),
                "request_trace_source": trace_source,
                "request_count_from_trace": len(request_beats),
                "ideal_runtime": ideal_runtime,
                "ideal_runtime_sec_for_manual_code_distance": simulator.ideal_runtime_sec,
                "runtime": mean_runtime,
                "runtime_samples": runtime_samples,
                "std_runtime": std_runtime,
                "mean_stall_sec": mean_stall,
                "std_stall_sec": std_stall,
                "mean_stall_beat": mean_stall_beat,
                "std_stall_beat": std_stall_beat,
                "stall_beat_samples": stall_beat_samples,
                "original_pq": original_pq,
                "msd_pq": msd_pq,
                "msc_pq": msc_pq,
                "time_cost": time_cost,
                "time_cost_std": time_cost_std,
                "space_cost": space_cost,
                "spacetime_cost": spacetime_cost,
                "code_distance": workload.ls_distance,
                "quration_code_distance": profile.get("code_distance"),
                "quration_code_distance_samples": [profile.get("code_distance")],
                "ideal_quration_code_distance": profile.get("code_distance"),
                "magic_state_consumption_count": profile[
                    "magic_state_consumption_count"
                ],
                "magic_state_consumption_depth": profile[
                    "magic_state_consumption_depth"
                ],
            }
        )
    return rows


def geomean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    if not len(array) or np.any(array <= 0):
        return math.nan
    return float(np.exp(np.log(array).mean()))


def analyze_fig19(
    input_dir: Path,
    output_dir: Path,
    workloads: list[Workload],
    num_shots: int,
    allow_partial: bool,
) -> tuple[Path, Path]:
    pairs = available_pairs(input_dir, workloads, "fig19", allow_partial)
    eval_rows = []
    sweep_rows = []
    scale_factors = np.round(np.arange(0.5, 2.1, 0.1), 1)
    for workload, ml_path, profile_path in pairs:
        result = parse_ml_json(str(ml_path))
        if result is None:
            raise ValueError(f"No instructions found in {ml_path}")
        result.update(
            {
                "ham_name": workload.ham_name,
                "run_name": f"{workload.ham_name}_ideal",
                "code_distance_for_space": workload.ls_distance,
            }
        )
        eval_rows.append(result)
        sweep_rows.extend(
            trace_sweep_for_workload(
                workload, ml_path, profile_path, scale_factors, max(num_shots, 1)
            )
        )
    target_dir = output_dir / "fig19"
    target_dir.mkdir(parents=True, exist_ok=True)
    eval_path = target_dir / "trace_ms_quration_eval_summary.csv"
    sweep_path = target_dir / "quration_trace_ms_overhead_sweep.csv"
    summary_path = target_dir / "quration_trace_ms_overhead_summary.csv"
    pd.DataFrame(eval_rows).to_csv(eval_path, index=False)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(sweep_path, index=False)
    summary_rows = []
    for (msd_factor, msc_factor), group in sweep.groupby(
        ["num_msd_sf", "num_msc_sf"]
    ):
        normalized_time = group["runtime"] / group["ideal_runtime"]
        normalized_space = (
            group["original_pq"] + group["msd_pq"] + group["msc_pq"]
        ) / group["original_pq"]
        summary_rows.append(
            {
                "rs_sf": (float(msd_factor), float(msc_factor)),
                "time": geomean(normalized_time),
                "space": geomean(normalized_space),
                "spacetime": geomean(normalized_time * normalized_space),
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote Fig. 19 Quration sweep: {sweep_path.resolve()}")
    print(f"Wrote Fig. 19 Quration summary: {summary_path.resolve()}")
    return sweep_path, summary_path


def analyze_outputs(
    input_dir: Path,
    output_dir: Path,
    targets: list[str],
    workloads: list[Workload],
    num_shots: int,
    allow_partial: bool,
) -> None:
    for target in targets:
        selected = target_workloads(workloads, target, require_nonempty=False)
        if not selected:
            print(f"[skip] no selected workload is used by {target}")
            continue
        if target == "fig17":
            analyze_fig17(input_dir, output_dir, workloads, allow_partial)
        else:
            analyze_fig19(
                input_dir, output_dir, workloads, num_shots, allow_partial
            )


def main() -> None:
    args = parse_args()
    workloads = load_workloads(args.workload)
    targets = ["fig17", "fig19"] if args.target == "both" else [args.target]
    if args.target == "fig19":
        target_workloads(workloads, "fig19")
    output_dir = args.output_dir or args.input_dir
    analyze_outputs(
        input_dir=args.input_dir.resolve(),
        output_dir=output_dir.resolve(),
        targets=targets,
        workloads=workloads,
        num_shots=args.num_shots,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
