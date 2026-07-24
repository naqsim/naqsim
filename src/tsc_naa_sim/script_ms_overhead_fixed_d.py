from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
os.chdir(SCRIPT_DIR)
for subpath in SRC_DIR.iterdir():
    if subpath.is_dir():
        sys.path.append(str(subpath))

from latency_functions import latency_functions
from precomputed_mapping_io import DEFAULT_PRECOMPUTED_MAPPING_DIR
from run_analysis_scripts import dump_zst, load_zst
from tsc_instructions import InstType, UopType


DEFAULT_CFG_NAME = "DIR_TOGL+DEDICATE_CELL2"
SCALE_FACTORS = tuple(round(float(x), 1) for x in np.arange(0.5, 2.1, 0.1))
REQUIRED_RESULT_FILES = [
    "cfg_in.zst",
    "comp_out.zst",
    "exec_out_IGNORE_NONE.zst",
]
_RSF_WORKER_QC_NAME = None
_RSF_WORKER_SIM = None
_RSF_WORKER_RESULT_DIR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ms_overhead_revision.ipynb analysis using execution-stage "
            "outputs from script_d3rot_opt_evaluation_fixed_d.py."
        )
    )
    parser.add_argument(
        "--ham-name",
        type=str,
        default="SELECT_10_Heisenberg1D_OBC_1_0",
        help="Target hamiltonian name used by script_d3rot_opt_evaluation_fixed_d.py.",
    )
    parser.add_argument(
        "--use-naive-mapping",
        action="store_true",
        help="Read outputs generated with --use-naive-mapping.",
    )
    parser.add_argument(
        "--use-precomputed-mapping",
        action="store_true",
        help="Read outputs generated with --use-precomputed-mapping.",
    )
    parser.add_argument(
        "--precomputed-mapping-dir",
        type=str,
        default=str(DEFAULT_PRECOMPUTED_MAPPING_DIR),
        help=(
            "Accepted for CLI compatibility with script_d3rot_opt_evaluation_fixed_d.py. "
            "This analysis reads saved outputs and does not load mappings directly."
        ),
    )
    parser.add_argument(
        "--code-distance",
        type=int,
        default=25,
        help="Code distance used by script_d3rot_opt_evaluation_fixed_d.py.",
    )
    parser.add_argument(
        "--num-threads",
        "--num_threads",
        dest="num_threads",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--cfg-name",
        type=str,
        default=DEFAULT_CFG_NAME,
        help="d3rot_opt configuration directory to analyze.",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=10,
        help="Number of Monte Carlo shots per scale-factor point.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for ms-overhead artifacts.",
    )
    args = parser.parse_args()
    if args.use_naive_mapping and args.use_precomputed_mapping:
        parser.error("--use-naive-mapping and --use-precomputed-mapping are mutually exclusive.")
    return args


def mapping_prefix(use_naive_mapping: bool, use_precomputed_mapping: bool) -> str:
    if use_naive_mapping:
        return "naive_mapping_"
    if use_precomputed_mapping:
        return "precomputed_mapping_"
    return ""


def default_d3rot_base_dir(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
    use_precomputed_mapping: bool,
) -> Path:
    return (
        SCRIPT_DIR
        / "output"
        / "d3rot_opt_evaluation_fixed_d"
        / f"{mapping_prefix(use_naive_mapping, use_precomputed_mapping)}{ham_name}_distance{code_dist}"
    )


def default_output_dir(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
    use_precomputed_mapping: bool,
) -> Path:
    return (
        SCRIPT_DIR
        / "output"
        / "ms_overhead_fixed_d"
        / f"{mapping_prefix(use_naive_mapping, use_precomputed_mapping)}{ham_name}_distance{code_dist}"
    )


def append_extension(output_prefix: Path, extension: str) -> Path:
    return output_prefix.with_name(output_prefix.name + extension)


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def validate_result_files(outdir: Path) -> None:
    missing = [outdir / filename for filename in REQUIRED_RESULT_FILES if not (outdir / filename).is_file()]
    if missing:
        lines = [
            "Required d3rot fixed-d outputs are missing.",
            "Run script_d3rot_opt_evaluation_fixed_d.py first with the same arguments.",
            "",
        ]
        lines.extend(f"- {path}" for path in missing)
        raise FileNotFoundError("\n".join(lines))


class resource_factory:
    def __init__(self, cfg_in, plane_char):
        code_dist = cfg_in.code_dist
        self.code_dist = code_dist
        assert cfg_in.rounds == 1

        lftn = latency_functions(
            cfg_in.hw_cfg,
            cfg_in.cx_type,
            cfg_in.meas_type,
            cfg_in.move_type,
        )
        self.lftn = lftn

        rst_us = lftn.rst_us
        h_us = lftn.sq_us
        cz_us = lftn.tq_us * 4
        ndm_us = lftn.ndm_us

        esm_us = 0
        esm_us += rst_us
        for _ in range(4):
            esm_us += h_us
            esm_us += cz_us
        esm_us += h_us
        esm_us += ndm_us
        self.esm_us = esm_us

        pick_us = drop_us = lftn.trf_us
        one_move_um = 1 * (code_dist + 1) * lftn.L_um
        one_move_us = pick_us + lftn.shuttle_us(one_move_um) + drop_us
        five_move_um = 5 * one_move_um
        five_move_us = pick_us + lftn.shuttle_us(five_move_um) + drop_us

        dir_rot_us = 0
        dir_rot_us += pick_us
        dir_rot_us += lftn.slm_onoff_us
        rot_um = (math.pi * math.sqrt(2) / 4) * (code_dist * lftn.L_um)
        dir_rot_us += lftn.shuttle_us(rot_um)
        dir_rot_us += lftn.slm_onoff_us
        dir_rot_us += drop_us

        self.init_plus_us = h_us + esm_us
        self.move_cx_us = one_move_us + (h_us + cz_us + h_us) + esm_us
        self.move_rot_us = (one_move_us - drop_us) + dir_rot_us + esm_us
        self.move_5_cx_us = five_move_us + (h_us + cz_us + h_us) + esm_us

        self.pq_per_patch = 2 * (code_dist**2) - 1
        self.fp_per_patch = ((code_dist + 1) * lftn.L_um) ** 2

    def ysd_cost(self):
        num_patch = 2
        per_ysd_pq = num_patch * self.pq_per_patch
        per_ysd_fp = 1 * self.fp_per_patch + 1 * (2 * self.fp_per_patch)

        per_ysd_us = 0
        per_ysd_us += self.init_plus_us
        per_ysd_us += self.move_cx_us
        per_ysd_us += self.move_rot_us
        per_ysd_us += self.move_cx_us
        return per_ysd_us, per_ysd_pq, per_ysd_fp

    def msd_cost(self):
        num_patch = 7
        per_msd_pq = num_patch * self.pq_per_patch
        per_msd_fp = num_patch * self.fp_per_patch

        per_msd_us = 0
        per_msd_us += self.init_plus_us
        per_msd_us += 5 * self.move_cx_us
        per_msd_us += 11 * self.move_cx_us
        per_msd_us += self.move_5_cx_us
        return per_msd_us, per_msd_pq, per_msd_fp

    def msc_cost(self):
        d3_color_pq = 13
        d3_color_us = 4 * self.esm_us
        d3_color_pacc = 0.822

        d5_surf_us = 3 * self.esm_us
        d5_surf_pacc = 0.77

        d11_surf_pq = 2 * (11**2) - 1
        d11_surf_us = 15 * self.esm_us
        d11_surf_pacc = 0.568
        d11_surf_fp = ((11 + 1) * self.lftn.L_um) ** 2

        expansion_us = (self.code_dist - 11) * self.esm_us

        per_msc_pq = (4 * d3_color_pq) + d11_surf_pq
        per_msc_fp = d11_surf_fp

        per_msc_us = 0
        per_msc_us += (d3_color_us + d5_surf_us) / (1 - (1 - d3_color_pacc * d5_surf_pacc) ** 4)
        per_msc_us += d11_surf_us / d11_surf_pacc
        per_msc_us += expansion_us
        return per_msc_us, per_msc_pq, per_msc_fp

    def sample_msc_us(self):
        d3_color_us = 4 * self.esm_us
        d3_color_pacc = 0.822

        d5_surf_us = 3 * self.esm_us
        d5_surf_pacc = 0.77

        d11_surf_us = 15 * self.esm_us
        d11_surf_pacc = 0.568

        expansion_us = (self.code_dist - 11) * self.esm_us

        phase_3_acc = False
        msc_sampling_us = 0
        while not phase_3_acc:
            phase_12_end = []
            for _ in range(4):
                latency = 0
                phase_2_acc = False
                while not phase_2_acc:
                    phase_1_acc = False
                    while not phase_1_acc:
                        latency += d3_color_us
                        phase_1_acc = bool(random.random() < d3_color_pacc)
                    latency += d5_surf_us
                    phase_2_acc = bool(random.random() < d5_surf_pacc)
                phase_12_end.append(latency)

            msc_sampling_us += min(phase_12_end)
            msc_sampling_us += d11_surf_us
            phase_3_acc = bool(random.random() < d11_surf_pacc)

        msc_sampling_us += expansion_us
        return msc_sampling_us


class rsf_trace_sim:
    def __init__(self, uop_schedule_trace, req_schedule_trace, plane_char, cfg_in):
        self.rsf = resource_factory(cfg_in, plane_char)
        self.uop_schedule_trace = uop_schedule_trace
        self.req_schedule_trace = req_schedule_trace
        self.num_patch = len(plane_char) * len(plane_char[0])

    def estimate_rsf_requirement(self):
        def get_finish_us(uop_schedule):
            finish_us = 0
            for intervals in uop_schedule.values():
                for uop, _, (_, end) in intervals:
                    if uop.uop_type == UopType.ESM:
                        finish_us = max(finish_us, end)
            return finish_us

        latency_list = [get_finish_us(uop_schedule) for uop_schedule in self.uop_schedule_trace]

        num_m_list = []
        num_y_list = []
        for req_schedule in self.req_schedule_trace:
            num_m = 0
            num_y = 0
            for req in req_schedule:
                if req.inst_type == InstType.REQ_MY:
                    num_m += 1
                    num_y += 1
                elif req.inst_type == InstType.REQ_Y:
                    num_y += 1
                else:
                    raise Exception(f"Unsupported request instruction type: {req.inst_type}")
            num_m_list.append(num_m)
            num_y_list.append(num_y)

        self.latency_list = latency_list
        self.num_m_list = num_m_list
        self.num_y_list = num_y_list

        total_latency = sum(latency_list)
        total_m = sum(num_m_list)
        total_y = sum(num_y_list)
        avg_m_thrpt = 0
        avg_y_thrpt = 0

        if total_m:
            last_m_idx = next(i for i in range(len(num_m_list) - 1, -1, -1) if num_m_list[i] != 0)
            time_to_last_m = sum(latency_list[:last_m_idx])
            avg_m_thrpt = total_m / (time_to_last_m if time_to_last_m > 0 else total_latency)
        if total_y:
            last_y_idx = next(i for i in range(len(num_y_list) - 1, -1, -1) if num_y_list[i] != 0)
            time_to_last_y = sum(latency_list[:last_y_idx])
            avg_y_thrpt = total_y / (time_to_last_y if time_to_last_y > 0 else total_latency)

        per_msd_us, _, _ = self.rsf.msd_cost()
        per_msc_us, _, _ = self.rsf.msc_cost()
        per_ysd_us, _, _ = self.rsf.ysd_cost()

        if total_m == 0:
            self.est_num_msd = 0
            self.est_num_msc = 0
        else:
            self.est_num_msd = max(round(avg_m_thrpt * per_msd_us), 1)
            self.est_num_msc = max(int(round(avg_m_thrpt * 15 * per_msc_us) * 1.1), 1)
        if total_m == 0 and total_y == 0:
            self.est_num_ysd = 0
        else:
            self.est_num_ysd = max(round((avg_y_thrpt + 15 * avg_m_thrpt) * per_ysd_us), 1)

    def shot(self, num_msd, num_msc, num_ysd):
        per_msd_us, _, _ = self.rsf.msd_cost()
        per_ysd_us, _, _ = self.rsf.ysd_cost()

        sim_time = 0
        msc_generated_list = [self.rsf.sample_msc_us() for _ in range(num_msc)]
        msc_time = 0
        ysd_time = 0
        msd_time_list = [0] * num_msd
        ysd_stock = 0
        msd_stall = 0
        ysd_stall = 0

        for idx, (latency, num_m, num_y) in enumerate(zip(self.latency_list, self.num_m_list, self.num_y_list)):
            if idx == 0:
                sim_time += latency
                continue

            if num_m == 0 and num_y == 0:
                sim_time += latency
                continue

            if num_m != 0:
                last_msd_end = 0
                for _ in range(num_m):
                    for _ in range(15):
                        msc_generated = min(msc_generated_list)
                        msc_idx = msc_generated_list.index(msc_generated)
                        msc_time = max(msc_time, msc_generated)
                        msc_generated_list[msc_idx] = msc_generated + self.rsf.sample_msc_us()

                    while ysd_stock < 15:
                        ysd_stock += num_ysd
                        ysd_time += per_ysd_us
                    ysd_stock -= 15

                    msd_time = min(msd_time_list)
                    msd_idx = msd_time_list.index(msd_time)
                    msd_start = max(ysd_time, msc_time, msd_time)
                    msd_end = msd_start + per_msd_us
                    msd_time_list[msd_idx] = msd_end
                    last_msd_end = max(msd_end, last_msd_end)

                if last_msd_end > sim_time:
                    msd_stall += last_msd_end - sim_time
                    sim_time = last_msd_end

            if num_y != 0:
                while ysd_stock < num_y:
                    ysd_stock += num_ysd
                    ysd_time += per_ysd_us
                ysd_stock -= num_y

                if ysd_time > sim_time:
                    ysd_stall += ysd_time - sim_time
                    sim_time = ysd_time

            sim_time += latency

        orig_latency = sum(self.latency_list)
        stall = msd_stall + ysd_stall
        print("sim_time: ", sim_time, "stall+orig: ", stall + orig_latency)
        assert abs(sim_time - (stall + orig_latency)) < 1e-4
        return stall

    def run(self, num_shots=100, num_msd_sf=1, num_msc_sf=1, num_ysd_sf=1):
        if not hasattr(self, "est_num_msd"):
            self.estimate_rsf_requirement()

        num_msd = 0 if self.est_num_msd == 0 else max(round(self.est_num_msd * num_msd_sf), 1)
        num_msc = 0 if self.est_num_msc == 0 else max(round(self.est_num_msc * num_msc_sf), 1)
        num_ysd = 0 if self.est_num_ysd == 0 else max(round(self.est_num_ysd * num_ysd_sf), 1)

        self.num_msd = num_msd
        self.num_msc = num_msc
        self.num_ysd = num_ysd

        if num_msd == 0 and num_msc == 0 and num_ysd == 0:
            stall_samples = [0]
        else:
            stall_samples = [
                self.shot(num_msd, num_msc, num_ysd)
                for _ in range(num_shots)
            ]

        original_latency = sum(self.latency_list)
        mean_stall = np.mean(stall_samples)
        std_stall = np.std(stall_samples)
        time_cost = round((original_latency + mean_stall) / original_latency, 2)

        _, per_msd_pq, _ = self.rsf.msd_cost()
        _, per_msc_pq, _ = self.rsf.msc_cost()
        _, per_ysd_pq, _ = self.rsf.ysd_cost()

        original_pq = self.rsf.pq_per_patch * self.num_patch
        msd_pq = per_msd_pq * num_msd
        msc_pq = per_msc_pq * num_msc
        ysd_pq = per_ysd_pq * num_ysd
        space_cost = round((original_pq + msd_pq + msc_pq + ysd_pq) / original_pq, 2)
        spacetime_cost = round(space_cost * time_cost, 2)

        self.original_latency = original_latency
        self.mean_stall = mean_stall
        self.std_stall = std_stall
        self.original_pq = original_pq
        self.msd_pq = msd_pq
        self.msc_pq = msc_pq
        self.ysd_pq = ysd_pq
        self.time_cost = time_cost
        self.space_cost = space_cost
        self.spacetime_cost = spacetime_cost


def result_prefix(result_dir: Path, qc_name: str, num_msd_sf: float, num_msc_sf: float, num_ysd_sf: float) -> Path:
    return result_dir / f"{qc_name}_{num_msd_sf:.1f}_{num_msc_sf:.1f}_{num_ysd_sf:.1f}"


def load_rsf_trace_sim(outdir: Path) -> rsf_trace_sim:
    cfg_in = load_zst(str(outdir / "cfg_in"))
    uop_schedule_trace, _, _ = load_zst(str(outdir / "exec_out_IGNORE_NONE"))
    _, plane_char, _, req_schedule_trace, _ = load_zst(str(outdir / "comp_out"))

    sim = rsf_trace_sim(
        uop_schedule_trace,
        req_schedule_trace,
        plane_char,
        cfg_in,
    )
    sim.estimate_rsf_requirement()
    return sim


def run_rsf_point(
    qc_name: str,
    sim: rsf_trace_sim,
    num_msd_sf: float,
    num_msc_sf: float,
    num_ysd_sf: float,
    num_shots: int,
    result_dir: Path,
) -> dict:
    sim.run(
        num_shots=num_shots,
        num_msd_sf=num_msd_sf,
        num_msc_sf=num_msc_sf,
        num_ysd_sf=num_ysd_sf,
    )

    qc_cfg_res = {
        "qc_name": qc_name,
        "est_num_msd": sim.est_num_msd,
        "est_num_msc": sim.est_num_msc,
        "est_num_ysd": sim.est_num_ysd,
        "num_msd_sf": num_msd_sf,
        "num_msc_sf": num_msc_sf,
        "num_ysd_sf": num_ysd_sf,
        "num_msd": sim.num_msd,
        "num_msc": sim.num_msc,
        "num_ysd": sim.num_ysd,
        "orig_latency": sim.original_latency,
        "mean_stall": sim.mean_stall,
        "std_stall": sim.std_stall,
        "original_pq": sim.original_pq,
        "msd_pq": sim.msd_pq,
        "msc_pq": sim.msc_pq,
        "ysd_pq": sim.ysd_pq,
        "time_cost": sim.time_cost,
        "space_cost": sim.space_cost,
        "spacetime_cost": sim.spacetime_cost,
    }
    dump_zst(qc_cfg_res, str(result_prefix(result_dir, qc_name, num_msd_sf, num_msc_sf, num_ysd_sf)))
    return qc_cfg_res


def init_rsf_worker(qc_name: str, outdir: Path, result_dir: Path) -> None:
    global _RSF_WORKER_QC_NAME, _RSF_WORKER_SIM, _RSF_WORKER_RESULT_DIR
    random.seed(os.getpid())
    _RSF_WORKER_QC_NAME = qc_name
    _RSF_WORKER_SIM = load_rsf_trace_sim(outdir)
    _RSF_WORKER_RESULT_DIR = result_dir


def analyze_qc_cfg_rsf_from_worker(args) -> dict:
    if _RSF_WORKER_QC_NAME is None or _RSF_WORKER_SIM is None or _RSF_WORKER_RESULT_DIR is None:
        raise RuntimeError("RSF worker was not initialized.")
    num_msd_sf, num_msc_sf, num_ysd_sf, num_shots = args
    return run_rsf_point(
        _RSF_WORKER_QC_NAME,
        _RSF_WORKER_SIM,
        num_msd_sf,
        num_msc_sf,
        num_ysd_sf,
        num_shots,
        _RSF_WORKER_RESULT_DIR,
    )


def run_sweep(
    qc_name: str,
    outdir: Path,
    result_dir: Path,
    num_shots: int,
    num_threads: int,
) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        (num_msd_sf, num_msc_sf, num_ysd_sf, num_shots)
        for num_msd_sf, num_msc_sf, num_ysd_sf in product(SCALE_FACTORS, SCALE_FACTORS, SCALE_FACTORS)
    ]
    worker_count = max(1, min(num_threads or 1, len(inputs)))

    if worker_count == 1:
        sim = load_rsf_trace_sim(outdir)
        rows = [
            run_rsf_point(
                qc_name,
                sim,
                num_msd_sf,
                num_msc_sf,
                num_ysd_sf,
                num_shots,
                result_dir,
            )
            for num_msd_sf, num_msc_sf, num_ysd_sf, num_shots in inputs
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_rsf_worker,
            initargs=(qc_name, outdir, result_dir),
        ) as executor:
            chunksize = max(1, len(inputs) // (worker_count * 4))
            rows = list(executor.map(analyze_qc_cfg_rsf_from_worker, inputs, chunksize=chunksize))

    return pd.DataFrame(rows).reset_index(drop=True)


def geomean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values))))


def summarize_sweep(sweep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in sweep_df.groupby(["num_msd_sf", "num_msc_sf", "num_ysd_sf"]):
        rows.append(
            {
                "rs_sf": keys,
                "num_msd_sf": keys[0],
                "num_msc_sf": keys[1],
                "num_ysd_sf": keys[2],
                "time": geomean(group["time_cost"]),
                "space": geomean(group["space_cost"]),
                "spacetime": geomean(group["spacetime_cost"]),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def save_pareto_plot(summary_df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    df = summary_df
    df_min = df.groupby("time", as_index=False)["space"].min()
    df_min = df_min.sort_values("time")

    pareto_time = []
    pareto_space = []
    best_space = np.inf
    for time_value, space_value in zip(df_min["time"], df_min["space"]):
        if space_value < best_space:
            pareto_time.append(time_value)
            pareto_space.append(space_value)
            best_space = space_value

    best_row = df.loc[df["spacetime"].idxmin()]
    print(f"Min. space-time: {best_row['spacetime']}x at {best_row['rs_sf']}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["time"], df["space"], color="crimson", alpha=0.5, label="data")
    ax.plot(pareto_time, pareto_space, color="crimson", linewidth=2, label="pareto")
    ax.scatter([1], [1], color="navy", marker="x", s=100, label="d3rot")
    ax.scatter(
        [best_row["time"]],
        [best_row["space"]],
        color="darkgreen",
        marker="*",
        s=200,
        label="min_spacetime",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("space")
    ax.legend()
    fig.tight_layout()

    pdf_path = append_extension(output_prefix, ".pdf")
    png_path = append_extension(output_prefix, ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def run_analysis(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
    use_precomputed_mapping: bool,
    num_threads: int,
    cfg_name: str,
    num_shots: int,
    output_dir: Path | None,
) -> None:
    d3rot_base_dir = default_d3rot_base_dir(
        ham_name,
        code_dist,
        use_naive_mapping,
        use_precomputed_mapping,
    )
    outdir = d3rot_base_dir / f"{ham_name}_{cfg_name}"
    validate_result_files(outdir)

    if output_dir is None:
        output_dir = default_output_dir(
            ham_name,
            code_dist,
            use_naive_mapping,
            use_precomputed_mapping,
        )
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading d3rot fixed-d outputs from: {outdir}")
    print(f"writing ms-overhead artifacts to: {output_dir}")
    print(f"scale factors: {SCALE_FACTORS[0]} to {SCALE_FACTORS[-1]} step 0.1")

    result_dir = output_dir / "rsf_analysis" / safe_stem(cfg_name)
    sweep_df = run_sweep(
        ham_name,
        outdir,
        result_dir,
        num_shots,
        num_threads,
    )
    summary_df = summarize_sweep(sweep_df)

    output_prefix = output_dir / f"{ham_name}_ms_overhead_{safe_stem(cfg_name)}_d{code_dist}"
    sweep_csv_path = append_extension(output_prefix, "_sweep.csv")
    summary_csv_path = append_extension(output_prefix, "_summary.csv")
    sweep_df.to_csv(sweep_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    pdf_path, png_path = save_pareto_plot(summary_df, output_prefix)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\nMS overhead sweep summary")
    print(summary_df.sort_values("spacetime").head(20))
    print(f"\nsaved sweep table: {sweep_csv_path}")
    print(f"saved summary table: {summary_csv_path}")
    print(f"saved pareto graph: {pdf_path}")
    print(f"saved pareto graph: {png_path}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        ham_name=args.ham_name,
        code_dist=args.code_distance,
        use_naive_mapping=args.use_naive_mapping,
        use_precomputed_mapping=args.use_precomputed_mapping,
        num_threads=args.num_threads,
        cfg_name=args.cfg_name,
        num_shots=args.num_shots,
        output_dir=args.output_dir,
    )
