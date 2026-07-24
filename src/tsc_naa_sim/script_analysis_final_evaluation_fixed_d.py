from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = "output"
SRC_DIR = SCRIPT_DIR.parent
os.chdir(SCRIPT_DIR)
for subpath in SRC_DIR.iterdir():
    if subpath.is_dir():
        sys.path.append(str(subpath))

from config import experiment_config
from macro import (
    AodSchedOpt,
    CellSize,
    RotPlaneOpt,
    RotSchedOpt,
    RotTransOpt,
    RotType,
    RunOpt,
)
from plane_initializer import get_rectangle
from run_analysis_scripts import (
    analyze_qc_cfg_from_exec_out_dict,
    error_qc_cfg_from_outputs,
    get_aod_latency_from_uop_schedule,
    get_esm_latency_from_uop_schedule,
    get_finish_us_from_uop_schedule,
    load_exec_out_dict,
    load_zst,
)


FINAL_CFG_NAME_LIST = ["Baseline", "+ opt. #1", "+ opt. #2"]
SOURCE_CFG_MAP = {
    "Baseline": ("comp_opt", "Baseline"),
    "+ opt. #1": ("comp_opt", "Skip+Aggr"),
    "+ opt. #2": ("d3rot_opt", "DIR_TOGL+DEDICATE_CELL3"),
}
REQUIRED_RESULT_FILES = [
    "comp_out.zst",
    "exec_out_IGNORE_NONE.zst",
]
FULL_BREAKDOWN_RUN_OPTS = [
    RunOpt.IGNORE_PC_ROT,
    RunOpt.IGNORE_ROT,
    RunOpt.IGNORE_PC,
    RunOpt.IGNORE_NONE,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze final evaluation from existing comp_opt and d3rot_opt "
            "fixed-d outputs without running script_final_evaluation_fixed_d.py."
        )
    )
    parser.add_argument(
        "--ham-name",
        type=str,
        default="SELECT_10_Heisenberg1D_OBC_1_0",
        help="Target hamiltonian name used by the evaluation scripts.",
    )
    parser.add_argument(
        "--use-naive-mapping",
        action="store_true",
        help="Read outputs generated with --use-naive-mapping.",
    )
    parser.add_argument(
        "--code-distance",
        type=int,
        default=25,
        help="Code distance used by the evaluation scripts.",
    )
    parser.add_argument(
        "--num-threads",
        "--num_threads",
        dest="num_threads",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes for per-cfg analysis. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--comp-opt-base-dir",
        type=Path,
        default=None,
        help="Optional comp_opt result base directory.",
    )
    parser.add_argument(
        "--d3rot-opt-base-dir",
        type=Path,
        default=None,
        help="Optional d3rot_opt result base directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for final-analysis artifacts.",
    )
    parser.add_argument(
        "--d3rot-opt2-cfg-name",
        type=str,
        default=SOURCE_CFG_MAP["+ opt. #2"][1],
        help=(
            "d3rot_opt source configuration for final '+ opt. #2'. "
            "Default matches num_rot_cell=2."
        ),
    )
    parser.add_argument(
        "--ls-summary-csv",
        type=Path,
        default=None,
        help="Optional lattice-surgery summary CSV to append LS bars.",
    )
    return parser.parse_args()


def prefixed_name(ham_name: str, code_dist: int, use_naive_mapping: bool) -> str:
    naive_prefix = "naive_mapping_" if use_naive_mapping else ""
    return f"{naive_prefix}{ham_name}_distance{code_dist}"


def default_comp_opt_base_dir(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
) -> Path:
    return (
        SCRIPT_DIR
        / OUTPUT_DIR
        / "comp_opt_evaluation_fixed_d"
        / prefixed_name(ham_name, code_dist, use_naive_mapping)
    )


def default_d3rot_opt_base_dir(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
) -> Path:
    return (
        SCRIPT_DIR
        / OUTPUT_DIR
        / "d3rot_opt_evaluation_fixed_d"
        / prefixed_name(ham_name, code_dist, use_naive_mapping)
    )


def default_output_dir(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
) -> Path:
    return (
        SCRIPT_DIR
        / OUTPUT_DIR
        / "final_evaluation_fixed_d"
        / prefixed_name(ham_name, code_dist, use_naive_mapping)
    )


def make_final_target_cfg_dict(
    code_dist: int,
) -> dict[str, tuple[experiment_config, list[RunOpt]]]:
    run_opts = [RunOpt.IGNORE_NONE]
    target_cfg_dict = {}

    cfg_in_base = experiment_config()
    cfg_in_base.code_dist = code_dist
    target_cfg_dict["Baseline"] = (cfg_in_base, run_opts)

    cfg_in_opt1 = experiment_config()
    cfg_in_opt1.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_opt1.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg_in_opt1.code_dist = code_dist
    target_cfg_dict["+ opt. #1"] = (cfg_in_opt1, run_opts)

    cfg_in_opt2 = experiment_config()
    cfg_in_opt2.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_opt2.rot_sched_opt = RotSchedOpt.DISTRIBUTE
    cfg_in_opt2.cell_size = CellSize.SMALLEST
    cfg_in_opt2.rot_type = RotType.DIR_TOGL
    cfg_in_opt2.refl_type_h = None
    cfg_in_opt2.refl_type_d = None
    cfg_in_opt2.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    cfg_in_opt2.num_rot_cell = 2
    cfg_in_opt2.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    cfg_in_opt2.code_dist = code_dist
    target_cfg_dict["+ opt. #2"] = (cfg_in_opt2, run_opts)

    return target_cfg_dict


def build_source_outdir_dict(
    ham_name: str,
    comp_opt_base_dir: Path,
    d3rot_opt_base_dir: Path,
    d3rot_opt2_cfg_name: str,
) -> dict[str, dict[str, Path]]:
    source_cfg_map = dict(SOURCE_CFG_MAP)
    source_cfg_map["+ opt. #2"] = ("d3rot_opt", d3rot_opt2_cfg_name)

    outdirs = {}
    for final_cfg_name in FINAL_CFG_NAME_LIST:
        source_kind, source_cfg_name = source_cfg_map[final_cfg_name]
        if source_kind == "comp_opt":
            base_dir = comp_opt_base_dir
        elif source_kind == "d3rot_opt":
            base_dir = d3rot_opt_base_dir
        else:
            raise ValueError(f"Unknown source kind: {source_kind}")
        outdirs[final_cfg_name] = base_dir / f"{ham_name}_{source_cfg_name}"

    return {ham_name: outdirs}


def exec_out_path(outdir: Path, run_opt: RunOpt) -> Path:
    return outdir / f"exec_out_{run_opt.name}.zst"


def has_full_breakdown_outputs(outdir: Path) -> bool:
    return all(exec_out_path(outdir, run_opt).is_file() for run_opt in FULL_BREAKDOWN_RUN_OPTS)


def validate_result_files(outdir_dict: dict[str, dict[str, Path]]) -> None:
    missing = []
    for cfg_dirs in outdir_dict.values():
        for cfg_name, outdir in cfg_dirs.items():
            for filename in REQUIRED_RESULT_FILES:
                path = outdir / filename
                if not path.is_file():
                    missing.append((cfg_name, path))

    if missing:
        lines = [
            "Required fixed-d final-analysis outputs are missing.",
            "Run script_comp_opt_evaluation_fixed_d.py and "
            "script_d3rot_opt_evaluation_fixed_d.py first with at least "
            "RunOpt.IGNORE_NONE and the same arguments.",
            "",
        ]
        lines.extend(f"- {cfg_name}: {path}" for cfg_name, path in missing)
        raise FileNotFoundError("\n".join(lines))


def load_num_qubits(outdir_dict: dict[str, dict[str, Path]], ham_name: str) -> int:
    for cfg_name in FINAL_CFG_NAME_LIST:
        qc_path = outdir_dict[ham_name][cfg_name] / "qc_in"
        if qc_path.with_suffix(".zst").is_file():
            return load_zst(str(qc_path)).num_qubits
    raise FileNotFoundError("No qc_in.zst file found in the source result directories.")


def analyze_qc_cfg_from_ignore_none_exec_out(qc_name: str, exec_out, draw_graph=False) -> dict:
    uop_schedule_trace_all, _, _ = exec_out

    label_list = []
    esm_list = []
    others_list = []
    move_list = []
    pc_mov_list = []
    rot_list = []
    pc_rot_list = []
    sum_list = []

    for idx, uop_schedule_all in enumerate(uop_schedule_trace_all):
        finish_all = get_finish_us_from_uop_schedule(uop_schedule_all)
        esm_latency = get_esm_latency_from_uop_schedule(uop_schedule_all)
        move_latency = get_aod_latency_from_uop_schedule(uop_schedule_all)
        others_latency = finish_all - esm_latency - move_latency

        label_list.append(f"L{idx}")
        esm_list.append(esm_latency)
        others_list.append(others_latency)
        move_list.append(move_latency)
        pc_mov_list.append(0)
        rot_list.append(0)
        pc_rot_list.append(0)
        sum_list.append(finish_all)

    qc_cfg_res = {
        "Label": label_list,
        "ESM": esm_list,
        "Others": others_list,
        "Move": move_list,
        "Route_Conflict (Move)": pc_mov_list,
        "Route_Conflict (Rot)": pc_rot_list,
        "Rotation": rot_list,
        "Sum": sum_list,
    }

    if draw_graph:
        df = pd.DataFrame(qc_cfg_res)
        df = df.sort_values(by="Sum").reset_index(drop=True)
        num_cols = df.select_dtypes(include="number")
        df[num_cols.columns] = num_cols.clip(lower=0)
        ax = df.plot(
            x="Label",
            y=[col for col in df.columns if col not in ["Label", "Sum"]],
            kind="area",
            stacked=True,
            linewidth=0,
        )
        plt.title(f"qc: {qc_name}")
        ax.set_ylabel("Latency (us)")
        plt.xlabel("Layer idx")
        plt.tight_layout()
        plt.show()

    return qc_cfg_res


def final_cfg_row(
    qc_name: str,
    cfg_name: str,
    cfg_in: experiment_config,
    outdir: Path,
    code_dist: int,
) -> tuple[str, dict]:
    comp_out = load_zst(str(outdir / "comp_out"))

    if has_full_breakdown_outputs(outdir):
        exec_out_dict = load_exec_out_dict(str(outdir), FULL_BREAKDOWN_RUN_OPTS)
        exec_out = exec_out_dict[RunOpt.IGNORE_NONE]
        qc_cfg_res = analyze_qc_cfg_from_exec_out_dict(
            qc_name,
            exec_out_dict,
            draw_graph=False,
        )
    else:
        exec_out = load_zst(str(outdir / f"exec_out_{RunOpt.IGNORE_NONE.name}"))
        qc_cfg_res = analyze_qc_cfg_from_ignore_none_exec_out(
            qc_name,
            exec_out,
            draw_graph=False,
        )

    qc_cfg_err = error_qc_cfg_from_outputs(
        qc_name,
        cfg_in,
        exec_out,
        comp_out,
        code_dist,
        draw_graph=False,
    )

    row = {"Label": cfg_name}
    for key, values in qc_cfg_res.items():
        if key == "Label":
            row["Num_Layers"] = len(values)
        else:
            row[key] = sum(values) / len(values)

    for key, values in qc_cfg_err.items():
        if key == "Label":
            continue
        row[key] = sum(values) / len(values)

    return cfg_name, row


def collect_final_qc_result(
    qc_name: str,
    target_cfg_dict: dict[str, tuple[experiment_config, list[RunOpt]]],
    outdir_dict: dict[str, dict[str, Path]],
    code_dist: int,
    num_threads: int,
) -> dict:
    worker_count = max(1, min(num_threads or 1, len(FINAL_CFG_NAME_LIST)))

    if worker_count == 1:
        row_by_cfg = {}
        for cfg_name in FINAL_CFG_NAME_LIST:
            print(cfg_name, "start")
            _, row = final_cfg_row(
                qc_name,
                cfg_name,
                target_cfg_dict[cfg_name][0],
                outdir_dict[qc_name][cfg_name],
                code_dist,
            )
            print(cfg_name, "end")
            row_by_cfg[cfg_name] = row
    else:
        row_by_cfg = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_by_cfg = {
                cfg_name: executor.submit(
                    final_cfg_row,
                    qc_name,
                    cfg_name,
                    target_cfg_dict[cfg_name][0],
                    outdir_dict[qc_name][cfg_name],
                    code_dist,
                )
                for cfg_name in FINAL_CFG_NAME_LIST
            }
            for cfg_name in FINAL_CFG_NAME_LIST:
                _, row = future_by_cfg[cfg_name].result()
                row_by_cfg[cfg_name] = row

    qc_res = {"Label": []}
    for cfg_name in FINAL_CFG_NAME_LIST:
        row = row_by_cfg[cfg_name]
        for key, value in row.items():
            if key not in qc_res:
                qc_res[key] = []
            qc_res[key].append(value)
    return qc_res


def logical_error_rate_by_distance() -> dict[int, float]:
    a = -0.5180095419542551
    b = -1.774907026108311
    ret = {
        3: 0.0001907376,
        5: 3.3788e-05,
        7: 3.913e-06,
        9: 3.772e-07,
        11: 3.48e-08,
        13: 2.8e-09,
        15: 3.0e-10,
    }
    for distance in [17, 19, 21, 23, 25, 27, 29, 31]:
        ret[distance] = 10 ** (a * distance + b)
    return ret


def save_result_table(results: list[dict], output_path: Path) -> None:
    rows = []
    for qc_res in results:
        rows.append(pd.DataFrame(qc_res))
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)


def append_extension(output_prefix: Path, extension: str) -> Path:
    return output_prefix.with_name(output_prefix.name + extension)


def save_final_graph(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
    output_dir: Path,
    results: list[dict],
    num_qubits: int,
    ls_summary_csv: Path | None,
) -> tuple[Path, Path]:
    width, height = get_rectangle(n=num_qubits, max_diff=2)
    physical_qubits_transversal = width * height * (2 * code_dist * code_dist - 1)

    ls_summary_df = None
    filtered_ls_df = None
    ls_distance_list = [code_dist, code_dist + 2, code_dist + 4]
    if ls_summary_csv is not None:
        ls_summary_df = pd.read_csv(ls_summary_csv)
        filtered_ls_df = ls_summary_df[ls_summary_df["ham_name"] == ham_name]
        if filtered_ls_df.empty:
            raise ValueError(f"{ham_name} was not found in {ls_summary_csv}.")

    all_keys = list(next(iter(results)).keys())
    except_keys = ["Label", "Num_Layers", "Sum", "Logical_Error"]
    val_keys = [key for key in all_keys if key not in except_keys]
    cmap_mpl = plt.get_cmap("tab10")
    colors = {key: cmap_mpl(i) for i, key in enumerate(val_keys)}

    offsets = []
    current = 0
    ls_bar_count = len(ls_distance_list) if filtered_ls_df is not None else 0
    for result in results:
        current += 0.5
        length = len(next(iter(result.values())))
        start = current
        end = current + length
        offsets.append((start, end))
        current = end + ls_bar_count + 0.5

    fig, ax = plt.subplots(figsize=(max(8, 4 * len(results)), 5.5))
    ax2 = ax.twinx()
    legend_added = False
    tick_positions = []
    tick_labels = []
    logical_error_rate_dict = logical_error_rate_by_distance()
    esm_duration_sec = 0.000237

    for result, (start, end) in zip(results, offsets):
        x = np.arange(start, end)
        bottom = np.zeros_like(x, dtype=float)

        for key in val_keys:
            vals = np.array(result.get(key, [0] * len(x))) * result["Num_Layers"][0]
            ax.bar(
                x,
                vals,
                width=1,
                bottom=bottom,
                edgecolor="black",
                linewidth=1.0,
                color=colors[key],
                label=key if not legend_added else None,
            )
            bottom += vals

        labels = list(result["Label"])
        x_with_ls = np.array(x)
        y = np.array(result["Logical_Error"]) * np.array(result["Num_Layers"])

        if filtered_ls_df is not None:
            ls_color_list = ["0.6", "0.5", "0.4"]
            for ls_idx, ls_distance in enumerate(ls_distance_list):
                ls_x = end + ls_idx
                duration = (
                    filtered_ls_df["beats"].item()
                    * ls_distance
                    * esm_duration_sec
                    * 1e6
                )
                ax.bar(
                    ls_x,
                    duration,
                    width=1,
                    bottom=0,
                    edgecolor="black",
                    linewidth=1.0,
                    color=ls_color_list[ls_idx],
                    label=f"LS (d={ls_distance})" if not legend_added else None,
                )
                qubits_per_cell = ls_distance * ls_distance * 2 - 1
                physical_qubits_lattice_surgery = (
                    filtered_ls_df["chip_width"].item()
                    * filtered_ls_df["chip_height"].item()
                    * qubits_per_cell
                )
                ax.text(
                    ls_x,
                    duration,
                    f"#Qubits\n={physical_qubits_lattice_surgery}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )
                y = np.append(
                    y,
                    filtered_ls_df["space_time_volume"].item()
                    * logical_error_rate_dict[ls_distance],
                )
                labels.append(f"LS\n$d={ls_distance}$")

            x_with_ls = np.arange(start, end + len(ls_distance_list))

        ax2.plot(x_with_ls, y, color="black", marker="o", ms=3)
        tick_positions.extend(x_with_ls)
        tick_labels.extend(labels)
        legend_added = True

    first_result = results[0]
    ax.set_title(ham_name + f", #Layers: {first_result['Num_Layers'][0]}")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.text(
        0.98,
        0.95,
        f"#Qubits={physical_qubits_transversal}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set_xlabel("Configuration name")
    ax.set_ylabel("Total execution time (us)")
    ax2.set_ylabel("Logical error rate of program")
    ax2.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    naive_prefix = "naive_mapping_" if use_naive_mapping else ""
    output_prefix = output_dir / f"{ham_name}_{naive_prefix}d_{code_dist}"
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
    num_threads: int,
    comp_opt_base_dir: Path | None,
    d3rot_opt_base_dir: Path | None,
    output_dir: Path | None,
    d3rot_opt2_cfg_name: str,
    ls_summary_csv: Path | None,
) -> None:
    if comp_opt_base_dir is None:
        comp_opt_base_dir = default_comp_opt_base_dir(
            ham_name,
            code_dist,
            use_naive_mapping,
        )
    else:
        comp_opt_base_dir = comp_opt_base_dir.resolve()

    if d3rot_opt_base_dir is None:
        d3rot_opt_base_dir = default_d3rot_opt_base_dir(
            ham_name,
            code_dist,
            use_naive_mapping,
        )
    else:
        d3rot_opt_base_dir = d3rot_opt_base_dir.resolve()

    if output_dir is None:
        output_dir = default_output_dir(ham_name, code_dist, use_naive_mapping)
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_cfg_dict = make_final_target_cfg_dict(code_dist)
    outdir_dict = build_source_outdir_dict(
        ham_name,
        comp_opt_base_dir,
        d3rot_opt_base_dir,
        d3rot_opt2_cfg_name,
    )

    print(f"reading comp_opt outputs from: {comp_opt_base_dir}")
    print(f"reading d3rot_opt outputs from: {d3rot_opt_base_dir}")
    print(f"writing final-analysis artifacts to: {output_dir}")
    for cfg_name, outdir in outdir_dict[ham_name].items():
        print(f"  {cfg_name}: {outdir}")

    validate_result_files(outdir_dict)
    num_qubits = load_num_qubits(outdir_dict, ham_name)

    qc_res = collect_final_qc_result(
        ham_name,
        target_cfg_dict,
        outdir_dict,
        code_dist,
        num_threads,
    )
    results = [qc_res]

    naive_prefix = "naive_mapping_" if use_naive_mapping else ""
    output_prefix = output_dir / f"{ham_name}_{naive_prefix}d_{code_dist}"
    pkl_path = append_extension(output_prefix, ".pkl")
    csv_path = append_extension(output_prefix, ".csv")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    save_result_table(results, csv_path)

    pdf_path, png_path = save_final_graph(
        ham_name,
        code_dist,
        use_naive_mapping,
        output_dir,
        results,
        num_qubits,
        ls_summary_csv,
    )

    print("\nFinal evaluation analysis")
    print(pd.DataFrame(qc_res))
    print(f"\nsaved result cache: {pkl_path}")
    print(f"saved table: {csv_path}")
    print(f"saved graph: {pdf_path}")
    print(f"saved graph: {png_path}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        ham_name=args.ham_name,
        code_dist=args.code_distance,
        use_naive_mapping=args.use_naive_mapping,
        num_threads=args.num_threads,
        comp_opt_base_dir=args.comp_opt_base_dir,
        d3rot_opt_base_dir=args.d3rot_opt_base_dir,
        output_dir=args.output_dir,
        d3rot_opt2_cfg_name=args.d3rot_opt2_cfg_name,
        ls_summary_csv=args.ls_summary_csv,
    )
