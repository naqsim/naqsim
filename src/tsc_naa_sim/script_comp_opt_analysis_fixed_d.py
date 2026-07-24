from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
for subpath in SRC_DIR.iterdir():
    if subpath.is_dir():
        sys.path.append(str(subpath))

from macro import RunOpt
from run_analysis_scripts import (
    analyze_qc_cfg_from_exec_out_dict,
    load_exec_out_dict,
    load_zst,
)
from tsc_instructions import InstType


CFG_NAME_LIST = ["Baseline", "Skip", "Skip+Aggr", "Skip+Dist"]
REQUIRED_RESULT_FILES = [
    "comp_out.zst",
    "exec_out_IGNORE_NONE.zst",
    "exec_out_IGNORE_PC_ROT.zst",
    "exec_out_IGNORE_ROT.zst",
    "exec_out_IGNORE_PC.zst",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Case study #1 Analysis section from "
            "comp_opt_revision_original.ipynb using saved fixed-d outputs."
        )
    )
    parser.add_argument(
        "--ham-name",
        type=str,
        default="SELECT_10_Heisenberg1D_OBC_1_0",
        help="Target hamiltonian name used by script_comp_opt_evaluation_fixed_d.py.",
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
        "--code-distance",
        type=int,
        default=25,
        help="Code distance used by script_comp_opt_evaluation_fixed_d.py.",
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
        "--base-dir",
        type=Path,
        default=None,
        help="Optional explicit result base directory. Defaults to the fixed-d output path.",
    )
    parser.add_argument(
        "--ylim-latency",
        type=float,
        default=7200,
        help="Y-axis upper bound for layer latency.",
    )
    parser.add_argument(
        "--ylim-rotcount",
        type=float,
        default=10,
        help="Y-axis upper bound for rotation counts.",
    )
    args = parser.parse_args()
    if args.use_naive_mapping and args.use_precomputed_mapping:
        parser.error("--use-naive-mapping and --use-precomputed-mapping are mutually exclusive.")
    return args


def default_base_dir(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
    use_precomputed_mapping: bool,
) -> Path:
    if use_naive_mapping:
        mapping_prefix = "naive_mapping_"
    elif use_precomputed_mapping:
        mapping_prefix = "precomputed_mapping_"
    else:
        mapping_prefix = ""
    return (
        SCRIPT_DIR
        / "output"
        / "comp_opt_evaluation_fixed_d"
        / f"{mapping_prefix}{ham_name}_distance{code_dist}"
    )


def build_outdir_dict(ham_name: str, base_dir: Path) -> dict[str, dict[str, Path]]:
    return {
        ham_name: {
            cfg_name: base_dir / f"{ham_name}_{cfg_name}"
            for cfg_name in CFG_NAME_LIST
        }
    }


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
            "Required fixed-d evaluation outputs are missing.",
            "Run script_comp_opt_evaluation_fixed_d.py first with the same arguments.",
            "",
        ]
        lines.extend(f"- {cfg_name}: {path}" for cfg_name, path in missing)
        raise FileNotFoundError("\n".join(lines))


def append_extension(output_path_prefix: Path, extension: str) -> Path:
    return output_path_prefix.with_name(output_path_prefix.name + extension)


def rotcount_cs1_single(
    qc_name: str,
    cfg_name: str,
    outdir_dict: dict[str, dict[str, Path]],
) -> dict[str, list[int] | list[str]]:
    outdir = outdir_dict[qc_name][cfg_name]
    comp_out = load_zst(str(outdir / "comp_out"))
    _, _, inst_schedule_trace, _, _ = comp_out

    label_list = []
    rot_count_list = []
    rot_types = {InstType.ROTATION, InstType.TRANS_H_ROT}

    for idx, inst_schedule in enumerate(inst_schedule_trace):
        rot_count = sum(1 for inst in inst_schedule if inst.inst_type in rot_types)
        label_list.append(f"L{idx}")
        rot_count_list.append(rot_count)

    return {"Label": label_list, "RotCount": rot_count_list}


def latency_cs1_single(
    qc_name: str,
    cfg_name: str,
    outdir_dict: dict[str, dict[str, Path]],
) -> dict[str, list[float] | list[str]]:
    outdir = outdir_dict[qc_name][cfg_name]
    exec_out_dict = load_exec_out_dict(
        str(outdir),
        [
            RunOpt.IGNORE_PC_ROT,
            RunOpt.IGNORE_ROT,
            RunOpt.IGNORE_PC,
            RunOpt.IGNORE_NONE,
        ],
    )
    return analyze_qc_cfg_from_exec_out_dict(qc_name, exec_out_dict, draw_graph=False)


def getdf_cs1_single(
    qc_name: str,
    cfg_name: str,
    outdir_dict: dict[str, dict[str, Path]],
) -> pd.DataFrame:
    outdir = outdir_dict[qc_name][cfg_name]
    exec_out_dict = load_exec_out_dict(
        str(outdir),
        [
            RunOpt.IGNORE_PC_ROT,
            RunOpt.IGNORE_ROT,
            RunOpt.IGNORE_PC,
            RunOpt.IGNORE_NONE,
        ],
    )
    comp_out = load_zst(str(outdir / "comp_out"))

    latency_df = pd.DataFrame(
        analyze_qc_cfg_from_exec_out_dict(qc_name, exec_out_dict)
    )
    _, _, inst_schedule_trace, _, _ = comp_out

    rot_types = {InstType.ROTATION, InstType.TRANS_H_ROT}
    rotcount_df = pd.DataFrame(
        {
            "Label": [f"L{idx}" for idx, _ in enumerate(inst_schedule_trace)],
            "RotCount": [
                sum(1 for inst in inst_schedule if inst.inst_type in rot_types)
                for inst_schedule in inst_schedule_trace
            ],
        }
    )

    merged_df = pd.merge(latency_df, rotcount_df, on="Label")
    return merged_df.sort_values(
        by=["RotCount", "Sum"],
        ascending=[True, True],
    ).reset_index(drop=True)


def collect_case_study_data(
    qc_name: str,
    outdir_dict: dict[str, dict[str, Path]],
    num_threads: int,
) -> dict[str, pd.DataFrame]:
    worker_count = max(1, min(num_threads or 1, len(CFG_NAME_LIST)))
    if worker_count == 1:
        return {
            cfg_name: getdf_cs1_single(qc_name, cfg_name, outdir_dict)
            for cfg_name in CFG_NAME_LIST
        }

    data_by_cfg = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_by_cfg = {
            cfg_name: executor.submit(
                getdf_cs1_single,
                qc_name,
                cfg_name,
                outdir_dict,
            )
            for cfg_name in CFG_NAME_LIST
        }
        for cfg_name in CFG_NAME_LIST:
            data_by_cfg[cfg_name] = future_by_cfg[cfg_name].result()
    return data_by_cfg


def draw_cs1_single(
    cfg_name: str,
    df: pd.DataFrame,
    ax,
    ylim_latency: float,
    ylim_rotcount: float,
):
    latency_cols = [
        col for col in df.columns
        if col not in ["Label", "Sum", "RotCount"]
    ]
    plot_df = df.copy()
    plot_df[latency_cols] = plot_df[latency_cols].clip(lower=0)

    plot_df.plot(
        ax=ax,
        x="Label",
        y=latency_cols,
        kind="area",
        stacked=True,
        linewidth=0,
    )
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_xlabel("Layer id")
    ax.set_ylabel("Layer latency (us)")
    ax.set_ylim(0, ylim_latency)
    ax.legend().remove()
    ax.set_title(cfg_name)

    ax2 = ax.twinx()
    plot_df.plot(
        ax=ax2,
        x="Label",
        y="RotCount",
        kind="line",
        marker="o",
        color="black",
        linewidth=1.0,
        markersize=1,
    )
    ax2.set_ylabel("Rotation counts")
    ax2.set_ylim(0, ylim_rotcount)
    ax2.legend().remove()
    return ax, ax2


def draw_case_study_graph(
    qc_name: str,
    data_by_cfg: dict[str, pd.DataFrame],
    output_path_prefix: Path,
    ylim_latency: float,
    ylim_rotcount: float,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes_flat = axes.ravel()
    ax1 = ax2 = None

    for ax, cfg_name in zip(axes_flat, CFG_NAME_LIST):
        ax1, ax2 = draw_cs1_single(
            cfg_name,
            data_by_cfg[cfg_name],
            ax,
            ylim_latency=ylim_latency,
            ylim_rotcount=ylim_rotcount,
        )

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper center",
        ncol=3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    pdf_path = append_extension(output_path_prefix, ".pdf")
    png_path = append_extension(output_path_prefix, ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def make_case_study_table(data_by_cfg: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rotcount_name = "Rotation counts"
    rotmax_name = "Max rotations per layer"
    rotlayer_name = "Rotation layer occupancy (%)"
    latency_name = "Average layer latency (us)"

    ret_df = pd.DataFrame(
        data=np.zeros((4, len(CFG_NAME_LIST))),
        index=[rotcount_name, rotmax_name, rotlayer_name, latency_name],
        columns=CFG_NAME_LIST,
    )

    for cfg_name in CFG_NAME_LIST:
        df = data_by_cfg[cfg_name]
        ret_df.loc[rotcount_name, cfg_name] = df["RotCount"].sum()
        ret_df.loc[rotmax_name, cfg_name] = df["RotCount"].max()
        ret_df.loc[rotlayer_name, cfg_name] = (
            100 * (df["RotCount"] > 0).sum() / len(df)
        )
        ret_df.loc[latency_name, cfg_name] = df["Sum"].mean()

    return ret_df


def save_layer_csvs(
    qc_name: str,
    code_dist: int,
    base_dir: Path,
    data_by_cfg: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    csv_paths = {}
    for cfg_name in CFG_NAME_LIST:
        output_prefix = (
            base_dir
            / f"{qc_name}_case_study_1_analysis_{cfg_name}_layers_d{code_dist}"
        )
        csv_path = append_extension(output_prefix, ".csv")
        data_by_cfg[cfg_name].to_csv(csv_path, index=False)
        csv_paths[cfg_name] = csv_path
    return csv_paths


def run_analysis(
    ham_name: str,
    code_dist: int,
    use_naive_mapping: bool,
    use_precomputed_mapping: bool,
    base_dir: Path | None,
    ylim_latency: float,
    ylim_rotcount: float,
    num_threads: int,
) -> None:
    qc_name = ham_name
    if base_dir is None:
        base_dir = default_base_dir(
            ham_name,
            code_dist,
            use_naive_mapping,
            use_precomputed_mapping,
        )
    else:
        base_dir = base_dir.resolve()

    outdir_dict = build_outdir_dict(qc_name, base_dir)
    print(f"reading fixed-d outputs from: {base_dir}")
    validate_result_files(outdir_dict)

    data_by_cfg = collect_case_study_data(qc_name, outdir_dict, num_threads)

    output_prefix = base_dir / f"{qc_name}_case_study_1_analysis_d{code_dist}"
    pdf_path, png_path = draw_case_study_graph(
        qc_name,
        data_by_cfg,
        output_prefix,
        ylim_latency=ylim_latency,
        ylim_rotcount=ylim_rotcount,
    )

    table_df = make_case_study_table(data_by_cfg)
    csv_path = append_extension(output_prefix, ".csv")
    table_df.to_csv(csv_path)
    layer_csv_paths = save_layer_csvs(qc_name, code_dist, base_dir, data_by_cfg)

    print("\nCase study #1. Result analysis (Table)")
    print(table_df)
    print(f"\nsaved graph: {pdf_path}")
    print(f"saved graph: {png_path}")
    print(f"saved table: {csv_path}")
    for cfg_name, layer_csv_path in layer_csv_paths.items():
        print(f"saved {cfg_name} layer data: {layer_csv_path}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        ham_name=args.ham_name,
        code_dist=args.code_distance,
        use_naive_mapping=args.use_naive_mapping,
        use_precomputed_mapping=args.use_precomputed_mapping,
        base_dir=args.base_dir,
        ylim_latency=args.ylim_latency,
        ylim_rotcount=args.ylim_rotcount,
        num_threads=args.num_threads,
    )
