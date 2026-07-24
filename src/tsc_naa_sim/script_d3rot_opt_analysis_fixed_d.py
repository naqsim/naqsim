from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
os.chdir(SCRIPT_DIR)
for subpath in SRC_DIR.iterdir():
    if subpath.is_dir():
        sys.path.append(str(subpath))

from config import experiment_config
from macro import (
    AodSchedOpt,
    CellSize,
    ReflType,
    RotPlaneOpt,
    RotSchedOpt,
    RotTransOpt,
    RotType,
    RunOpt,
)
from run_analysis_scripts import (
    analyze_qc_cfg_from_exec_out_dict,
    error_qc_cfg_from_outputs,
    load_exec_out_dict,
    load_zst,
)


CFG_NAME_LIST = [
    "REFL_SE",
    "REFL_TE",
    "DIR_CHANGE",
    "DIR_TOGL",
    "DIR_TOGL+DEDICATE_CELL1",
    "DIR_TOGL+DEDICATE_CELL2",
    "DIR_TOGL+DEDICATE_CELL3",
    "DIR_IDEAL",
]
DEDICATE_CFG_NAME_LIST = [
    "DIR_TOGL+DEDICATE_CELL1",
    "DIR_TOGL+DEDICATE_CELL2",
    "DIR_TOGL+DEDICATE_CELL3",
]
REQUIRED_CFG_NAME_LIST = [
    cfg_name for cfg_name in CFG_NAME_LIST
    if cfg_name not in DEDICATE_CFG_NAME_LIST
]
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
            "Run the Analysis section from d3rot_opt_revision_original.ipynb "
            "using saved fixed-d outputs."
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
        help="Number of worker processes for per-cfg analysis. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Optional explicit result base directory. Defaults to the fixed-d output path.",
    )
    parser.add_argument(
        "--detail-cfg-name",
        type=str,
        default="DIR_TOGL+DEDICATE_CELL2",
        choices=CFG_NAME_LIST,
        help="Configuration name for the per-layer analyze_qc_cfg output.",
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
        / "d3rot_opt_evaluation_fixed_d"
        / f"{mapping_prefix}{ham_name}_distance{code_dist}"
    )


def make_target_cfg_dict(code_dist: int) -> dict[str, tuple[experiment_config, list[RunOpt]]]:
    run_opts = [
        RunOpt.IGNORE_NONE,
        RunOpt.IGNORE_PC_ROT,
        RunOpt.IGNORE_ROT,
        RunOpt.IGNORE_PC,
    ]
    target_cfg_dict = {}

    cfg_in_refl_se = experiment_config()
    cfg_in_refl_se.code_dist = code_dist
    cfg_in_refl_se.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_refl_se.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg_in_refl_se.cell_size = CellSize.SMALLEST
    cfg_in_refl_se.rot_type = RotType.REFL
    cfg_in_refl_se.refl_type_h = ReflType.STATIC_SE
    cfg_in_refl_se.refl_type_d = ReflType.STATIC_SE
    cfg_in_refl_se.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_refl_se.aod_sched_opt = AodSchedOpt.NAIVE_DRAIN
    target_cfg_dict["REFL_SE"] = (cfg_in_refl_se, run_opts)

    cfg_in_refl_te = experiment_config()
    cfg_in_refl_te.code_dist = code_dist
    cfg_in_refl_te.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_refl_te.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg_in_refl_te.cell_size = CellSize.DOUBLE_TE
    cfg_in_refl_te.rot_type = RotType.REFL
    cfg_in_refl_te.refl_type_h = ReflType.STATIC_TE
    cfg_in_refl_te.refl_type_d = ReflType.STATIC_TE
    cfg_in_refl_te.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_refl_te.aod_sched_opt = AodSchedOpt.NAIVE_DRAIN
    target_cfg_dict["REFL_TE"] = (cfg_in_refl_te, run_opts)

    cfg_in_dir_change = experiment_config()
    cfg_in_dir_change.code_dist = code_dist
    cfg_in_dir_change.cell_size = CellSize.DOUBLE_DIR
    cfg_in_dir_change.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_change.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg_in_dir_change.rot_type = RotType.DIR_CHANGE
    cfg_in_dir_change.refl_type_h = None
    cfg_in_dir_change.refl_type_d = None
    cfg_in_dir_change.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_dir_change.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    target_cfg_dict["DIR_CHANGE"] = (cfg_in_dir_change, run_opts)

    cfg_in_dir_togl = experiment_config()
    cfg_in_dir_togl.code_dist = code_dist
    cfg_in_dir_togl.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg_in_dir_togl.cell_size = CellSize.DOUBLE_DIR
    cfg_in_dir_togl.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl.refl_type_h = None
    cfg_in_dir_togl.refl_type_d = None
    cfg_in_dir_togl.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_dir_togl.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    target_cfg_dict["DIR_TOGL"] = (cfg_in_dir_togl, run_opts)

    cfg_in_dir_togl_dedicate = experiment_config()
    cfg_in_dir_togl_dedicate.code_dist = code_dist
    cfg_in_dir_togl_dedicate.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl_dedicate.rot_sched_opt = RotSchedOpt.DISTRIBUTE
    cfg_in_dir_togl_dedicate.cell_size = CellSize.SMALLEST
    cfg_in_dir_togl_dedicate.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl_dedicate.refl_type_h = None
    cfg_in_dir_togl_dedicate.refl_type_d = None
    cfg_in_dir_togl_dedicate.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    cfg_in_dir_togl_dedicate.num_rot_cell = 1
    cfg_in_dir_togl_dedicate.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    target_cfg_dict["DIR_TOGL+DEDICATE_CELL1"] = (cfg_in_dir_togl_dedicate, run_opts)

    cfg_in_dir_togl_dedicate = experiment_config()
    cfg_in_dir_togl_dedicate.code_dist = code_dist
    cfg_in_dir_togl_dedicate.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl_dedicate.rot_sched_opt = RotSchedOpt.DISTRIBUTE
    cfg_in_dir_togl_dedicate.cell_size = CellSize.SMALLEST
    cfg_in_dir_togl_dedicate.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl_dedicate.refl_type_h = None
    cfg_in_dir_togl_dedicate.refl_type_d = None
    cfg_in_dir_togl_dedicate.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    cfg_in_dir_togl_dedicate.num_rot_cell = 2
    cfg_in_dir_togl_dedicate.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    target_cfg_dict["DIR_TOGL+DEDICATE_CELL2"] = (cfg_in_dir_togl_dedicate, run_opts)

    cfg_in_dir_togl_dedicate = experiment_config()
    cfg_in_dir_togl_dedicate.code_dist = code_dist
    cfg_in_dir_togl_dedicate.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl_dedicate.rot_sched_opt = RotSchedOpt.DISTRIBUTE
    cfg_in_dir_togl_dedicate.cell_size = CellSize.SMALLEST
    cfg_in_dir_togl_dedicate.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl_dedicate.refl_type_h = None
    cfg_in_dir_togl_dedicate.refl_type_d = None
    cfg_in_dir_togl_dedicate.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    cfg_in_dir_togl_dedicate.num_rot_cell = 3
    cfg_in_dir_togl_dedicate.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    target_cfg_dict["DIR_TOGL+DEDICATE_CELL3"] = (cfg_in_dir_togl_dedicate, run_opts)

    cfg_in_dir_ideal = experiment_config()
    cfg_in_dir_ideal.code_dist = code_dist
    cfg_in_dir_ideal.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_ideal.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg_in_dir_ideal.cell_size = CellSize.DOUBLE_DIR
    cfg_in_dir_ideal.rot_type = RotType.DIR_IDEAL
    cfg_in_dir_ideal.refl_type_h = None
    cfg_in_dir_ideal.refl_type_d = None
    cfg_in_dir_ideal.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_dir_ideal.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    target_cfg_dict["DIR_IDEAL"] = (cfg_in_dir_ideal, run_opts)

    return target_cfg_dict


def build_outdir_dict(ham_name: str, base_dir: Path) -> dict[str, dict[str, Path]]:
    return {
        ham_name: {
            cfg_name: base_dir / f"{ham_name}_{cfg_name}"
            for cfg_name in CFG_NAME_LIST
        }
    }


def is_cfg_complete(outdir: Path) -> bool:
    return all((outdir / filename).is_file() for filename in REQUIRED_RESULT_FILES)


def complete_cfg_names(outdir_dict: dict[str, dict[str, Path]], qc_name: str) -> list[str]:
    return [
        cfg_name for cfg_name in CFG_NAME_LIST
        if is_cfg_complete(outdir_dict[qc_name][cfg_name])
    ]


def validate_result_files(
    outdir_dict: dict[str, dict[str, Path]],
    qc_name: str,
) -> list[str]:
    complete_names = complete_cfg_names(outdir_dict, qc_name)
    missing_required = [
        cfg_name for cfg_name in REQUIRED_CFG_NAME_LIST
        if cfg_name not in complete_names
    ]
    complete_dedicate_names = [
        cfg_name for cfg_name in DEDICATE_CFG_NAME_LIST
        if cfg_name in complete_names
    ]

    if missing_required or not complete_dedicate_names:
        lines = [
            "Required fixed-d evaluation outputs are missing.",
            "Run script_d3rot_opt_evaluation_fixed_d.py first with the same arguments.",
            "",
        ]
        for cfg_name in missing_required:
            outdir = outdir_dict[qc_name][cfg_name]
            missing = [
                str(outdir / filename)
                for filename in REQUIRED_RESULT_FILES
                if not (outdir / filename).is_file()
            ]
            lines.append(f"- {cfg_name}:")
            lines.extend(f"  {path}" for path in missing)

        if not complete_dedicate_names:
            lines.append("- DIR_TOGL+DEDICATE_CELL1-3:")
            lines.append("  no complete dedicated-cell result was found")

        raise FileNotFoundError("\n".join(lines))

    skipped_dedicate_names = [
        cfg_name for cfg_name in DEDICATE_CFG_NAME_LIST
        if cfg_name not in complete_dedicate_names
    ]
    if skipped_dedicate_names:
        print(
            "[WARN] skipping incomplete dedicated-cell cfg(s): "
            + ", ".join(skipped_dedicate_names)
        )

    return [
        cfg_name for cfg_name in CFG_NAME_LIST
        if cfg_name in complete_names
    ]


def filter_by_cfg_names(
    qc_name: str,
    target_cfg_dict: dict,
    outdir_dict: dict[str, dict[str, Path]],
    cfg_names: list[str],
) -> tuple[dict, dict[str, dict[str, Path]]]:
    return (
        {cfg_name: target_cfg_dict[cfg_name] for cfg_name in cfg_names},
        {qc_name: {cfg_name: outdir_dict[qc_name][cfg_name] for cfg_name in cfg_names}},
    )


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def append_extension(output_prefix: Path, extension: str) -> Path:
    return output_prefix.with_name(output_prefix.name + extension)


def aggregate_cfg_row(
    qc_name: str,
    cfg_name: str,
    cfg_in: experiment_config,
    outdir: Path,
    code_dist: int,
    include_detail: bool = False,
) -> tuple[str, dict, dict | None]:
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

    qc_cfg_res = analyze_qc_cfg_from_exec_out_dict(qc_name, exec_out_dict, draw_graph=False)
    qc_cfg_err = error_qc_cfg_from_outputs(
        qc_name,
        cfg_in,
        exec_out_dict[RunOpt.IGNORE_NONE],
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

    detail_res = None
    if include_detail:
        detail_res = analyze_qc_cfg_from_exec_out_dict(qc_name, exec_out_dict, draw_graph=False)

    return cfg_name, row, detail_res


def collect_aggregate_df(
    qc_name: str,
    target_cfg_dict: dict[str, tuple[experiment_config, list[RunOpt]]],
    outdir_dict: dict[str, dict[str, Path]],
    code_dist: int,
    cfg_names: list[str],
    num_threads: int,
    detail_cfg_name: str | None,
) -> tuple[pd.DataFrame, dict | None]:
    worker_count = max(1, min(num_threads, len(cfg_names)))
    if worker_count == 1:
        row_by_cfg = {}
        detail_res = None
        for cfg_name in cfg_names:
            _, row, maybe_detail = aggregate_cfg_row(
                qc_name,
                cfg_name,
                target_cfg_dict[cfg_name][0],
                outdir_dict[qc_name][cfg_name],
                code_dist,
                include_detail=(cfg_name == detail_cfg_name),
            )
            row_by_cfg[cfg_name] = row
            if maybe_detail is not None:
                detail_res = maybe_detail
        rows = [row_by_cfg[cfg_name] for cfg_name in cfg_names]
        return pd.DataFrame(rows).reset_index(drop=True), detail_res

    row_by_cfg = {}
    detail_res = None
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_by_cfg = {
            cfg_name: executor.submit(
                aggregate_cfg_row,
                qc_name,
                cfg_name,
                target_cfg_dict[cfg_name][0],
                outdir_dict[qc_name][cfg_name],
                code_dist,
                cfg_name == detail_cfg_name,
            )
            for cfg_name in cfg_names
        }
        for cfg_name in cfg_names:
            _, row, maybe_detail = future_by_cfg[cfg_name].result()
            row_by_cfg[cfg_name] = row
            if maybe_detail is not None:
                detail_res = maybe_detail

    return pd.DataFrame([row_by_cfg[cfg_name] for cfg_name in cfg_names]).reset_index(drop=True), detail_res


def save_aggregate_graph(qc_name: str, aggregate_df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(10, 6))
    aggregate_df.plot(
        ax=ax,
        x="Label",
        y=[
            col for col in aggregate_df.columns
            if col not in ["Label", "Num_Layers", "Sum", "Logical_Error"]
        ],
        kind="bar",
        stacked=True,
        edgecolor="black",
        linewidth=1.0,
    )

    ax2 = ax.twinx()
    ax2.plot(
        aggregate_df["Label"],
        aggregate_df["Logical_Error"],
        color="black",
        marker="o",
    )

    ax.set_title(f"qc: {qc_name}")
    ax.set_ylabel("Average per-layer latency (us)")
    ax2.set_ylabel("Average per-layer logical error rate")
    ax2.set_yscale("log")
    ax.set_xlabel("Configuration name")
    fig.tight_layout()

    pdf_path = append_extension(output_prefix, ".pdf")
    png_path = append_extension(output_prefix, ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def save_detail_graph(qc_name: str, cfg_name: str, detail_df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    plot_df = detail_df.sort_values(by="Sum").reset_index(drop=True)
    latency_cols = [
        col for col in plot_df.columns
        if col not in ["Label", "Sum"]
    ]
    plot_df[latency_cols] = plot_df[latency_cols].clip(lower=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(
        ax=ax,
        x="Label",
        y=latency_cols,
        kind="area",
        stacked=True,
        linewidth=0,
    )
    ax.set_title(f"qc: {qc_name}, cfg: {cfg_name}")
    ax.set_ylabel("Latency (us)")
    ax.set_xlabel("Layer idx")
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
    base_dir: Path | None,
    detail_cfg_name: str,
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

    target_cfg_dict = make_target_cfg_dict(code_dist)
    outdir_dict = build_outdir_dict(qc_name, base_dir)

    print(f"reading fixed-d outputs from: {base_dir}")
    available_cfg_names = validate_result_files(outdir_dict, qc_name)
    target_cfg_dict, outdir_dict = filter_by_cfg_names(
        qc_name,
        target_cfg_dict,
        outdir_dict,
        available_cfg_names,
    )
    print("analyzing cfg(s): " + ", ".join(available_cfg_names))

    if detail_cfg_name not in available_cfg_names:
        fallback_detail_cfg_name = next(
            cfg_name for cfg_name in DEDICATE_CFG_NAME_LIST
            if cfg_name in available_cfg_names
        )
        print(
            "[WARN] detail cfg "
            f"{detail_cfg_name} is incomplete; using {fallback_detail_cfg_name} instead"
        )
        detail_cfg_name = fallback_detail_cfg_name

    aggregate_df, detail_res = collect_aggregate_df(
        qc_name,
        target_cfg_dict,
        outdir_dict,
        code_dist,
        available_cfg_names,
        num_threads,
        detail_cfg_name,
    )
    aggregate_prefix = base_dir / f"{qc_name}_d3rot_analysis_d{code_dist}"
    aggregate_csv_path = append_extension(aggregate_prefix, ".csv")
    aggregate_df.to_csv(aggregate_csv_path, index=False)
    aggregate_pdf_path, aggregate_png_path = save_aggregate_graph(
        qc_name,
        aggregate_df,
        aggregate_prefix,
    )

    if detail_res is None:
        raise RuntimeError(f"detail analysis result was not produced for {detail_cfg_name}")
    detail_df = pd.DataFrame(detail_res).reset_index(drop=True)
    detail_prefix = (
        base_dir
        / f"{qc_name}_d3rot_analysis_{safe_stem(detail_cfg_name)}_layers_d{code_dist}"
    )
    detail_csv_path = append_extension(detail_prefix, ".csv")
    detail_df.to_csv(detail_csv_path, index=False)
    detail_pdf_path, detail_png_path = save_detail_graph(
        qc_name,
        detail_cfg_name,
        detail_df,
        detail_prefix,
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\nAnalysis - with Logical error rate")
    print(aggregate_df)
    print(f"\nsaved aggregate graph: {aggregate_pdf_path}")
    print(f"saved aggregate graph: {aggregate_png_path}")
    print(f"saved aggregate table: {aggregate_csv_path}")

    print(f"\nPer-layer analysis for {detail_cfg_name}")
    print(detail_df)
    print(f"\nsaved detail graph: {detail_pdf_path}")
    print(f"saved detail graph: {detail_png_path}")
    print(f"saved detail table: {detail_csv_path}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        ham_name=args.ham_name,
        code_dist=args.code_distance,
        use_naive_mapping=args.use_naive_mapping,
        use_precomputed_mapping=args.use_precomputed_mapping,
        base_dir=args.base_dir,
        detail_cfg_name=args.detail_cfg_name,
        num_threads=args.num_threads,
    )
