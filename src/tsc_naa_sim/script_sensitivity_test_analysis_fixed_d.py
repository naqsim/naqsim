from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gmean


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
from misc import getJsonData
from run_analysis_scripts import get_finish_us_from_uop_schedule, load_zst


HWCFG_NAME_LIST = [
    "sensitivity_hwcfg_togl1_trf40.json",
    "sensitivity_hwcfg_togl10_trf40.json",
    "sensitivity_hwcfg_togl1_trf120.json",
    "sensitivity_hwcfg_togl10_trf120.json",
]
SCHEME_ORDER = [
    "Base",
    "D3ROT",
]
SUB_ORDER = [
    (10, 120),
    (1, 120),
    (10, 40),
    (1, 40),
]
AOD_ORDER = [
    "2",
    "4",
    "8",
    "inf",
]
REQUIRED_RESULT_FILES = [
    "exec_out_IGNORE_NONE.zst",
]


def normalize_ham_name(ham_name: str) -> str:
    ham_name = os.path.basename(ham_name)
    if ham_name.endswith(".qasm"):
        return ham_name[:-len(".qasm")]
    return ham_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Analysis section from sensitivity_test_revision.ipynb "
            "using saved fixed-d outputs."
        )
    )
    parser.add_argument(
        "--ham-name",
        type=str,
        default="SELECT_10_Heisenberg1D_OBC_1_0",
        help="Target hamiltonian name used by script_sensitivity_test_evaluation_fixed_d.py.",
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
        help="Code distance used by script_sensitivity_test_evaluation_fixed_d.py.",
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
        "--norm-cfg-name",
        type=str,
        default="D3ROT_CELL3_togl1_trf40_aodinf",
        help="Configuration name used as the normalization baseline.",
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
        / "sensitivity_test_evaluation_fixed_d"
        / f"{mapping_prefix}{ham_name}_distance{code_dist}"
    )


def make_target_cfg_dict(code_dist: int) -> dict[str, tuple[experiment_config, list[RunOpt]]]:
    target_cfg_dict = {}
    run_opts = [RunOpt.IGNORE_NONE]

    # Two architecture configs:
    arch_cfg_list = []

    ## 1. Base
    base_cfg_name = "Base"
    base_cfg = experiment_config()
    base_cfg.code_dist = code_dist
    base_cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    arch_cfg_list.append((base_cfg_name, base_cfg))

    ## 2. D3ROT
    num_rot_cell = 3 # FIXME
    d3rot_cfg_name = f"D3ROT_CELL{num_rot_cell}"
    d3rot_cfg = experiment_config()
    d3rot_cfg.code_dist= code_dist
    d3rot_cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    d3rot_cfg.rot_sched_opt = RotSchedOpt.DISTRIBUTE
    d3rot_cfg.cell_size = CellSize.SMALLEST
    d3rot_cfg.rot_type = RotType.DIR_TOGL
    d3rot_cfg.refl_type_h = None
    d3rot_cfg.refl_type_d = None
    d3rot_cfg.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    d3rot_cfg.num_rot_cell = num_rot_cell
    d3rot_cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    arch_cfg_list.append((d3rot_cfg_name, d3rot_cfg))

    # Four (two SLM on/off x two AOD pick/drop) hwcfg files:
    for ((arch_cfg_name, arch_cfg), hwcfg_name) in product(arch_cfg_list, HWCFG_NAME_LIST):
        # name set
        temp_name = arch_cfg_name
        temp_name += '_'
        hwcfg_fn, _ = os.path.splitext(hwcfg_name)
        temp_name += '_'.join(hwcfg_fn.split('_')[-2:])

        # arch_cfg
        temp_cfg = deepcopy(arch_cfg)

        # hwcfg
        temp_cfg.hw_cfg = getJsonData(hwcfg_name)

        # Four # AOD configurations for base and d3rot
        ## Num aods in total: 2, 4, 8, inf
        ## How to separate? rot_type: REFL vs. DIR_TOGL
        if temp_cfg.rot_type == RotType.REFL: # Base
            ## 1. REFL (i.e., for Base)
            # 1-1. (# AODH, # AODD) = (1, 1)
            cfg_name = temp_name + '_aod2'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = False
            cfg_in.skip_uop_grouping = False
            cfg_in.num_aodh_max = 1
            cfg_in.num_aodd_max = 1
            cfg_in.num_aodr_max = 1 # not used
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)

            # 1-2. (# AODH, # AODD) = (2, 2)
            cfg_name = temp_name + '_aod4'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = False
            cfg_in.skip_uop_grouping = False
            cfg_in.num_aodh_max = 2
            cfg_in.num_aodd_max = 2
            cfg_in.num_aodr_max = 1 # not used
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)

            # 1-3. (# AODH, # AODD) = (4, 4)
            cfg_name = temp_name + '_aod8'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = False
            cfg_in.skip_uop_grouping = False
            cfg_in.num_aodh_max = 4
            cfg_in.num_aodd_max = 4
            cfg_in.num_aodr_max = 1 # not used
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)

            # 1-4. (# AODH, # AODD) = (inf, inf)
            cfg_name = temp_name + '_aodinf'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = True
            cfg_in.skip_uop_grouping = True
            cfg_in.num_aodh_max = None
            cfg_in.num_aodd_max = None
            cfg_in.num_aodr_max = None
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)

        elif temp_cfg.rot_type == RotType.DIR_TOGL: # D3ROT
            ## 2. DIR_TOGL (i.e., for D3ROT) (NOTE: # AODR = 2 will be sufficient)
            # 2-1. (# AODH, # AODR) = (1, 1)
            cfg_name = temp_name + '_aod2'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = False
            cfg_in.skip_uop_grouping = False
            cfg_in.num_aodh_max = 1
            cfg_in.num_aodd_max = 1 # not used
            cfg_in.num_aodr_max = 1
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)

            # 2-2. (# AODH, # AODR) = (2, 2)
            cfg_name = temp_name + '_aod4'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = False
            cfg_in.skip_uop_grouping = False
            cfg_in.num_aodh_max = 2
            cfg_in.num_aodd_max = 1 # not used
            cfg_in.num_aodr_max = 2
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)

            # 2-3. (# AODH, # AODR) = (6, 2)
            cfg_name = temp_name + '_aod8'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = False
            cfg_in.skip_uop_grouping = False
            cfg_in.num_aodh_max = 6
            cfg_in.num_aodd_max = 1 # not used
            cfg_in.num_aodr_max = 2
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)

            # 2-4. (AODH, # AODR) = (inf, inf)
            cfg_name = temp_name + '_aodinf'
            cfg_in = deepcopy(temp_cfg)
            cfg_in.is_aod_infinite = True
            cfg_in.skip_uop_grouping = True
            cfg_in.num_aodh_max = None
            cfg_in.num_aodd_max = None
            cfg_in.num_aodr_max = None
            target_cfg_dict[cfg_name] = (cfg_in, run_opts)
        else:
            raise Exception()

    return target_cfg_dict


def build_outdir_dict(ham_name: str, base_dir: Path) -> dict[str, dict[str, Path]]:
    target_cfg_dict = make_target_cfg_dict(code_dist=1)
    return {
        ham_name: {
            cfg_name: base_dir / f"{ham_name}_{cfg_name}"
            for cfg_name in target_cfg_dict.keys()
        }
    }


def cfg_name_list() -> list[str]:
    return list(make_target_cfg_dict(code_dist=1).keys())


def is_cfg_complete(outdir: Path) -> bool:
    return all((outdir / filename).is_file() for filename in REQUIRED_RESULT_FILES)


def complete_cfg_names(outdir_dict: dict[str, dict[str, Path]], qc_name: str) -> list[str]:
    return [
        cfg_name for cfg_name in cfg_name_list()
        if is_cfg_complete(outdir_dict[qc_name][cfg_name])
    ]


def validate_result_files(
    outdir_dict: dict[str, dict[str, Path]],
    qc_name: str,
    norm_cfg_name: str,
) -> list[str]:
    complete_names = complete_cfg_names(outdir_dict, qc_name)
    if not complete_names:
        lines = [
            "Required fixed-d evaluation outputs are missing.",
            "Run script_sensitivity_test_evaluation_fixed_d.py first with the same arguments.",
            "",
        ]
        lines.extend(
            f"- {cfg_name}: {outdir_dict[qc_name][cfg_name] / REQUIRED_RESULT_FILES[0]}"
            for cfg_name in cfg_name_list()
        )
        raise FileNotFoundError("\n".join(lines))

    if norm_cfg_name not in complete_names:
        outdir = outdir_dict[qc_name].get(norm_cfg_name)
        missing_path = outdir / REQUIRED_RESULT_FILES[0] if outdir else norm_cfg_name
        raise FileNotFoundError(
            "Normalization cfg output is missing. "
            "Run script_sensitivity_test_evaluation_fixed_d.py first with the same arguments.\n"
            f"- {norm_cfg_name}: {missing_path}"
        )

    skipped_cfg_names = [
        cfg_name for cfg_name in cfg_name_list()
        if cfg_name not in complete_names
    ]
    if skipped_cfg_names:
        print(
            "[WARN] skipping incomplete cfg(s): "
            + ", ".join(skipped_cfg_names)
        )

    return [
        cfg_name for cfg_name in cfg_name_list()
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


def analyze_qc_cfg_svt(outdir: Path) -> dict:
    exec_out = load_zst(str(outdir / f"exec_out_{RunOpt.IGNORE_NONE.name}"))
    uop_schedule_trace, _, _ = exec_out
    label_list = []
    finish_list = []
    for idx, uop_schedule in enumerate(uop_schedule_trace):
        label_list.append(f"L{idx}")
        finish_list.append(get_finish_us_from_uop_schedule(uop_schedule))
    return {
        "Label": label_list,
        "Finish": finish_list,
        "Sum": finish_list,
    }


def aggregate_cfg_row(
    qc_name: str,
    cfg_name: str,
    cfg_in: experiment_config,
    outdir: Path,
    code_dist: int,
    include_detail: bool = False,
) -> tuple[str, dict, dict | None]:
    qc_cfg_res = analyze_qc_cfg_svt(outdir)

    row = {
        "cfg_name": cfg_name,
        "Num_Layers": len(qc_cfg_res["Sum"]),
        qc_name: sum(qc_cfg_res["Sum"]),
    }

    detail_res = None
    if include_detail:
        detail_res = qc_cfg_res

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
    if not num_threads:
        num_threads = 1
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


def make_normalized_df(aggregate_df: pd.DataFrame, qc_name: str, norm_cfg_name: str) -> pd.DataFrame:
    res_df = aggregate_df[["cfg_name", qc_name]].set_index("cfg_name")
    res_df_norm = res_df.div(res_df.loc[norm_cfg_name])
    res_df_norm["geomean"] = gmean(res_df_norm, axis=1)
    res_df_norm = res_df_norm.reset_index()

    # Parse and sort using the same order as sensitivity_test_revision.ipynb.
    sort_df = res_df_norm.copy()
    sort_df["scheme"] = sort_df["cfg_name"].str.extract(r"^(Base|D3ROT)")
    sort_df["togl"] = sort_df["cfg_name"].str.extract(r"_togl(\d+)_").astype(int)
    sort_df["trf"] = sort_df["cfg_name"].str.extract(r"_trf(\d+)_").astype(int)
    sort_df["aod"] = sort_df["cfg_name"].str.extract(r"_aod(\d+|inf)$")[0]

    sub_rank = {v: i for i, v in enumerate(SUB_ORDER)}
    scheme_rank = {v: i for i, v in enumerate(SCHEME_ORDER)}
    aod_rank = {v: i for i, v in enumerate(AOD_ORDER)}

    sort_df["sub_rank"] = list(zip(sort_df["togl"], sort_df["trf"]))
    sort_df["sub_rank"] = sort_df["sub_rank"].map(sub_rank)
    sort_df["scheme_rank"] = sort_df["scheme"].map(scheme_rank)
    sort_df["aod_rank"] = sort_df["aod"].map(aod_rank)

    sort_df = sort_df.sort_values(
        ["sub_rank", "scheme_rank", "aod_rank"]
    ).reset_index(drop=True)
    return sort_df[["cfg_name", qc_name, "geomean"]]


def save_aggregate_graph(qc_name: str, aggregate_df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(12, 4))
    aggregate_df.plot(
        ax=ax,
        x="cfg_name",
        y=qc_name,
        kind="bar",
        legend=False,
        edgecolor="black",
        linewidth=1.0,
        color="#4f8a8b",
    )
    ax.set_title(f"qc: {qc_name}")
    ax.set_ylabel("Total latency (us)")
    ax.set_xlabel("Configuration name")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()

    pdf_path = append_extension(output_prefix, ".pdf")
    png_path = append_extension(output_prefix, ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def save_normalized_graph(qc_name: str, normalized_df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(12, 4))

    bars = ax.bar(
        range(len(normalized_df)),
        normalized_df["geomean"],
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=1,
    )

    ax.set_xticks(range(len(normalized_df)))
    ax.set_xticklabels(normalized_df["cfg_name"], rotation=90)
    ax.set_ylabel("Slowdown (vs. final result)")
    ax.set_xlabel("Configuration name")
    ax.set_title(f"qc: {qc_name}")

    fig.tight_layout()

    pdf_path = append_extension(output_prefix, ".pdf")
    png_path = append_extension(output_prefix, ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def save_detail_graph(qc_name: str, cfg_name: str, detail_df: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    plot_df = detail_df.sort_values(by="Sum").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(
        ax=ax,
        x="Label",
        y=["Finish"],
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
    norm_cfg_name: str,
    num_threads: int,
) -> None:
    ham_name = normalize_ham_name(ham_name)
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
    if norm_cfg_name not in target_cfg_dict:
        raise ValueError(
            f"unknown --norm-cfg-name {norm_cfg_name}. "
            f"Choose one of: {', '.join(target_cfg_dict.keys())}"
        )

    outdir_dict = build_outdir_dict(qc_name, base_dir)

    print(f"reading fixed-d outputs from: {base_dir}")
    available_cfg_names = validate_result_files(outdir_dict, qc_name, norm_cfg_name)
    target_cfg_dict, outdir_dict = filter_by_cfg_names(
        qc_name,
        target_cfg_dict,
        outdir_dict,
        available_cfg_names,
    )
    print("analyzing cfg(s): " + ", ".join(available_cfg_names))

    aggregate_df, detail_res = collect_aggregate_df(
        qc_name,
        target_cfg_dict,
        outdir_dict,
        code_dist,
        available_cfg_names,
        num_threads,
        norm_cfg_name,
    )
    aggregate_prefix = base_dir / f"{qc_name}_sensitivity_test_analysis_d{code_dist}"
    aggregate_csv_path = append_extension(aggregate_prefix, ".csv")
    aggregate_df.to_csv(aggregate_csv_path, index=False)
    aggregate_pdf_path, aggregate_png_path = save_aggregate_graph(
        qc_name,
        aggregate_df,
        aggregate_prefix,
    )

    normalized_df = make_normalized_df(aggregate_df, qc_name, norm_cfg_name)
    normalized_prefix = base_dir / f"{qc_name}_sensitivity_test_analysis_normalized_d{code_dist}"
    normalized_csv_path = append_extension(normalized_prefix, ".csv")
    normalized_df.to_csv(normalized_csv_path, index=False)
    normalized_pdf_path, normalized_png_path = save_normalized_graph(
        qc_name,
        normalized_df,
        normalized_prefix,
    )

    if detail_res is None:
        raise RuntimeError(f"detail analysis result was not produced for {norm_cfg_name}")
    detail_df = pd.DataFrame(detail_res).reset_index(drop=True)
    detail_prefix = (
        base_dir
        / f"{qc_name}_sensitivity_test_analysis_{safe_stem(norm_cfg_name)}_layers_d{code_dist}"
    )
    detail_csv_path = append_extension(detail_prefix, ".csv")
    detail_df.to_csv(detail_csv_path, index=False)
    detail_pdf_path, detail_png_path = save_detail_graph(
        qc_name,
        norm_cfg_name,
        detail_df,
        detail_prefix,
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\nAnalysis - test one qc for all cfgs")
    print(aggregate_df)
    print(f"\nsaved aggregate graph: {aggregate_pdf_path}")
    print(f"saved aggregate graph: {aggregate_png_path}")
    print(f"saved aggregate table: {aggregate_csv_path}")

    print(f"\nNormalized analysis using {norm_cfg_name}")
    print(normalized_df)
    print(f"\nsaved normalized graph: {normalized_pdf_path}")
    print(f"saved normalized graph: {normalized_png_path}")
    print(f"saved normalized table: {normalized_csv_path}")

    print(f"\nPer-layer analysis for {norm_cfg_name}")
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
        norm_cfg_name=args.norm_cfg_name,
        num_threads=args.num_threads,
    )
