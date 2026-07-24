#!/usr/bin/env python3
"""Shared CSV aggregation and matplotlib rendering for the AE entry points."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


AE_DIR = Path(__file__).resolve().parents[1]
TSC_DIR = AE_DIR.parent
WORKLOAD_CSV = AE_DIR / "config" / "workloads.csv"

BREAKDOWN_COLUMNS = [
    "ESM",
    "Others",
    "Move",
    "Route_Conflict (Move)",
    "Route_Conflict (Rot)",
    "Rotation",
]
PLOT_BREAKDOWN_COLUMNS = ["ESM", "Others", "Move", "Route conflict", "Rotation"]
BREAKDOWN_COLORS = {
    "ESM": "#4C78A8",
    "Others": "#F58518",
    "Move": "#A7A7A7",
    "Route conflict": "#F2C14E",
    "Rotation": "#72A0C1",
}

TRANSVERSAL_DISTANCE_COLUMN = "transversal_distance"


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    ham_name: str
    logical_qubits: int
    loop_count: int
    distance: int
    ls_distance: int
    fig19: bool


def parse_args(target: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Aggregate current-format raw data and render {target}."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Input root containing current-format CSV/PKL results. Repeat for multiple "
            "roots. Defaults to raw_data/<target>, followed by repository outputs and "
            "known scp directories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AE_DIR / target / "results",
        help="Directory for the raw-data CSV and PDF/PNG figure.",
    )
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="Workload key, label, or full ham_name. Repeat to select several.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Skip unavailable workloads instead of failing.",
    )
    return parser.parse_args()


def default_input_roots(target: str) -> list[Path]:
    roots = [
        AE_DIR / "raw_data" / target,
        AE_DIR / "quration" / "results",
        TSC_DIR / "output" / "comp_opt_evaluation_fixed_d",
        TSC_DIR / "output" / "d3rot_opt_evaluation_fixed_d",
        TSC_DIR / "output" / "final_evaluation_fixed_d",
        TSC_DIR / "output" / "sensitivity_test_evaluation_fixed_d",
        TSC_DIR / "output" / "hs_skip_evaluation_fixed_d",
        TSC_DIR / "output" / "ms_overhead_fixed_d",
        TSC_DIR / "output" / "ms_overhead_fixed_d_geomean",
        TSC_DIR / "work" / "quration_eval_result",
        TSC_DIR / "work" / "quration_trace_ms_overhead",
        Path("/home/youte/comp_opt_res/comp_opt_csv"),
        Path("/home/youte/comp_opt_res/csv"),
        Path("/home/youte/d3rot_opt_res/d3rot_csv"),
    ]
    return [root for root in roots if root.exists()]


def load_workloads(target: str, selectors: list[str]) -> list[Workload]:
    workloads: list[Workload] = []
    with WORKLOAD_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            candidates = {row["key"], row["label"], row["ham_name"]}
            if target == "fig19" and row["fig19"] != "1":
                continue
            if selectors and not any(selector in candidates for selector in selectors):
                continue
            workloads.append(
                Workload(
                    key=row["key"],
                    label=row["label"],
                    ham_name=row["ham_name"],
                    logical_qubits=int(row["logical_qubits"]),
                    loop_count=int(row["loop_count"]),
                    distance=int(row[TRANSVERSAL_DISTANCE_COLUMN]),
                    ls_distance=int(row["ls_distance"]),
                    fig19=row["fig19"] == "1",
                )
            )
    if selectors:
        matched = {value for item in workloads for value in (item.key, item.label, item.ham_name)}
        missing = [selector for selector in selectors if selector not in matched]
        if missing:
            raise ValueError(f"Unknown or unavailable selector(s) for {target}: {missing}")
    if not workloads:
        raise ValueError(f"No workloads selected for {target}")
    return workloads


def usable_path(path: Path) -> bool:
    return "with_error_model_bug" not in path.parts


def find_named_file(roots: list[Path], filename: str) -> Path | None:
    for root in roots:
        if root.is_file() and root.name == filename and usable_path(root):
            return root
        direct = root / filename
        if direct.is_file() and usable_path(direct):
            return direct
        if root.is_dir():
            matches = sorted(path for path in root.rglob(filename) if usable_path(path))
            if matches:
                return matches[0]
    return None


def require_named_file(roots: list[Path], filename: str) -> Path:
    path = find_named_file(roots, filename)
    if path is None:
        searched = "\n".join(f"- {root}" for root in roots)
        raise FileNotFoundError(f"Could not find {filename}. Searched:\n{searched}")
    return path


def geomean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array) or np.any(array <= 0):
        return math.nan
    return float(np.exp(np.log(array).mean()))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def normalized_breakdown(row: pd.Series) -> dict[str, float]:
    values = {
        "ESM": float(row.get("ESM", 0.0)),
        "Others": float(row.get("Others", 0.0)),
        "Move": float(row.get("Move", 0.0)),
        "Route conflict": float(row.get("Route_Conflict (Move)", 0.0))
        + float(row.get("Route_Conflict (Rot)", 0.0)),
        "Rotation": float(row.get("Rotation", 0.0)),
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Breakdown total must be positive")
    return {key: value / total for key, value in values.items()}


def load_comp_aggregate(roots: list[Path], workload: Workload) -> pd.DataFrame:
    filename = f"{workload.ham_name}_case_study_1_analysis_d{workload.distance}.csv"
    return pd.read_csv(require_named_file(roots, filename), index_col=0)


def load_comp_layers(roots: list[Path], workload: Workload, cfg: str) -> pd.DataFrame:
    filename = (
        f"{workload.ham_name}_case_study_1_analysis_{cfg}_layers_d"
        f"{workload.distance}.csv"
    )
    return pd.read_csv(require_named_file(roots, filename))


def load_d3_aggregate(roots: list[Path], workload: Workload) -> pd.DataFrame:
    filename = f"{workload.ham_name}_d3rot_analysis_d{workload.distance}.csv"
    return pd.read_csv(require_named_file(roots, filename)).set_index("Label")


def load_final_table(roots: list[Path], workload: Workload) -> pd.DataFrame:
    filename = f"{workload.ham_name}_naive_mapping_d_{workload.distance}.csv"
    return pd.read_csv(require_named_file(roots, filename)).set_index("Label")


def selected_or_skip(
    workloads: list[Workload],
    loader,
    allow_partial: bool,
) -> list[tuple[Workload, object]]:
    loaded = []
    for workload in workloads:
        try:
            value = loader(workload)
        except FileNotFoundError as exc:
            if not allow_partial:
                raise
            print(f"warning: skipping {workload.label}: {exc}", file=sys.stderr)
            continue
        loaded.append((workload, value))
    if not loaded:
        raise FileNotFoundError("No selected workload data was available")
    return loaded


def analyze_fig11a(args, roots: list[Path], workloads: list[Workload]) -> None:
    loaded = selected_or_skip(
        workloads,
        lambda workload: load_comp_layers(roots, workload, "Baseline"),
        args.allow_partial,
    )
    rows = []
    for workload, frame in loaded:
        mean_row = frame[BREAKDOWN_COLUMNS].mean(numeric_only=True)
        breakdown = normalized_breakdown(mean_row)
        rows.append({"workload": workload.label, "ham_name": workload.ham_name,
                     "code_distance": workload.distance, **breakdown})
    average = {key: fmean(row[key] for row in rows) for key in PLOT_BREAKDOWN_COLUMNS}
    plot_rows = rows + [{"workload": "Average", "ham_name": "Average",
                         "code_distance": "", **average}]
    output = pd.DataFrame(plot_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "fig11a_raw_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(6.5, 0.85 * len(output)), 3.6))
    x = np.arange(len(output))
    bottom = np.zeros(len(output))
    for key in PLOT_BREAKDOWN_COLUMNS:
        values = output[key].to_numpy(float)
        ax.bar(x, values, bottom=bottom, color=BREAKDOWN_COLORS[key],
               edgecolor="black", linewidth=0.6, label=key)
        bottom += values
    ax.set_ylabel("Normalized layer latency")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, output["workload"], rotation=40, ha="right")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, args.output_dir, "fig11a")


TABLE3_CONFIGS = ["Baseline", "Skip", "Skip+Aggr", "Skip+Dist"]
TABLE3_CONFIG_LABELS = {
    "Baseline": "Baseline", "Skip": "Skip", "Skip+Aggr": "Skip+Aggr.",
    "Skip+Dist": "Skip+Dist.",
}


def analyze_table3(args, roots: list[Path], workloads: list[Workload]) -> None:
    def loader(workload):
        aggregate = load_comp_aggregate(roots, workload)
        layers = {cfg: load_comp_layers(roots, workload, cfg) for cfg in TABLE3_CONFIGS}
        return aggregate, layers

    loaded = selected_or_skip(workloads, loader, args.allow_partial)
    metric_specs = [
        ("Layers with rotation (%)", "Rotation layer occupancy (%)", False, "arith", False),
        ("Rotation count (%)", "Rotation counts", True, "geo", False),
        ("Execution time (%)", "Average layer latency (us)", True, "geo", True),
        ("Max. per-layer rotations", "Max rotations per layer", False, "arith", False),
    ]
    rows = []
    for metric_label, source, normalized, avg_kind, use_layers in metric_specs:
        for cfg in TABLE3_CONFIGS:
            values = []
            row = {"Metric": metric_label, "Optimization": TABLE3_CONFIG_LABELS[cfg]}
            for workload, (aggregate, layers) in loaded:
                if use_layers:
                    value = float(layers[cfg]["Sum"].sum())
                    baseline = float(layers["Baseline"]["Sum"].sum())
                else:
                    value = float(aggregate.loc[source, cfg])
                    baseline = float(aggregate.loc[source, "Baseline"])
                if normalized:
                    value = 100.0 * value / baseline
                row[workload.label] = value
                values.append(value)
            row["Average"] = fmean(values) if avg_kind == "arith" else geomean(values)
            rows.append(row)
    output = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "table3_raw_data.csv", index=False)

    display = output.copy()
    for column in display.columns[2:]:
        display[column] = display[column].map(lambda value: f"{value:.1f}")
    fig_height = max(4.5, 0.30 * len(display) + 1.2)
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * len(display.columns)), fig_height))
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    ax.set_title("Compiler optimization results", pad=10)
    fig.tight_layout()
    save_figure(fig, args.output_dir, "table3")


FIG15_CONFIGS = [
    "REFL_SE", "REFL_TE", "DIR_CHANGE", "DIR_TOGL",
    "DIR_TOGL+DEDICATE_CELL3", "DIR_IDEAL",
]
FIG15_LABELS = ["REFL_SE", "REFL_TE", "DIR_CHANGE", "DIR_TOGL", "D3-ROT", "DIR_IDEAL"]
FIG15_FOOTPRINT = [1.0, 2.25, 2.0, 2.0, 1.22, 1.0]


def analyze_fig15(args, roots: list[Path], workloads: list[Workload]) -> None:
    loaded = selected_or_skip(
        workloads, lambda workload: load_d3_aggregate(roots, workload), args.allow_partial
    )
    numeric_columns = [*BREAKDOWN_COLUMNS, "Sum", "Logical_Error"]
    averaged = sum(frame.loc[FIG15_CONFIGS, numeric_columns] for _, frame in loaded) / len(loaded)
    averaged = averaged.reset_index().rename(columns={"Label": "config", "index": "config"})
    averaged.insert(1, "plot_label", FIG15_LABELS)
    averaged["Qubit footprint"] = FIG15_FOOTPRINT
    averaged["Route conflict"] = (
        averaged["Route_Conflict (Move)"] + averaged["Route_Conflict (Rot)"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    averaged.to_csv(args.output_dir / "fig15_raw_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax_ler = ax.twinx()
    x = np.arange(len(averaged))
    bottom = np.zeros(len(averaged))
    for key in PLOT_BREAKDOWN_COLUMNS:
        values = averaged[key].to_numpy(float)
        ax.bar(x, values, bottom=bottom, color=BREAKDOWN_COLORS[key],
               edgecolor="black", linewidth=0.6, label=key)
        bottom += values
    ax_ler.plot(x, averaged["Logical_Error"], color="#3B73B9", marker="o",
                linewidth=1.2, label="LER")
    ax_ler.set_yscale("log")
    labels = [f"{label}\n({footprint:g})" for label, footprint in
              zip(averaged["plot_label"], averaged["Qubit footprint"])]
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Average per-layer latency (µs)")
    ax_ler.set_ylabel("Average per-layer LER")
    handles, labels_left = ax.get_legend_handles_labels()
    handle_r, label_r = ax_ler.get_legend_handles_labels()
    ax.legend(handles + handle_r, labels_left + label_r, frameon=False,
              fontsize=8, ncols=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, args.output_dir, "fig15")


def rectangle(n: int, max_diff: int = 2) -> tuple[int, int]:
    best = None
    for width in range(1, n + 1):
        height = math.ceil(n / width)
        if abs(width - height) <= max_diff:
            area = width * height
            candidate = (area, abs(width - height), max(width, height), min(width, height))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        width = int(math.sqrt(n))
        while width > 1 and math.ceil(n / width) - width > max_diff:
            width += 1
        return max(width, math.ceil(n / width)), min(width, math.ceil(n / width))
    return best[2], best[3]


def d3_footprint(workload: Workload) -> float:
    width, height = rectangle(workload.logical_qubits)
    baseline_area = width * height
    d3_area = (width - 3 + math.sqrt(2) * 3) * (height - 1 + math.sqrt(2))
    return d3_area / baseline_area


def analyze_fig16(args, roots: list[Path], workloads: list[Workload]) -> None:
    loaded = selected_or_skip(
        workloads, lambda workload: load_final_table(roots, workload), args.allow_partial
    )
    raw_rows = []
    normalized_rows = []
    current_format_rows = []
    footprint_rows = []
    config_labels = ["Baseline", "+ opt. #1", "+ opt. #2"]
    display_labels = ["Baseline", "Comp. Opt.", "D3-ROT"]
    for workload, frame in loaded:
        frame = frame.loc[config_labels]
        baseline = float(frame.loc["Baseline", "Sum"])
        for cfg, display in zip(config_labels, display_labels):
            row = frame.loc[cfg]
            raw = {"workload": workload.label, "ham_name": workload.ham_name,
                   "code_distance": workload.distance, "config": display}
            raw.update({column: float(row[column]) for column in frame.columns if
                        column not in {"Num_Layers"}})
            raw["Num_Layers"] = int(row["Num_Layers"])
            raw_rows.append(raw)
            normalized_split = {
                column: float(row[column]) / baseline for column in BREAKDOWN_COLUMNS
            }
            current_format_rows.append({
                "ham_name": workload.ham_name,
                "code_dist": workload.distance,
                "config": cfg,
                **normalized_split,
            })
            footprint = d3_footprint(workload) if display == "D3-ROT" else 1.0
            footprint_rows.append({
                "ham_name": workload.ham_name,
                "code_dist": workload.distance,
                "config": cfg,
                "qubit_footprint_overhead": footprint,
            })
            normalized = {
                "ESM": normalized_split["ESM"],
                "Others": normalized_split["Others"],
                "Move": normalized_split["Move"],
                "Route conflict": normalized_split["Route_Conflict (Move)"]
                + normalized_split["Route_Conflict (Rot)"],
                "Rotation": normalized_split["Rotation"],
            }
            normalized_rows.append({
                "workload": workload.label, "ham_name": workload.ham_name,
                "code_distance": workload.distance, "config": display,
                **normalized,
                "Qubit footprint": footprint,
            })
    normalized_df = pd.DataFrame(normalized_rows)
    average_rows = []
    for display in display_labels:
        group = normalized_df[normalized_df["config"] == display]
        average_rows.append({"workload": "Average", "ham_name": "Average",
                             "code_distance": "", "config": display,
                             **{key: fmean(group[key]) for key in PLOT_BREAKDOWN_COLUMNS},
                             "Qubit footprint": fmean(group["Qubit footprint"])})
    normalized_df = pd.concat([normalized_df, pd.DataFrame(average_rows)], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(args.output_dir / "fig16_workload_data.csv", index=False)
    normalized_df.to_csv(args.output_dir / "fig16_raw_data.csv", index=False)
    pd.DataFrame(current_format_rows).to_csv(
        args.output_dir / "fig16_execution_breakdown.csv", index=False
    )
    pd.DataFrame(footprint_rows).to_csv(
        args.output_dir / "fig16_footprint.csv", index=False
    )

    group_order = [workload.label for workload, _ in loaded] + ["Average"]
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(group_order)), 4.2))
    ax_fp = ax.twinx()
    bar_width = 0.24
    x = np.arange(len(group_order))
    for cfg_index, display in enumerate(display_labels):
        group = normalized_df[normalized_df["config"] == display].set_index("workload").loc[group_order]
        positions = x + (cfg_index - 1) * bar_width
        bottom = np.zeros(len(group))
        for key in PLOT_BREAKDOWN_COLUMNS:
            values = group[key].to_numpy(float)
            ax.bar(positions, values, width=bar_width, bottom=bottom,
                   color=BREAKDOWN_COLORS[key], edgecolor="black", linewidth=0.45,
                   label=key if cfg_index == 0 else None)
            bottom += values
    footprint_offsets = (np.arange(len(display_labels)) - 1) * bar_width
    for workload_index, workload_label in enumerate(group_order):
        footprint = (
            normalized_df[normalized_df["workload"] == workload_label]
            .set_index("config")
            .loc[display_labels, "Qubit footprint"]
            .to_numpy(float)
        )
        ax_fp.plot(
            x[workload_index] + footprint_offsets,
            footprint,
            color="#4C9A57",
            marker="o",
            linewidth=0.8,
            markersize=3,
            label="Footprint" if workload_index == 0 else None,
        )
    ax.set_xticks(x, group_order, rotation=35, ha="right")
    ax.set_ylabel("Execution time (normalized to Baseline)")
    ax_fp.set_ylabel("Qubit array footprint (normalized to Baseline)")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax_fp.set_ylim(0.6, 1.4)
    ax_fp.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left")
    ax_fp.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, args.output_dir, "fig16")


def logical_error_by_distance(distance: int) -> float:
    known = {3: 0.0002811373, 5: 4.87791e-05, 7: 6.0007e-06,
             9: 6.096e-07, 11: 5.72e-08, 13: 4e-09}
    if distance in known:
        return known[distance]
    return float(np.exp(-1.1190551249171585 * distance - 4.456335115474365))


def analyze_fig17(args, roots: list[Path], workloads: list[Workload]) -> None:
    final_loaded = selected_or_skip(
        workloads, lambda workload: load_final_table(roots, workload), args.allow_partial
    )
    ls_path = require_named_file(roots, "quration_eval_summary.csv")
    ls_summary = pd.read_csv(ls_path).set_index("ham_name")
    duration_rows = []
    ler_rows = []
    for workload, frame in final_loaded:
        if workload.ham_name not in ls_summary.index:
            if args.allow_partial:
                print(f"warning: no Quration row for {workload.ham_name}", file=sys.stderr)
                continue
            raise ValueError(f"No Quration row for {workload.ham_name} in {ls_path}")
        ls = ls_summary.loc[workload.ham_name]
        # Lattice surgery has its own Table-2 distance column.
        ls_distance = workload.ls_distance
        sc_duration_us = float(ls["beats"]) * ls_distance * 1.0
        na_ls_duration_us = float(ls["beats"]) * ls_distance * 237.0
        baseline = frame.loc["Baseline"]
        d3 = frame.loc["+ opt. #2"]
        duration_rows.append({
            "ham_name": workload.ham_name,
            "workload": workload.label,
            "Baseline": float(baseline["Sum"]) * float(baseline["Num_Layers"]) / sc_duration_us,
            "D3-ROT": float(d3["Sum"]) * float(d3["Num_Layers"]) / sc_duration_us,
            "LS": na_ls_duration_us / sc_duration_us,
            "LS (supercond.)": 1.0,
        })
        ls_ler = float(ls["space_time_volume"]) * logical_error_by_distance(ls_distance)
        ls_ler *= workload.loop_count
        ler_rows.append({
            "ham_name": workload.ham_name,
            "workload": workload.label,
            "Baseline": float(baseline["Logical_Error"]) * float(baseline["Num_Layers"])
            * workload.loop_count,
            "D3-ROT": float(d3["Logical_Error"]) * float(d3["Num_Layers"])
            * workload.loop_count,
            "LS": ls_ler,
            "LS (supercond.)": ls_ler,
        })
    duration_df = pd.DataFrame(duration_rows)
    ler_df = pd.DataFrame(ler_rows)
    if duration_df.empty:
        raise ValueError("No common transversal/Quration workloads were available")
    # Preserve the CSV column names produced by draw_qpe_graph_separate.py.
    # The plotting tables below additionally carry human-readable workload
    # labels and a computed Average row.
    duration_raw = duration_df[[
        "ham_name", "Baseline", "D3-ROT", "LS", "LS (supercond.)"
    ]].rename(columns={"D3-ROT": "+ opt. #2"})
    ler_raw = ler_df[[
        "ham_name", "Baseline", "D3-ROT", "LS", "LS (supercond.)"
    ]].rename(columns={
        "D3-ROT": "Opt. 2",
        "LS (supercond.)": "LS (Supercond.)",
    })
    average_duration = {"ham_name": "Average", "workload": "Average"}
    average_ler = {"ham_name": "Average", "workload": "Average"}
    series_names = ["Baseline", "D3-ROT", "LS", "LS (supercond.)"]
    for name in series_names:
        average_duration[name] = geomean(duration_df[name])
        average_ler[name] = geomean(ler_df[name])
    duration_df = pd.concat([duration_df, pd.DataFrame([average_duration])], ignore_index=True)
    ler_df = pd.concat([ler_df, pd.DataFrame([average_ler])], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    duration_raw.to_csv(args.output_dir / "fig17_execution_time.csv", index=False)
    ler_raw.to_csv(args.output_dir / "fig17_logical_error_rate.csv", index=False)
    duration_df.to_csv(args.output_dir / "fig17_plot_execution_time.csv", index=False)
    ler_df.to_csv(args.output_dir / "fig17_plot_logical_error_rate.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(duration_df)), 4.0))
    ax_ler = ax.twinx()
    x = np.arange(len(duration_df))
    width = 0.19
    colors = ["#111111", "#BBBBBB", "#777777", "#FFFFFF"]
    hatches = [None, "....", "----", None]
    for index, (name, color, hatch) in enumerate(zip(series_names, colors, hatches)):
        position = x + (index - 1.5) * width
        ax.bar(position, duration_df[name], width=width, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.6, label=name)
    series_offsets = (np.arange(len(series_names)) - 1.5) * width
    ler_label_added = False
    for workload_index, row in ler_df.iterrows():
        if row["workload"] == "Average":
            continue
        ax_ler.plot(
            x[workload_index] + series_offsets,
            row[series_names].to_numpy(float),
            marker="s",
            markersize=3,
            color="#4C8CCB",
            linewidth=1.0,
            label="Logical error rate" if not ler_label_added else None,
        )
        ler_label_added = True
    ax_ler.set_yscale("log")
    ax_ler.set_ylim(top=2e-3)
    ax_ler.axhline(1e-3, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, duration_df["workload"], rotation=35, ha="right")
    ax.set_ylabel("Execution time\n(normalized to LS (supercond.))")
    ax_ler.set_ylabel("Logical error rate")
    handles, labels = ax.get_legend_handles_labels()
    right_h, right_l = ax_ler.get_legend_handles_labels()
    ax.legend(handles + right_h, labels + right_l, frameon=False, fontsize=7,
              ncols=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, args.output_dir, "fig17")


CFG_SENSITIVITY_RE = re.compile(
    r"^(?P<arch>Base|D3ROT(?:_CELL3)?)_togl(?P<togl>\d+)_trf(?P<trf>\d+)_aod(?P<aod>\d+|inf)$"
)


def analyze_fig18(args, roots: list[Path], workloads: list[Workload]) -> None:
    def loader(workload):
        filename = (
            f"{workload.ham_name}_sensitivity_test_analysis_normalized_d"
            f"{workload.distance}.csv"
        )
        return pd.read_csv(require_named_file(roots, filename))

    loaded = selected_or_skip(workloads, loader, args.allow_partial)
    rows = []
    for workload, frame in loaded:
        value_column = workload.ham_name if workload.ham_name in frame.columns else "geomean"
        for _, row in frame.iterrows():
            match = CFG_SENSITIVITY_RE.match(str(row["cfg_name"]))
            if not match or int(match.group("togl")) != 1:
                continue
            if match.group("aod") not in {"4", "8", "inf"}:
                continue
            rows.append({
                "workload": workload.label,
                "architecture": "Baseline" if match.group("arch") == "Base" else "D3-ROT",
                "aod_pick_drop_us": int(match.group("trf")),
                "num_aods": match.group("aod"),
                "normalized_latency": float(row[value_column]),
            })
    long_df = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in long_df.groupby(["architecture", "aod_pick_drop_us", "num_aods"]):
        summary_rows.append({
            "Architecture": keys[0],
            "AOD pick/drop (us)": keys[1],
            "# AODs": keys[2],
            "Geomean normalized latency": geomean(group["normalized_latency"]),
        })
    summary = pd.DataFrame(summary_rows)
    # Match export_sensitivity_aod_for_excel.py's chart-data ordering.
    arch_order = {"D3-ROT": 0, "Baseline": 1}
    aod_order = {"inf": 0, "8": 1, "4": 2}
    summary["_arch"] = summary["Architecture"].map(arch_order)
    summary["_aod"] = summary["# AODs"].map(aod_order)
    summary = summary.sort_values(["_arch", "AOD pick/drop (us)", "_aod"]).drop(
        columns=["_arch", "_aod"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "fig18_raw_data.csv", index=False)
    long_df.to_csv(args.output_dir / "fig18_workload_data.csv", index=False)

    values = summary["Geomean normalized latency"].to_numpy(float)
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9, 4.0))
    bars = ax.bar(
        x,
        values,
        width=0.38,
        color="#A6A6A6",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=8)
    ax.set_xlim(-0.5, len(summary) - 0.5)
    ax.set_ylim(0.0, 8.0)
    ax.set_yticks([0.0, 4.0, 8.0])
    ax.set_xticks([])
    ax.set_ylabel("Normalized latency")

    def contiguous_spans(columns: list[str]) -> list[tuple[tuple[object, ...], int, int]]:
        spans = []
        start = 0
        previous = tuple(summary.iloc[0][columns])
        for index in range(1, len(summary)):
            current = tuple(summary.iloc[index][columns])
            if current != previous:
                spans.append((previous, start, index - 1))
                start = index
                previous = current
        spans.append((previous, start, len(summary) - 1))
        return spans

    label_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    row_bounds = [0.0, -0.09, -0.18, -0.27]
    for y in row_bounds:
        ax.plot(
            [-0.5, len(summary) - 0.5],
            [y, y],
            transform=label_transform,
            color="black",
            linewidth=0.55,
            clip_on=False,
        )
    for boundary in np.arange(-0.5, len(summary) + 0.5, 1.0):
        ax.plot(
            [boundary, boundary],
            [row_bounds[1], row_bounds[0]],
            transform=label_transform,
            color="black",
            linewidth=0.45,
            clip_on=False,
        )
    for (architecture, pick_drop), start, end in contiguous_spans(
        ["Architecture", "AOD pick/drop (us)"]
    ):
        center = (start + end) / 2
        ax.text(
            center,
            (row_bounds[1] + row_bounds[2]) / 2,
            f"{pick_drop} us",
            transform=label_transform,
            ha="center",
            va="center",
            fontsize=9,
            clip_on=False,
        )
        ax.plot(
            [start - 0.5, start - 0.5],
            [row_bounds[2], row_bounds[1]],
            transform=label_transform,
            color="black",
            linewidth=0.45,
            clip_on=False,
        )
    ax.plot(
        [len(summary) - 0.5, len(summary) - 0.5],
        [row_bounds[2], row_bounds[1]],
        transform=label_transform,
        color="black",
        linewidth=0.45,
        clip_on=False,
    )
    for (architecture,), start, end in contiguous_spans(["Architecture"]):
        center = (start + end) / 2
        ax.text(
            center,
            (row_bounds[2] + row_bounds[3]) / 2,
            architecture,
            transform=label_transform,
            ha="center",
            va="center",
            fontsize=9,
            clip_on=False,
        )
        ax.plot(
            [start - 0.5, start - 0.5],
            [row_bounds[3], row_bounds[2]],
            transform=label_transform,
            color="black",
            linewidth=0.45,
            clip_on=False,
        )
    ax.plot(
        [len(summary) - 0.5, len(summary) - 0.5],
        [row_bounds[3], row_bounds[2]],
        transform=label_transform,
        color="black",
        linewidth=0.45,
        clip_on=False,
    )
    for position, num_aods in zip(x, summary["# AODs"]):
        ax.text(
            position,
            (row_bounds[0] + row_bounds[1]) / 2,
            "Inf" if num_aods == "inf" else str(num_aods),
            transform=label_transform,
            ha="center",
            va="center",
            fontsize=9,
            clip_on=False,
        )
    for label, y in zip(
        ["# AODs", "Pick/drop latency", "Architecture"],
        [
            (row_bounds[0] + row_bounds[1]) / 2,
            (row_bounds[1] + row_bounds[2]) / 2,
            (row_bounds[2] + row_bounds[3]) / 2,
        ],
    ):
        ax.text(
            -0.65,
            y,
            label,
            transform=label_transform,
            ha="right",
            va="center",
            fontsize=9,
            clip_on=False,
        )

    fig.subplots_adjust(bottom=0.30, left=0.15, right=0.98, top=0.94)
    save_figure(fig, args.output_dir, "fig18")


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    reduced = frame.groupby("time", as_index=False)["space"].min().sort_values("time")
    keep = []
    best_space = math.inf
    for index, row in reduced.iterrows():
        if float(row["space"]) < best_space:
            keep.append(index)
            best_space = float(row["space"])
    return reduced.loc[keep]


def analyze_fig19(args, roots: list[Path], workloads: list[Workload]) -> None:
    def d3_sweep_loader(workload):
        filename = (
            f"{workload.ham_name}_ms_overhead_DIR_TOGL_DEDICATE_CELL2_d"
            f"{workload.distance}_sweep.csv"
        )
        return pd.read_csv(require_named_file(roots, filename))

    selected_hams = {workload.ham_name for workload in workloads}
    accepted_sweep_path = find_named_file(roots, "ms_overhead_geomean_sweep.csv")
    if len(workloads) == 4 and accepted_sweep_path is not None:
        # The archived paper data retains the unrounded per-workload latency and
        # physical-qubit values needed to reproduce the accepted Pareto points.
        d3_sweep = pd.read_csv(accepted_sweep_path)
        d3_sweep = d3_sweep[d3_sweep["qc_name"].isin(selected_hams)].copy()
        covered_hams = set(d3_sweep["qc_name"])
        missing_hams = selected_hams - covered_hams
        if missing_hams:
            raise ValueError(
                f"{accepted_sweep_path} is missing D3-ROT workload(s): "
                f"{sorted(missing_hams)}"
            )
    else:
        d3_loaded = selected_or_skip(
            workloads, d3_sweep_loader, args.allow_partial
        )
        d3_frames = []
        covered_hams = set()
        for workload, frame in d3_loaded:
            temp = frame.copy()
            temp["qc_name"] = workload.ham_name
            d3_frames.append(temp)
            covered_hams.add(workload.ham_name)
        d3_sweep = pd.concat(d3_frames, ignore_index=True)
        selected_hams = covered_hams

    d3_sweep["_time"] = (
        pd.to_numeric(d3_sweep["orig_latency"], errors="raise")
        + pd.to_numeric(d3_sweep["mean_stall"], errors="raise")
    ) / pd.to_numeric(d3_sweep["orig_latency"], errors="raise")
    d3_sweep["_space"] = (
        pd.to_numeric(d3_sweep["original_pq"], errors="raise")
        + pd.to_numeric(d3_sweep["msd_pq"], errors="raise")
        + pd.to_numeric(d3_sweep["msc_pq"], errors="raise")
        + pd.to_numeric(d3_sweep["ysd_pq"], errors="raise")
    ) / pd.to_numeric(d3_sweep["original_pq"], errors="raise")
    d3_sweep["_spacetime"] = d3_sweep["_time"] * d3_sweep["_space"]
    d3_summary = d3_sweep.groupby(
        ["num_msd_sf", "num_msc_sf", "num_ysd_sf"], as_index=False
    ).agg(
        time=("_time", geomean),
        space=("_space", geomean),
        spacetime=("_spacetime", geomean),
    )
    d3_summary.insert(0, "architecture", "D3-ROT")

    ls_sweep_path = require_named_file(roots, "quration_trace_ms_overhead_sweep.csv")
    ls_sweep = pd.read_csv(ls_sweep_path)
    ls_sweep = ls_sweep[ls_sweep["ham_name"].isin(selected_hams)]
    if ls_sweep.empty:
        raise ValueError("Quration trace sweep has no selected workloads")
    ls_sweep["_time"] = (
        pd.to_numeric(ls_sweep["runtime"], errors="raise")
        / pd.to_numeric(ls_sweep["ideal_runtime"], errors="raise")
    )
    ls_sweep["_space"] = (
        pd.to_numeric(ls_sweep["original_pq"], errors="raise")
        + pd.to_numeric(ls_sweep["msd_pq"], errors="raise")
        + pd.to_numeric(ls_sweep["msc_pq"], errors="raise")
    ) / pd.to_numeric(ls_sweep["original_pq"], errors="raise")
    ls_sweep["_spacetime"] = ls_sweep["_time"] * ls_sweep["_space"]
    ls_summary = ls_sweep.groupby(["num_msd_sf", "num_msc_sf"], as_index=False).agg(
        time=("_time", geomean),
        space=("_space", geomean),
        spacetime=("_spacetime", geomean),
    )
    ls_summary.insert(0, "architecture", "LS")
    ls_summary["num_ysd_sf"] = np.nan

    output = pd.concat([d3_summary, ls_summary], ignore_index=True, sort=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "fig19_raw_data.csv", index=False)

    fig = plt.figure(figsize=(8.2, 3.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.32], wspace=0.25)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[0, 2])
    plot_specs = [
        ("D3-ROT", d3_summary, "#00B050"),
        ("Lattice surgery", ls_summary, "#C00000"),
    ]
    for panel_index, (ax, (architecture, frame, color)) in enumerate(
        zip(axes, plot_specs)
    ):
        frontier = pareto_front(frame)
        best = frame.loc[frame["spacetime"].idxmin()]
        ax.scatter([1.0], [1.0], color=color, marker="D", s=40, zorder=4)
        ax.plot(
            frontier["time"],
            frontier["space"],
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=1.5,
            zorder=2,
        )
        ax.scatter(
            [best["time"]],
            [best["space"]],
            facecolors="white",
            edgecolors=color,
            marker="o",
            s=70,
            linewidths=1.4,
            zorder=5,
        )
        ax.text(
            0.96,
            0.94,
            architecture,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
        )
        annotation_position = (0.54, 0.68) if panel_index == 0 else (0.32, 0.43)
        ax.text(
            *annotation_position,
            f"{best['spacetime']:.2f}x space-time\noverhead",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.set_xlim(0.9, 2.5)
        ax.set_xticks([1.0, 1.5, 2.0, 2.5])
        ax.set_ylim(0.9, 2.2)
        ax.set_yticks([1.0, 1.5, 2.0])
        ax.set_xlabel("Normalized time overhead")
    axes[0].set_ylabel("Normalized space\noverhead")

    legend_handles = [
        Line2D([], [], color="black", marker="D", linestyle="None", markersize=7,
               label="Ideal"),
        Line2D([], [], color="black", marker="o", linewidth=1.2, markersize=6,
               label="Pareto"),
        Line2D([], [], color="black", marker="o", markerfacecolor="white",
               linestyle="None", markersize=8, label="Best"),
    ]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=9,
        borderpad=0.5,
        labelspacing=0.9,
        handlelength=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.95)
    save_figure(fig, args.output_dir, "fig19")


HS_CONFIGS = [
    ("noS / noH", "noS_noH"),
    ("S / noH", "S_noH"),
    ("noS / H", "noS_H"),
    ("S / H", "S_H"),
]


def analyze_fig20(args, roots: list[Path], workloads: list[Workload]) -> None:
    def loader(workload):
        filename = f"{workload.ham_name}_hs_skip_analysis_d{workload.distance}.csv"
        return pd.read_csv(require_named_file(roots, filename)).set_index("cfg_name")

    loaded = selected_or_skip(workloads, loader, args.allow_partial)
    workload_rows = []
    for workload, frame in loaded:
        for architecture, prefix in [("Baseline", "BASE"), ("D3-ROT CELL3", "D3ROT_CELL3")]:
            baseline_value = float(frame.loc[f"{prefix}_noS_noH", "total_us"])
            for option_label, suffix in HS_CONFIGS:
                value = float(frame.loc[f"{prefix}_{suffix}", "total_us"]) / baseline_value
                workload_rows.append({
                    "workload": workload.label,
                    "architecture": architecture,
                    "HS option": option_label,
                    "normalized_total_latency": value,
                })
    workload_df = pd.DataFrame(workload_rows)
    summary_rows = []
    for keys, group in workload_df.groupby(["architecture", "HS option"], sort=False):
        summary_rows.append({
            "Architecture": keys[0],
            "HS option": keys[1],
            "Geomean normalized total latency": geomean(group["normalized_total_latency"]),
        })
    summary = pd.DataFrame(summary_rows)
    arch_order = {"Baseline": 0, "D3-ROT CELL3": 1}
    option_order = {label: index for index, (label, _) in enumerate(HS_CONFIGS)}
    summary["_arch"] = summary["Architecture"].map(arch_order)
    summary["_option"] = summary["HS option"].map(option_order)
    summary = summary.sort_values(["_arch", "_option"]).drop(columns=["_arch", "_option"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "fig20_raw_data.csv", index=False)
    workload_df.to_csv(args.output_dir / "fig20_workload_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    display_labels = ["H, S excl.", "H excl.", "S excl.", "All gates"]
    architecture_specs = [
        ("Baseline", "Baseline", "white"),
        ("D3-ROT CELL3", "D3-ROT", "black"),
    ]
    all_values = []
    all_colors = []
    for architecture, _, color in architecture_specs:
        group = summary[summary["Architecture"] == architecture].set_index("HS option")
        values = [group.loc[label, "Geomean normalized total latency"] for label, _ in HS_CONFIGS]
        all_values.extend(values)
        all_colors.extend([color] * len(values))

    x = np.arange(len(all_values))
    bars = ax.bar(
        x,
        all_values,
        width=0.32,
        color=all_colors,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.bar_label(
        bars,
        labels=[f"{value:.2f}" for value in all_values],
        fontsize=8,
        padding=2,
    )
    ax.set_xlim(-0.5, len(all_values) - 0.5)
    ax.set_ylim(0.0, 4.4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xticks(x, display_labels * len(architecture_specs))
    ax.set_ylabel("Normalized\nexecution time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    label_transform = ax.get_xaxis_transform()
    for group_index, (_, group_label, _) in enumerate(architecture_specs):
        group_start = group_index * len(HS_CONFIGS)
        group_center = group_start + (len(HS_CONFIGS) - 1) / 2
        ax.text(
            group_center,
            -0.25,
            group_label,
            transform=label_transform,
            ha="center",
            va="top",
            fontsize=9,
            clip_on=False,
        )
    ax.vlines(
        [-0.5, len(HS_CONFIGS) - 0.5, len(all_values) - 0.5],
        ymin=-0.28,
        ymax=0.0,
        transform=label_transform,
        color="black",
        linewidth=0.6,
        clip_on=False,
    )
    ax.tick_params(axis="x", labelsize=8, pad=4)
    ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.34, top=0.90)
    save_figure(fig, args.output_dir, "fig20")


ANALYZERS = {
    "fig11a": analyze_fig11a,
    "table3": analyze_table3,
    "fig15": analyze_fig15,
    "fig16": analyze_fig16,
    "fig17": analyze_fig17,
    "fig18": analyze_fig18,
    "fig19": analyze_fig19,
    "fig20": analyze_fig20,
}


def main(target: str) -> None:
    if target not in ANALYZERS:
        raise ValueError(f"Unsupported target: {target}")
    args = parse_args(target)
    roots = (
        [path.resolve() for path in args.input_dir]
        if args.input_dir
        else default_input_roots(target)
    )
    workloads = load_workloads(target, args.workload)
    ANALYZERS[target](args, roots, workloads)
    print(f"Wrote {target} artifacts under: {args.output_dir.resolve()}")
