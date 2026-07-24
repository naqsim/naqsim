#!/usr/bin/env python3
"""Reproduce Fig. 10(a) and Fig. 10(b) from the shared accepted data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from model import (
    AE_DIR,
    DEFAULT_INPUT_DIR,
    DISTANCES,
    HIGH_VALIDATION_DISTANCE,
    fit_high_single,
    fit_high_two,
    fit_model_single,
    fit_model_two,
    load_high_single,
    load_high_two,
    load_model_single,
    load_model_two,
    predict_single,
    predict_two,
)


COLORS = {
    3: "#4472C4",
    5: "#A5A5A5",
    7: "#5B9BD5",
    9: "#264478",
    11: "#636363",
    13: "#9DC3E6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read the shared accepted data and render Fig. 10(a) and (b)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root containing the four Fig. 10 CSVs (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AE_DIR / "fig10" / "results",
        help="Destination for generated CSV, PDF, and PNG files.",
    )
    return parser.parse_args()


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _validation_rows(
    single_data: pd.DataFrame,
    two_data: pd.DataFrame,
    single_coefficients: dict[str, float],
    two_coefficients: dict[str, float],
) -> pd.DataFrame:
    selected_single_p = float(single_data["p"].max())
    single = single_data[
        np.isclose(single_data["p"], selected_single_p)
        & (single_data["d"] == HIGH_VALIDATION_DISTANCE)
        & np.isfinite(single_data["p_l"])
        & (single_data["p_l"] > 0)
    ].copy()
    single["operation"] = "single_lq"
    single["n_control"] = single["n"]
    single["n_target"] = np.nan
    single["predicted_p_l"] = predict_single(
        single_coefficients, single["d"], single["n"]
    )

    selected_two_p = float(two_data["p"].max())
    two = two_data[
        np.isclose(two_data["p"], selected_two_p)
        & (two_data["d"] == HIGH_VALIDATION_DISTANCE)
        & np.isfinite(two_data["p_l"])
        & (two_data["p_l"] > 0)
    ].copy()
    two["operation"] = "two_lq"
    two["n_control"] = two["n1"]
    two["n_target"] = two["n2"]
    two["predicted_p_l"] = predict_two(
        two_coefficients, two["d"], two["n1"], two["n2"]
    )

    columns = [
        "operation",
        "p",
        "d",
        "n_control",
        "n_target",
        "p_l",
        "predicted_p_l",
    ]
    combined = pd.concat([single[columns], two[columns]], ignore_index=True)
    combined["stim_log10_p_l"] = np.log10(combined["p_l"])
    combined["predicted_log10_p_l"] = np.log10(combined["predicted_p_l"])
    combined["log10_residual"] = (
        combined["stim_log10_p_l"] - combined["predicted_log10_p_l"]
    )
    return combined


def make_fig10a(rows: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5.0, 4.1))
    styles = {
        "single_lq": ("#F28E2B", "D", "Single LQ ($\\times 11$)"),
        "two_lq": ("#4E79A7", "o", "Two LQ ($\\times 7$)"),
    }
    for operation, (color, marker, label) in styles.items():
        selected = rows[rows["operation"] == operation]
        axis.scatter(
            selected["stim_log10_p_l"],
            selected["predicted_log10_p_l"],
            color=color,
            marker=marker,
            s=25,
            alpha=0.85,
            linewidths=0.25,
            edgecolors="black",
            label=label,
        )
    lower = float(
        min(rows["stim_log10_p_l"].min(), rows["predicted_log10_p_l"].min())
    )
    upper = float(
        max(rows["stim_log10_p_l"].max(), rows["predicted_log10_p_l"].max())
    )
    padding = 0.04 * (upper - lower)
    limits = (lower - padding, upper + padding)
    axis.plot(limits, limits, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Stim $\log_{10}(p_L)$")
    axis.set_ylabel(r"Predicted $\log_{10}(p_L)$")
    axis.grid(True, color="#dddddd", linewidth=0.5)
    axis.legend(frameon=False, fontsize=8)
    axis.text(0.96, 0.06, "$d=23$", transform=axis.transAxes, ha="right")
    figure.tight_layout()
    return figure


def _fig10b_rows(
    single_data: pd.DataFrame,
    two_data: pd.DataFrame,
    single_coefficients: dict[str, float],
    two_coefficients: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    single_fit_mask = ~(
        (single_data["d"] == 3)
        | (single_data["d"] == 5)
        | ((single_data["d"] == 11) & (single_data["n"] < 3))
        | ((single_data["d"] == 13) & (single_data["n"] < 6))
        | ((single_data["d"] == 15) & (single_data["n"] < 6))
    )
    for index, item in single_data.iterrows():
        if item["d"] not in DISTANCES or not np.isfinite(item["p_l"]) or item["p_l"] <= 0:
            continue
        rows.append(
            {
                "operation": "single_lq",
                "d": int(item["d"]),
                "n_control": int(item["n"]),
                "n_target": np.nan,
                "p_l": float(item["p_l"]),
                "predicted_p_l": float(
                    predict_single(single_coefficients, item["d"], item["n"])
                ),
                "used_for_fit": bool(single_fit_mask.loc[index]),
            }
        )

    two_fit_mask = (two_data["d"] >= 5) & (two_data["n1"] + two_data["n2"] >= 4)
    for index, item in two_data[two_data["n2"].isin((0, 8))].iterrows():
        if item["d"] not in DISTANCES or not np.isfinite(item["p_l"]) or item["p_l"] <= 0:
            continue
        rows.append(
            {
                "operation": "two_lq",
                "d": int(item["d"]),
                "n_control": int(item["n1"]),
                "n_target": int(item["n2"]),
                "p_l": float(item["p_l"]),
                "predicted_p_l": float(
                    predict_two(
                        two_coefficients, item["d"], item["n1"], item["n2"]
                    )
                ),
                "used_for_fit": bool(two_fit_mask.loc[index]),
            }
        )
    return pd.DataFrame(rows)


def make_fig10b(
    rows: pd.DataFrame,
    single_coefficients: dict[str, float],
    two_coefficients: dict[str, float],
) -> plt.Figure:
    figure, (single_axis, two_axis) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    continuous_n = np.linspace(0, 15, 301)

    single_rows = rows[rows["operation"] == "single_lq"]
    for distance in DISTANCES:
        color = COLORS[distance]
        selected = single_rows[single_rows["d"] == distance]
        single_axis.scatter(
            selected["n_control"], selected["p_l"], color=color, s=13, zorder=3
        )
        single_axis.plot(
            continuous_n,
            predict_single(single_coefficients, distance, continuous_n),
            color=color,
            linewidth=1.2,
        )

    two_rows = rows[rows["operation"] == "two_lq"]
    for distance in DISTANCES:
        color = COLORS[distance]
        for target_transports, marker in ((0, "o"), (8, "^")):
            selected = two_rows[
                (two_rows["d"] == distance)
                & (two_rows["n_target"] == target_transports)
            ]
            two_axis.scatter(
                selected["n_control"],
                selected["p_l"],
                color=color,
                marker=marker,
                s=15,
                zorder=3,
            )
            two_axis.plot(
                continuous_n,
                predict_two(
                    two_coefficients, distance, continuous_n, target_transports
                ),
                color=color,
                linewidth=1.2,
            )

    for axis in (single_axis, two_axis):
        axis.set_yscale("log")
        axis.set_xlim(-0.3, 15.3)
        axis.set_ylim(1e-10, 1e-2)
        axis.set_xticks((0, 3, 6, 9, 12, 15))
        axis.grid(True, which="major", color="#dddddd", linewidth=0.5)
        axis.set_xlabel("Transports of control LQ ($n_{ctrl}$)")
    single_axis.set_xlabel("Transports of a LQ ($n$)")
    single_axis.set_ylabel("Logical error rate ($p_L$)")
    single_axis.set_title("Single LQ", fontsize=10)
    two_axis.set_title("Two LQ", fontsize=10)

    distance_handles = [
        Line2D([0], [0], color=COLORS[distance], linewidth=2, label=f"d={distance}")
        for distance in DISTANCES
    ]
    single_axis.legend(
        handles=distance_handles,
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="lower right",
    )
    target_handles = [
        Line2D(
            [0], [0], color="#555555", marker="o", linewidth=1.2, label="$n_{targ}=0$"
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="^",
            linewidth=1.2,
            label="$n_{targ}=8$",
        ),
    ]
    two_axis.legend(
        handles=distance_handles + target_handles,
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="lower right",
    )
    figure.tight_layout()
    return figure


def _write_parameter_csv(
    output_dir: Path,
    filename: str,
    operation_and_coefficients: list[tuple[str, dict[str, float]]],
) -> None:
    rows = []
    for operation, coefficients in operation_and_coefficients:
        rows.extend(
            {"operation": operation, "parameter": name, "value": value}
            for name, value in coefficients.items()
        )
    pd.DataFrame(rows).to_csv(output_dir / filename, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    high_single_data = load_high_single(args.input_dir)
    high_two_data = load_high_two(args.input_dir)
    high_single_fit = fit_high_single(high_single_data)
    high_two_fit = fit_high_two(high_two_data)
    fig10a_rows = _validation_rows(
        high_single_data,
        high_two_data,
        high_single_fit.coefficients,
        high_two_fit.coefficients,
    )
    fig10a_rows.to_csv(args.output_dir / "fig10a_raw_data.csv", index=False)
    _write_parameter_csv(
        args.output_dir,
        "fig10a_fitted_coefficients.csv",
        [
            ("single_lq", high_single_fit.coefficients),
            ("two_lq", high_two_fit.coefficients),
        ],
    )
    save_figure(make_fig10a(fig10a_rows), args.output_dir, "fig10a")

    model_single_data = load_model_single(args.input_dir)
    model_two_data = load_model_two(args.input_dir)
    model_single_fit = fit_model_single(model_single_data)
    model_two_fit = fit_model_two(model_two_data)
    fig10b_rows = _fig10b_rows(
        model_single_data,
        model_two_data,
        model_single_fit.coefficients,
        model_two_fit.coefficients,
    )
    fig10b_rows.to_csv(args.output_dir / "fig10b_raw_data.csv", index=False)
    save_figure(
        make_fig10b(
            fig10b_rows,
            model_single_fit.coefficients,
            model_two_fit.coefficients,
        ),
        args.output_dir,
        "fig10b",
    )

    rmse = (
        fig10a_rows.groupby("operation")["log10_residual"]
        .apply(lambda values: float(np.sqrt(np.mean(np.square(values)))))
        .rename("log10_rmse")
        .reset_index()
    )
    rmse.to_csv(args.output_dir / "fig10a_validation_summary.csv", index=False)

    print(f"Read Fig. 10 data from {args.input_dir.resolve()}")
    print(f"Wrote Fig. 10(a)/(b) CSV, PDF, and PNG files to {args.output_dir.resolve()}")
    print(rmse.to_string(index=False))


if __name__ == "__main__":
    main()
