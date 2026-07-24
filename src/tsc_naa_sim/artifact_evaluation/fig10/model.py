#!/usr/bin/env python3
"""Data loading and model fitting shared by the Fig. 10 AE scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


AE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = AE_DIR / "raw_data" / "fig10"

HIGH_SINGLE_FILENAME = "compressed_result_raw_memory_with_pick_drop_high_p_CZ.csv"
HIGH_TWO_FILENAME = "compressed_result_raw_bell_pair_with_pick_drop_high_p_CZ.csv"
MODEL_SINGLE_FILENAME = (
    "compressed_result_raw_memory_with_pick_drop_1qsmall_CZ.csv"
)
MODEL_TWO_FILENAME = (
    "compressed_result_raw_bell_pair_with_pick_drop_1qsmall_CZ.csv"
)

DISTANCES = (3, 5, 7, 9, 11, 13)
HIGH_TRAIN_DISTANCES = (7, 9, 11, 13)
HIGH_VALIDATION_DISTANCE = 23


@dataclass(frozen=True)
class FitResult:
    """Fitted physical coefficients and the exact rows used for fitting."""

    coefficients: dict[str, float]
    covariance: np.ndarray
    fit_rows: pd.DataFrame


def find_input(input_dir: Path, filename: str) -> Path:
    """Find an accepted-data CSV below an explicitly selected input root."""

    input_dir = input_dir.resolve()
    if input_dir.is_file() and input_dir.name == filename:
        return input_dir
    direct = input_dir / filename
    if direct.is_file():
        return direct
    matches = sorted(input_dir.rglob(filename)) if input_dir.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} below {input_dir}")
    if len(matches) > 1:
        listed = "\n".join(f"- {path}" for path in matches)
        raise RuntimeError(f"Found multiple copies of {filename}:\n{listed}")
    return matches[0]


def load_csv(input_dir: Path, filename: str, required: set[str]) -> pd.DataFrame:
    path = find_input(input_dir, filename)
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def load_high_single(input_dir: Path) -> pd.DataFrame:
    return load_csv(input_dir, HIGH_SINGLE_FILENAME, {"d", "p", "n", "p_l"})


def load_high_two(input_dir: Path) -> pd.DataFrame:
    return load_csv(
        input_dir, HIGH_TWO_FILENAME, {"d", "p", "n1", "n2", "p_l"}
    )


def load_model_single(input_dir: Path) -> pd.DataFrame:
    return load_csv(input_dir, MODEL_SINGLE_FILENAME, {"d", "n", "p_l"})


def load_model_two(input_dir: Path) -> pd.DataFrame:
    return load_csv(input_dir, MODEL_TWO_FILENAME, {"d", "n1", "n2", "p_l"})


def _finite_positive(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isfinite(frame["p_l"]) & (frame["p_l"] > 0)].copy()


def _single_log_model(x, log_a: float, c_transport: float, c_base: float):
    distance, transports = x
    effective_error = np.maximum(c_transport * transports + c_base, 1e-12)
    return log_a + distance / 2.0 * np.log(effective_error)


def _two_log_model(
    x,
    log_a: float,
    c_control: float,
    c_target: float,
    c_base: float,
):
    distance, control_transports, target_transports = x
    effective_error = np.maximum(
        c_control * control_transports + c_target * target_transports + c_base,
        1e-12,
    )
    return log_a + distance / 2.0 * np.log(effective_error)


def _single_log10_model(x, log10_a: float, c_transport: float, c_base: float):
    distance, transports = x
    effective_error = np.maximum(c_transport * transports + c_base, 1e-12)
    return log10_a + distance / 2.0 * np.log10(effective_error)


def _two_log10_model(
    x,
    log10_a: float,
    c_control: float,
    c_target: float,
    c_base: float,
):
    distance, control_transports, target_transports = x
    effective_error = np.maximum(
        c_control * control_transports + c_target * target_transports + c_base,
        1e-12,
    )
    return log10_a + distance / 2.0 * np.log10(effective_error)


def _fit_single(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    params, covariance = curve_fit(
        _single_log_model,
        (frame["d"].to_numpy(), frame["n"].to_numpy()),
        np.log(frame["p_l"].to_numpy()),
        p0=(0.0, 1e-3, 1e-2),
        bounds=([-np.inf, 1e-8, 1e-80], [np.inf, 1.0, 1.0]),
        maxfev=100_000,
    )
    return params, covariance


def _fit_two(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    params, covariance = curve_fit(
        _two_log_model,
        (
            frame["d"].to_numpy(),
            frame["n1"].to_numpy(),
            frame["n2"].to_numpy(),
        ),
        np.log(frame["p_l"].to_numpy()),
        p0=(0.0, 1e-3, 1e-3, 1e-2),
        bounds=([-np.inf, 1e-8, 1e-8, 1e-8], [np.inf, 1.0, 1.0, 1.0]),
        maxfev=100_000,
    )
    return params, covariance


def _fit_high_single(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    params, covariance = curve_fit(
        _single_log10_model,
        (frame["d"].to_numpy(), frame["n"].to_numpy()),
        np.log10(frame["p_l"].to_numpy()),
        p0=(0.0, 1e-3, 1e-2),
        bounds=([-np.inf, 1e-8, 1e-8], [np.inf, 1.0, 1.0]),
        maxfev=100_000,
    )
    return params, covariance


def _fit_high_two(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    params, covariance = curve_fit(
        _two_log10_model,
        (
            frame["d"].to_numpy(),
            frame["n1"].to_numpy(),
            frame["n2"].to_numpy(),
        ),
        np.log10(frame["p_l"].to_numpy()),
        p0=(0.0, 1e-3, 1e-3, 1e-2),
        bounds=([-np.inf, 1e-8, 1e-8, 1e-8], [np.inf, 1.0, 1.0, 1.0]),
        maxfev=100_000,
    )
    return params, covariance


def fit_model_single(frame: pd.DataFrame) -> FitResult:
    """Reproduce the single-LQ fit used by logical_error_model.py."""

    frame = _finite_positive(frame)
    excluded = (
        (frame["d"] == 3)
        | (frame["d"] == 5)
        | ((frame["d"] == 11) & (frame["n"] < 3))
        | ((frame["d"] == 13) & (frame["n"] < 6))
        | ((frame["d"] == 15) & (frame["n"] < 6))
    )
    fit_rows = frame[~excluded].copy()
    params, covariance = _fit_single(fit_rows)
    return FitResult(
        coefficients={
            "A": float(np.exp(params[0])),
            "C1": float(params[1]),
            "C0": float(params[2]),
        },
        covariance=covariance,
        fit_rows=fit_rows,
    )


def fit_model_two(frame: pd.DataFrame) -> FitResult:
    """Reproduce the two-LQ fit used by logical_error_model.py."""

    frame = _finite_positive(frame)
    fit_rows = frame[(frame["d"] >= 5) & (frame["n1"] + frame["n2"] >= 4)].copy()
    params, covariance = _fit_two(fit_rows)
    return FitResult(
        coefficients={
            "A": float(np.exp(params[0])),
            "c1": float(params[1]),
            "c2": float(params[2]),
            "c0": float(params[3]),
        },
        covariance=covariance,
        fit_rows=fit_rows,
    )


def fit_high_single(frame: pd.DataFrame) -> FitResult:
    """Fit high-p low-distance data for the held-out d=23 validation."""

    frame = _finite_positive(frame)
    selected_p = float(frame["p"].max())
    fit_rows = frame[
        np.isclose(frame["p"], selected_p)
        & frame["d"].isin(HIGH_TRAIN_DISTANCES)
    ].copy()
    params, covariance = _fit_high_single(fit_rows)
    return FitResult(
        coefficients={
            "A": float(np.power(10.0, params[0])),
            "c_transport": float(params[1]),
            "c_base": float(params[2]),
        },
        covariance=covariance,
        fit_rows=fit_rows,
    )


def fit_high_two(frame: pd.DataFrame) -> FitResult:
    """Fit high-p low-distance data for the held-out d=23 validation."""

    frame = _finite_positive(frame)
    selected_p = float(frame["p"].max())
    fit_rows = frame[
        np.isclose(frame["p"], selected_p)
        & frame["d"].isin(HIGH_TRAIN_DISTANCES)
    ].copy()
    params, covariance = _fit_high_two(fit_rows)
    return FitResult(
        coefficients={
            "A": float(np.power(10.0, params[0])),
            "c_control": float(params[1]),
            "c_target": float(params[2]),
            "c_base": float(params[3]),
        },
        covariance=covariance,
        fit_rows=fit_rows,
    )


def predict_single(coefficients: dict[str, float], distance, transports):
    c_transport = coefficients.get("C1", coefficients.get("c_transport"))
    c_base = coefficients.get("C0", coefficients.get("c_base"))
    return coefficients["A"] * (
        c_transport * np.asarray(transports) + c_base
    ) ** (np.asarray(distance) / 2.0)


def predict_two(
    coefficients: dict[str, float], distance, control_transports, target_transports
):
    c_control = coefficients.get("c1", coefficients.get("c_control"))
    c_target = coefficients.get("c2", coefficients.get("c_target"))
    return coefficients["A"] * (
        c_control * np.asarray(control_transports)
        + c_target * np.asarray(target_transports)
        + coefficients["c0" if "c0" in coefficients else "c_base"]
    ) ** (np.asarray(distance) / 2.0)
