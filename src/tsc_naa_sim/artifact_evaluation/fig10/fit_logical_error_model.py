#!/usr/bin/env python3
"""Refit and verify the coefficients embedded in logical_error_model.py."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from model import (
    AE_DIR,
    DEFAULT_INPUT_DIR,
    fit_model_single,
    fit_model_two,
    load_model_single,
    load_model_two,
)


LOGICAL_ERROR_MODEL = AE_DIR.parent / "logical_error_model.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the accepted Fig. 10(b) data and compare the result with the "
            "constants embedded in logical_error_model.py."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root containing the Fig. 10 modeling CSVs (searched recursively).",
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=LOGICAL_ERROR_MODEL,
        help="logical_error_model.py whose active constants are checked.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AE_DIR / "fig10" / "results",
        help="Destination for coefficient CSV and text output.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for the fitted-versus-embedded check.",
    )
    return parser.parse_args()


def embedded_assignments(
    source_path: Path, function_name: str, parameter_names: tuple[str, ...]
) -> dict[str, float]:
    """Read active numeric assignments from one function without importing it."""

    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"Could not find function {function_name} in {source_path}")

    values: dict[str, float] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in parameter_names:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (int, float)):
            values[target.id] = float(value)

    missing = set(parameter_names) - set(values)
    if missing:
        raise ValueError(
            f"Could not read active assignments {sorted(missing)} from "
            f"{function_name} in {source_path}"
        )
    return {name: values[name] for name in parameter_names}


def comparison_rows(
    operation: str,
    function_name: str,
    fitted: dict[str, float],
    embedded: dict[str, float],
    rtol: float,
) -> list[dict[str, object]]:
    rows = []
    for name, fitted_value in fitted.items():
        embedded_value = embedded[name]
        absolute_error = abs(fitted_value - embedded_value)
        relative_error = absolute_error / abs(embedded_value)
        rows.append(
            {
                "operation": operation,
                "function": function_name,
                "parameter": name,
                "fitted_value": fitted_value,
                "embedded_value": embedded_value,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "matches_embedded": bool(
                    np.isclose(fitted_value, embedded_value, rtol=rtol, atol=1e-12)
                ),
            }
        )
    return rows


def assignment_block(function_name: str, coefficients: dict[str, float]) -> str:
    assignments = "\n".join(
        f"{name} = {value:.12e}" for name, value in coefficients.items()
    )
    return f"{function_name}:\n{assignments}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    single_fit = fit_model_single(load_model_single(args.input_dir))
    two_fit = fit_model_two(load_model_two(args.input_dir))
    embedded_single = embedded_assignments(
        args.model_source, "sq_logical_error", ("A", "C1", "C0")
    )
    embedded_two = embedded_assignments(
        args.model_source, "tq_logical_error", ("A", "c1", "c2", "c0")
    )

    table = pd.DataFrame(
        comparison_rows(
            "single_lq",
            "sq_logical_error",
            single_fit.coefficients,
            embedded_single,
            args.rtol,
        )
        + comparison_rows(
            "two_lq",
            "tq_logical_error",
            two_fit.coefficients,
            embedded_two,
            args.rtol,
        )
    )
    csv_path = args.output_dir / "logical_error_model_coefficients.csv"
    table.to_csv(csv_path, index=False)

    passed = bool(table["matches_embedded"].all())
    report = "\n\n".join(
        [
            "Coefficients refitted from the accepted Fig. 10(b) CSVs",
            assignment_block("sq_logical_error", single_fit.coefficients),
            assignment_block("tq_logical_error", two_fit.coefficients),
            "Coefficients read from logical_error_model.py",
            assignment_block("sq_logical_error", embedded_single),
            assignment_block("tq_logical_error", embedded_two),
            f"Verification: {'PASS' if passed else 'FAIL'} (rtol={args.rtol:g})",
        ]
    ) + "\n"
    text_path = args.output_dir / "logical_error_model_coefficients.txt"
    text_path.write_text(report)
    print(report, end="")
    print(f"Detailed comparison: {csv_path.resolve()}")

    if not passed:
        raise SystemExit(
            "Refitted coefficients do not match the active embedded coefficients. "
            "See the comparison CSV."
        )


if __name__ == "__main__":
    main()
