# Fig. 10(a) and Fig. 10(b)

These scripts reproduce the two plots from the accepted-version decoding data
copied to `../raw_data/fig10/`. They do not run the architecture simulator and
therefore have no `run.sh` prerequisite.

From `artifact_evaluation/`, run:

```bash
uv run python fig10/analyze.py
```

This writes the plotted data, validation summary, fitted high-error-rate
coefficients, and PDF/PNG plots to `fig10/results/`. Use `--input-dir` to read an
equivalent four-CSV dataset from another location and `--output-dir` to change
the destination.

To reproduce the low-error-rate coefficients used by the architecture
simulator and verify them against the active assignments in
`../../logical_error_model.py`, run:

```bash
uv run python fig10/fit_logical_error_model.py
```

The script prints both the refitted and embedded values and writes a detailed
comparison to `fig10/results/logical_error_model_coefficients.csv`. The fit uses
the same accepted-version row filters and model equations used when those
constants were obtained.

The four input filenames and their unmodified schemas are documented in
[`../raw_data/README.md`](../raw_data/README.md).
