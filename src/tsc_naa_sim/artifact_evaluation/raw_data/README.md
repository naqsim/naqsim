# Shared paper raw data

This directory contains the existing per-workload CSV inputs used by each
artifact-evaluation `analyze.py`. It is intentionally separate from the default
analysis output directories (`../<target>/results/`). Running an analyzer does
not write into this directory.

The first directory level identifies the paper figure or table. The second
identifies the simulation or Quration stage that produced the CSV:

| Target | Input directories | Contents |
| --- | --- | --- |
| Fig. 10(a), (b) | `fig10/validation/`, `fig10/modeling/` | high-error-rate d=23 validation and accepted-version low-error-rate model-fitting data |
| Fig. 11(a) | `fig11a/compiler/` | Baseline per-layer compiler results |
| Table 3 | `table3/compiler/` | Compiler aggregate and per-layer results for all four configurations |
| Fig. 15 | `fig15/d3rot/` | D3-ROT aggregate results |
| Fig. 16 | `fig16/final/` | Final Baseline/compiler/D3-ROT comparison results |
| Fig. 17 | `fig17/final/`, `fig17/quration/` | Final comparison results and the Quration summary |
| Fig. 18 | `fig18/sensitivity/` | AOD sensitivity results |
| Fig. 19 | `fig19/d3rot/`, `fig19/quration/` | D3-ROT resource-state and Quration trace sweeps |
| Fig. 20 | `fig20/hs_skip/` | H/S-gate skipping results |

With no `--input-dir` argument, each analyzer searches its matching directory
under this root first. To restrict an analysis to only the shared files, use,
for example:

```bash
uv run python fig16/analyze.py --input-dir raw_data/fig16
```