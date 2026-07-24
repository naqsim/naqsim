# Artifact-evaluation scripts

This directory contains simulation and post-processing entry points for the numerical results reported in MICRO 2026 paper #50.
The architecture-simulation targets have one `run.sh` and one `analyze.py` each:

| Paper result | Directory | Simulation data used |
| --- | --- | --- |
| Fig. 10(a), (b) | `fig10/` | surface-code decoding and logical-error-model data |
| Fig. 11(a) | `fig11a/` | compiler-optimization evaluation |
| Table 3 | `table3/` | compiler-optimization evaluation |
| Fig. 15 | `fig15/` | D3-ROT optimization evaluation |
| Fig. 16 | `fig16/` | compiler and D3-ROT results, combined by the final-analysis stage |
| Fig. 17 | `fig17/` | Fig. 16 results and Quration summary |
| Fig. 18 | `fig18/` | AOD sensitivity evaluation |
| Fig. 19 | `fig19/` | D3-ROT resource-state sweep and Quration trace sweep |
| Fig. 20 | `fig20/` | H/S-gate skipping evaluation |

The wrappers retain the existing simulator output locations and CSV schemas.
The AE adapter in `common/stage_runner.py` separates compilation from execution, deduplicates both stages through a shared content-addressed cache, and admits worker processes according to available memory. Shared orchestration and plotting code is in `common/`; workload names, code distances, and requested worker counts are in `config/workloads.csv`.

Shared intermediate results are stored in `.cache/` in this directory and are ignored by Git.
Cache keys include the circuit, naive/precomputed mapping, the relevant compiler settings, the complete execution settings, run mode, and a hash of the simulator Python sources.
Consequently, reuse is allowed across different figure/table `run.sh` entry points, but a simulator source change automatically selects a new cache key.

## Requirements

The `uv` package manager must already be installed.
The tested environment uses Python 3.13.3 and `uv` 0.8.11.
```bash
cd /path/to/naqsim
uv sync
```

Run all commands below from the following directory.
```bash
cd src/tsc_naa_sim/artifact_evaluation
```

Figures 17 and 19 additionally consume Quration data.
For an independent full simulation, complete that prerequisite before invoking the full-simulation workflow.

## Experiment workflow

### 1. Fast validation from archived data

This step reads the archived CSV files under `raw_data/` and writes new outputs
under each target's `results/` directory.
This is the recommended first test and takes less than one minute on a laptop-class CPU.
It does not run the architecture simulator or Quration.
The following command writes the CSVs and figures:

```bash
./run_replotting.sh
```

The shared inputs are organized by figure and producing stage under
[`raw_data/`](raw_data/README.md).
This input tree is separate from the default analysis outputs generated in Step 2 under `<target>/results/`.

Fig. 10(a) and (b) can be reproduced directly from the shared decoding data:

```bash
uv run python fig10/analyze.py
uv run python fig10/fit_logical_error_model.py
```

The second command refits the Fig. 10(b) model and verifies the resulting coefficients against the active assignments in `logical_error_model.py`.

### 2. Full simulation

An independent end-to-end full simulation consists of the Quration prerequisite followed by the architecture-simulation workflow.
Generate the Quration results before invoking `run_full_simulation.sh`.

#### 2-1. Quration prerequisite

Quration is built from source.
The tested Quration commit is `293912c18ee6`.
Clone it before running the AE scripts; replace `/path/to/work` with an absolute path on the evaluation machine:

```bash
mkdir -p /path/to/work
git clone https://github.com/quration/quration.git /path/to/work/quration
cd /path/to/work/quration
git checkout 293912c18ee6
```

Install the build dependencies and build Quration by following the [Quration build instructions](https://github.com/quration/quration#install-quration-core-and-quration-algorithm) in its `README.md`.
With the current CMake presets, the resulting executable is located at the following path on Linux and macOS:

```text
/path/to/work/quration/build/main/qret
```

Return to the artifact-evaluation directory before running the prerequisite:

```bash
cd /path/to/naqsim/src/tsc_naa_sim/artifact_evaluation
```

`quration/run.sh` accepts the path to an external `qret` executable.
It runs the Quration calculations needed by Figs. 17 and 19 and then generates their current-format CSV files.

The following command writes the required CSVs for reproducing Figs. 17 and 19 under `quration/results/`:

```bash
./quration/run.sh --quration-bin /path/to/work/quration/build/main/qret
```

The following CSVs are generated under `quration/results/`:

- `fig17/quration_eval_summary.csv`
- `fig19/trace_ms_quration_eval_summary.csv`
- `fig19/quration_trace_ms_overhead_sweep.csv`
- `fig19/quration_trace_ms_overhead_summary.csv`

Use `--target fig17` or `--target fig19` to generate data for only one figure.
The Fig. 19 Quration workload set is Hei, FH, Jellium, and H4.
Its trace Monte Carlo post-processing uses 10 shots by default; this value can be changed with `--num-shots`.

#### 2-2. Run the complete simulation

**NOTE: This step takes approximately 30 hours even on our tested high-performance server, which has an AMD EPYC 9684X processor and 768 GB of memory. We recommend the optional workflow in Section 2-3 or 2-4.**


After the complete Quration prerequisite completes, run:
```bash
./run_full_simulation.sh
```

This is the standard full-simulation entry point.
It selects every workload and code distance from `config/workloads.csv`, runs all unique compilations
first, then all unique executions and target-specific analyses, and finally invokes `run_full_rendering.sh`
to render every paper-facing figure and table.
Shared compilation/execution results are reused across targets through the content-addressed cache (default `.cache/` in this directory).

By default, Figs. 17 and 19 read the Quration prerequisite from `quration/results/`.
If `quration/run.sh` was given `--output-dir DIR`, invoke the simulation as `./run_full_simulation.sh --quration-results-dir DIR`.
To intentionally use the accepted Quration CSVs supplied in `raw_data/`, pass `--quration-results-dir raw_data` explicitly.

Each analysis writes current-format raw-data and aggregated CSV files, together with `PDF`/`PNG` figures, under `<target>/results/` by default.
Use `--output-dir DIR` on an individual analyzer to select a different destination.
Fig. 18 includes all eight workloads and uses the same Table 2 `transversal_distance` column as every other figure.
For a fresh Fig. 19 simulation, each workload's `*_sweep.csv` is required.
The renderer normalizes each workload before taking the geometric mean, which preserves the Pareto points shown in the paper.

The Fig. 19 D3-ROT resource-state calculation is Monte Carlo-based.
A fresh run can therefore differ slightly from the accepted-version data.
When the complete shared four-workload paper dataset is supplied, `analyze.py` uses the archived, unrounded `ms_overhead_geomean_sweep.csv` so that the
accepted-version Pareto frontier is reproduced exactly.

#### (Optional) 2-3. Workload-wise execution

To check one benchmark independently, select one of `hei`, `fh`, `jellium`, `h4`, `adder`, `qft`, `ising1d`, or `ising2d`.

```bash
WORKLOAD=h4

./quration/run.sh \
  --quration-bin /path/to/work/quration/build/main/qret \
  --workload "$WORKLOAD"
./run_full_simulation.sh "$WORKLOAD"
```

The one-workload form runs all applicable compilation, execution, and simulation-analysis stages.
The above command omits the paper-facing renderer step because the paper figures and table aggregate multiple workloads.

For a complete workload-wise reproduction, first generate the full Quration prerequisite as described in Step 2-1, without `--workload`. Then run `run_full_simulation.sh` once for each of the eight workload selectors.
After all eight simulation and analysis runs have completed, render the complete figures and table once:

```bash
./run_full_rendering.sh
```

This command reads the combined simulator outputs produced by the workload-wise runs and the Quration CSVs under `quration/results/`; it does not rerun compilation or execution.
If Quration used a different output root, pass `--quration-results-dir DIR` to `run_full_rendering.sh`.

#### (Optional) 2-4. Slurm execution

For a Slurm cluster, first adapt the `FIXME` partition and wall-time values and the virtual-environment activation in the supplied batch templates.
Add appropriate CPU and memory directives for the local cluster.
After generating the complete Quration prerequisite (step 2-1), submit one workload per array task.
Submit the rendering job only after all eight array tasks finish successfully:

```bash
mkdir -p logs
SIM_JOB_ID=$(sbatch --parsable batch_run_full_simulation.sh)
sbatch --dependency=afterok:"${SIM_JOB_ID%%;*}" batch_run_full_rendering.sh
```

`params_full_simulation.txt` assigns 12, 16, 12, 4, 4, 4, 4, and 2 workers to Hei, FH, Jellium, H4, adder, QFT, Ising1d, and Ising2d, respectively.
These are the workload-specific concurrency limits in `config/workloads.csv`.
The template's 600 GiB `AE_MEMORY_BUDGET_GIB` default limits the adaptive pipeline controller; it does not request memory from Slurm.
The controller can still reduce the worker count when a batch would exceed that budget.

#### (Optional) 2-5. Manual stage-by-stage execution

For debugging, separate batch allocations, or inspection of a selected workload's analysis, run a target one stage at a time.
For example,
```bash
WORKLOAD=h4

./fig11a/run.sh --workload "$WORKLOAD" --stage compilation
./fig11a/run.sh --workload "$WORKLOAD" --stage execution
./fig11a/run.sh --workload "$WORKLOAD" --stage analysis
uv run python fig11a/analyze.py --workload "$WORKLOAD"
```

`--stage all` (the default) performs the same three stages in order.
The same commands apply to every target after replacing `fig11a` with one of `table3`, `fig15`, `fig16`, `fig17`, `fig18`, `fig19`, or `fig20`.
Keep the same `--workload` selector on the Fig. 17/19 analysis commands so that they validate and consume the matching Quration CSV generated by the prerequisite above.
Compilation and execution alone do not require Quration.

`run_full_simulation.sh` uses the following reuse-friendly order internally.
The equivalent commands below are useful when compilation, execution, and
analysis must be assigned to separate jobs or allocations:

```bash
# 1. Compilation producers. Table 3 shares fig11a/comp_opt; Fig. 16/17 share
#    fig11a+fig15; Fig. 19 shares fig15, so those consumers are omitted here.
./fig11a/run.sh --stage compilation
./fig15/run.sh  --stage compilation
./fig18/run.sh  --stage compilation
./fig20/run.sh  --stage compilation

# 2. Execution producers, in the same reuse-friendly order.
./fig11a/run.sh --stage execution
./fig15/run.sh  --stage execution
./fig18/run.sh  --stage execution
./fig20/run.sh  --stage execution

# 3. The full-workload Quration prerequisite has already been generated.
#    Generate the simulator analysis CSVs consumed by the renderers.
./fig11a/run.sh --stage analysis
./table3/run.sh --stage analysis
./fig15/run.sh  --stage analysis
./fig16/run.sh  --stage analysis
./fig17/run.sh  --stage analysis
./fig19/run.sh  --stage analysis
./fig18/run.sh  --stage analysis
./fig20/run.sh  --stage analysis

# 4. Render the paper-facing outputs from those analysis CSVs.
./run_full_rendering.sh
```

Useful options for the per-target `run.sh` scripts are:

- `--stage compilation|execution|analysis|all`: select a separated stage.
- `--dry-run`: show the stage commands without executing them.
- `--force`: rerun analysis even when its analysis CSV exists; cached simulator results remain reusable.
- `--num-threads N`: override the per-workload setting.
- `--memory-budget-gib N`: cap aggregate compilation/execution worker memory. Without it, the controller uses 75% of the smaller of host-available and cgroup-available memory.
- `--cache-dir DIR`: place the shared cache outside its default `.cache/` root.
- `--num-shots N`: set Fig. 19 D3-ROT Monte Carlo shots (default: 10).
- `--quration-results-dir DIR`: use a non-default prerequisite root for Fig. 17/19 (default: `quration/results/`).

`run_full_rendering.sh` accepts `--quration-results-dir DIR` and `--dry-run`.
