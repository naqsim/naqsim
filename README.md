# NAQsim

NAQsim is a full-stack architecture simulation framework for fault-tolerant quantum computing (FTQC) with neutral atoms.
It models quantum-circuit compilation and architecture execution to evaluate metrics such as execution time, logical-qubit footprint, and logical error rate.

Use NAQsim to explore design choices across the compiler, control hardware, and qubit plane.
The framework has three stages:

- **Compilation:** the instruction translator, instruction scheduler, and logical-qubit (LQ) mapper produce scheduled instruction layers and an initial LQ mapping for the input quantum program, logical instruction set, and qubit plane configuration.
- **Execution:** scheduled instructions are decomposed into laser-wise micro-operations and simulated under the control hardware configuration. This includes AOD shuttling, patch rotation, route conflicts, and error syndrome measurement (ESM), while tracking LQ status and location.
- **Estimation:** execution traces and input configurations determine execution time, physical-qubit count, and qubit-plane footprint. Program success probability is estimated by accumulating logical error rates (LERs).

## Installation

NAQsim requires Python 3.13 or later and the `uv` package manager.

```bash
git clone https://github.com/naqsim/naqsim.git
cd naqsim
uv sync
```

## Run a small simulation

The simulator can be used directly from Python.
Run the following example from `src/tsc_naa_sim`: the simulator modules and default control hardware configuration file are resolved from this directory.
Starting at the repository root:

```bash
cd src/tsc_naa_sim
uv run python - <<'PY'
from qiskit import QuantumCircuit

from config import experiment_config
from macro import RunOpt
from plane_initializer import get_rectangle
from run_analysis_scripts import (
    compilation_stage,
    execution_stage,
    get_finish_us_from_uop_schedule,
)

qc = QuantumCircuit(2, 2)
qc.reset(range(2))
qc.h(0)
qc.t(0)
qc.cx(0, 1)
qc.measure(range(2), range(2))

cfg = experiment_config()
cfg.code_dist = 15

# Place LQs in row-major order on the default ALLQ qubit plane.
width, _ = get_rectangle(qc.num_qubits, max_diff=2)
qmap = {f'Q{i}': (2 + i // width, i % width) for i in range(qc.num_qubits)}

compiled = compilation_stage(qc, cfg, qmap_in=qmap)
executed = execution_stage(compiled, cfg, RunOpt.IGNORE_NONE)
uop_trace, plane_trace, qregs_trace = executed
runtime_us = sum(get_finish_us_from_uop_schedule(layer) for layer in uop_trace)

print(f'Instruction layers: {len(compiled[2])}')
print(f'Estimated execution time: {runtime_us:.2f} us')
PY
```

### Use your own circuit

To use an OpenQASM 2 circuit, replace the circuit-construction portion of the example with the following code, keeping the configuration, initial LQ mapping, and simulation steps:

```python
import qiskit.qasm2
from qiskit import transpile

source = qiskit.qasm2.load(
    '/path/to/circuit.qasm',
    custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
)
source = transpile(
    source,
    basis_gates=['cx', 'h', 's', 't', 'x', 'y', 'z'],
    optimization_level=0,
)

qc = QuantumCircuit(source.num_qubits, source.num_clbits)
qc.reset(range(qc.num_qubits))
for item in source.data:
    qs = [source.find_bit(q).index for q in item.qubits]
    cs = [source.find_bit(c).index for c in item.clbits]
    qc.append(item.operation, qs, cs)
    if item.operation.name == 'measure':
        qc.reset(qs)
```

### Configure and inspect a run

[config.py](src/tsc_naa_sim/config.py) defines `experiment_config`, and [macro.py](src/tsc_naa_sim/macro.py) defines the available option values.
Set configuration fields before compilation and execution:

| Settings | Purpose |
| --- | --- |
| `code_dist`, `rounds` | Code distance and ESM rounds per scheduled layer. |
| `hw_cfg` | Control hardware parameters for physical-operation latency and error rates; defaults to [case_study_hwcfg.json](src/tsc_naa_sim/case_study_hwcfg.json). |
| `rot_trans_opt`, `rot_sched_opt` | Compiler policies for patch rotation. |
| `rot_type`, `rot_plane_opt`, `num_rot_cell` | Patch rotation implementation and allocation of dedicated rotation patches. |
| `is_aod_infinite`, `num_aodh_max`, `num_aodd_max`, `num_aodr_max` | AOD laser availability. |

The compilation result contains the logical instruction DAG, qubit plane layout, instruction and resource-state request traces, and initial LQ mapping.
The execution result contains per-layer execution traces of physical laser operations, qubit plane states, and LQ states.
The analysis functions in [run_analysis_scripts.py](src/tsc_naa_sim/run_analysis_scripts.py) consume these results for execution-time breakdowns and LER estimates.

## Run bundled benchmarks

Example OpenQASM circuits are provided in [src/benchmarks/qasm](src/benchmarks/qasm).
The batch wrappers under `artifact_evaluation` can also run selected benchmarks and compare compiler configurations.
For example, from the repository root:

```bash
cd src/tsc_naa_sim/artifact_evaluation
./fig11a/run.sh --workload qft --stage all --num-threads 1 --force
uv run python fig11a/analyze.py \
  --workload qft \
  --input-dir ../output/comp_opt_evaluation_fixed_d \
  --output-dir fig11a/results
```

The first command runs compilation, execution, and estimation for the bundled QFT workload; the second renders its execution-time breakdown from the newly generated data.
`--force` regenerates the analysis while allowing cached compilation and execution results to be reused.
Simulation outputs are stored under `src/tsc_naa_sim/output/`, shared compilation/execution caches under `artifact_evaluation/.cache/`, and the rendered CSV/PDF/PNG files under `artifact_evaluation/fig11a/results/`.
Add `--dry-run` to the wrapper command to inspect its stages before running a benchmark.
These runs are larger than the two-qubit example; runtime and memory requirements depend on the circuit and configuration.

## MICRO 2026 Artifact Evaluation

This repository includes the simulation, analysis, and archived data for the NAQsim paper accepted at MICRO 2026.
The artifact-evaluation snapshot is available at the [`v1.1` tag](https://github.com/naqsim/naqsim/tree/v1.1); ongoing development can include changes beyond that snapshot.

Detailed setup, execution, and result-reproduction instructions are available in the [Artifact Evaluation README](src/tsc_naa_sim/artifact_evaluation/README.md).
That guide covers replotting the archived results, full simulation, Quration prerequisites, and Slurm execution.

## Citation

If you use NAQsim in your research, please cite the associated MICRO 2026 paper and the software release you used.
Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

NAQsim is released under the [MIT License](LICENSE).

Copyright (c) 2026 Kyushu University
Copyright (c) 2026 Nanofiber Quantum Technologies, Inc.
