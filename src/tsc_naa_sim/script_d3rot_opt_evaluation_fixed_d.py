
## import & setup
####
import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(curr_dir) # src
os.chdir(curr_dir)
subpaths = [os.path.join(par_dir, d) for d in os.listdir(par_dir)]
subdirs = [d for d in subpaths if os.path.isdir(d)]
sys.path.extend(subdirs)
#
from macro import *
from copy import deepcopy
import qiskit.qasm2
from qiskit import QuantumCircuit, transpile


from config import experiment_config
from plane_initializer import get_rectangle
from precomputed_mapping_io import (
    DEFAULT_PRECOMPUTED_MAPPING_DIR,
    load_precomputed_qmaps,
)

import argparse

def find_qasm_path(ham_name):
    qasm_root = os.path.join(par_dir, "benchmarks", "qasm")
    direct_path = os.path.join(qasm_root, f"{ham_name}.qasm")
    if os.path.isfile(direct_path):
        return direct_path

    matches = []
    for root, _, files in os.walk(qasm_root):
        if f"{ham_name}.qasm" in files:
            matches.append(os.path.join(root, f"{ham_name}.qasm"))
    if not matches:
        raise FileNotFoundError(f"QASM file for {ham_name} was not found under {qasm_root}")
    if len(matches) > 1:
        raise RuntimeError(f"QASM file for {ham_name} is ambiguous: {matches}")
    return matches[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch simulations for shuttle circuits.")

    parser.add_argument("--ham-name", type=str, default="SELECT_10_Heisenberg1D_OBC_1_0",
                        help="Target hamiltonian name for QPE (SELECT).")
    parser.add_argument("--use-naive-mapping", action="store_true",
                   help="use naive qubit mapping without SA.")
    parser.add_argument("--use-precomputed-mapping", action="store_true",
                   help="use qmap_init saved by script_precompute_jit_mappings_fixed_d.py.")
    parser.add_argument("--precomputed-mapping-dir", type=str,
                        default=str(DEFAULT_PRECOMPUTED_MAPPING_DIR),
                        help="directory containing precomputed mapping manifest.json.")
    parser.add_argument("--code-distance", type=int, default=25,
                        help="code distance of each SC patch.")
    parser.add_argument("--num-threads", "--num_threads", dest="num_threads",
                        type=int, default=os.cpu_count(),
                        help="number of worker processes. Defaults to os.cpu_count().")

    return parser.parse_args()


# ## 1. Check each benchmark one by one

# target_qc: Qiskit.Circuit


#qc_in.draw('mpl')
def evaluate_qpe(
    ham_name: str,
    code_dist: int = 25,
    use_naive_mapping: bool = False,
    num_threads=None,
    use_precomputed_mapping: bool = False,
    precomputed_mapping_dir: str = str(DEFAULT_PRECOMPUTED_MAPPING_DIR),
    experiment_runner=None,
):
    if experiment_runner is None:
        raise RuntimeError(
            "No experiment runner was provided. Use artifact_evaluation/run_full_simulation.sh "
            "or a per-figure run.sh script."
        )
    if num_threads is None:
        num_threads = os.cpu_count()
    if use_naive_mapping and use_precomputed_mapping:
        raise ValueError("--use-naive-mapping and --use-precomputed-mapping are mutually exclusive.")


    #ham_name = "SELECT_4_FermiHubbard2D_cylinder_0_0"
    #ham_name = "SELECT_10_Heisenberg1D_OBC_1_0"
    qc_name = ham_name

    ### load QASM file
    qc_orig = qiskit.qasm2.load(find_qasm_path(ham_name), custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    basis_gates = ['cx', 'h', 's', 't', 'x', 'y', 'z']
    #if ham_name.find("FTCBench") > -1:
    if ham_name.find("err1em3") > -1:
        qc_decomposed = qc_orig
    else:
        qc_decomposed = transpile(qc_orig, basis_gates=basis_gates)

    ### add initialization
    reset_qc = QuantumCircuit(qc_decomposed.num_qubits, qc_decomposed.num_clbits)
    reset_qc.reset(range(reset_qc.num_qubits))
    qc_with_init = reset_qc.compose(qc_decomposed)

    ### add reset after mid-circuit measurement
    qc_with_meas_reset = QuantumCircuit(qc_with_init.num_qubits, qc_with_init.num_clbits)
    for inst in qc_with_init.data:
        qc_with_meas_reset.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == 'measure':
            for qubit in inst.qubits:
                qc_with_meas_reset.reset(qubit)

    qc_in = qc_with_meas_reset

    # ### Input quantum circuits

    input_qc_dict = dict()
    input_qc_dict[qc_name] = qc_in

    # ### Target configurations

    target_cfg_dict = dict()
    #
    run_opts = []
    run_opts.append(RunOpt.IGNORE_NONE)
    run_opts.append(RunOpt.IGNORE_PC_ROT)
    run_opts.append(RunOpt.IGNORE_ROT)
    run_opts.append(RunOpt.IGNORE_PC)

    # 1. REFL_SE
    cfg_name = "REFL_SE"
    cfg_in_refl_se = experiment_config()
    cfg_in_refl_se.code_dist = code_dist
    ###
    cfg_in_refl_se.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_refl_se.rot_sched_opt = RotSchedOpt.AGGREGATE
    ###
    cfg_in_refl_se.cell_size = CellSize.SMALLEST
    cfg_in_refl_se.rot_type = RotType.REFL
    cfg_in_refl_se.refl_type_h = ReflType.STATIC_SE
    cfg_in_refl_se.refl_type_d = ReflType.STATIC_SE
    cfg_in_refl_se.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_refl_se.aod_sched_opt = AodSchedOpt.NAIVE_DRAIN
    ###
    target_cfg_dict[cfg_name] = (cfg_in_refl_se, run_opts)


    ## 2. REFL_TE
    cfg_name = "REFL_TE"
    cfg_in_refl_te = experiment_config()
    cfg_in_refl_te.code_dist = code_dist
    ###
    cfg_in_refl_te.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_refl_te.rot_sched_opt = RotSchedOpt.AGGREGATE
    ###
    cfg_in_refl_te.cell_size = CellSize.DOUBLE_TE
    cfg_in_refl_te.rot_type = RotType.REFL
    cfg_in_refl_te.refl_type_h = ReflType.STATIC_TE
    cfg_in_refl_te.refl_type_d = ReflType.STATIC_TE
    cfg_in_refl_te.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_refl_te.aod_sched_opt = AodSchedOpt.NAIVE_DRAIN
    ###
    target_cfg_dict[cfg_name] = (cfg_in_refl_te, run_opts)


    ## 3. DIR_CHANGE
    cfg_name = "DIR_CHANGE"
    cfg_in_dir_change = experiment_config()
    cfg_in_dir_change.code_dist = code_dist
    ###
    cfg_in_dir_change.cell_size = CellSize.DOUBLE_DIR
    cfg_in_dir_change.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_change.rot_sched_opt = RotSchedOpt.AGGREGATE
    ###
    cfg_in_dir_change.rot_type = RotType.DIR_CHANGE
    cfg_in_dir_change.refl_type_h = None
    cfg_in_dir_change.refl_type_d = None
    cfg_in_dir_change.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_dir_change.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    ###
    target_cfg_dict[cfg_name] = (cfg_in_dir_change, run_opts)

    # 4. DIR_TOGL
    cfg_name = "DIR_TOGL"
    cfg_in_dir_togl = experiment_config()
    cfg_in_dir_togl.code_dist = code_dist
    ###
    cfg_in_dir_togl.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl.rot_sched_opt = RotSchedOpt.AGGREGATE
    ###
    cfg_in_dir_togl.cell_size = CellSize.DOUBLE_DIR
    cfg_in_dir_togl.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl.refl_type_h = None
    cfg_in_dir_togl.refl_type_d = None
    cfg_in_dir_togl.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_dir_togl.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    ###
    target_cfg_dict[cfg_name] = (cfg_in_dir_togl, run_opts)


    # 5-1. DIR_TOGL + DEDICATE CELL1
    cfg_name = "DIR_TOGL+DEDICATE_CELL1"
    cfg_in_dir_togl_dedicate = experiment_config()
    cfg_in_dir_togl_dedicate.code_dist = code_dist
    ###
    cfg_in_dir_togl_dedicate.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl_dedicate.rot_sched_opt = RotSchedOpt.DISTRIBUTE # NOTE
    ###
    cfg_in_dir_togl_dedicate.cell_size = CellSize.SMALLEST
    cfg_in_dir_togl_dedicate.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl_dedicate.refl_type_h = None
    cfg_in_dir_togl_dedicate.refl_type_d = None
    cfg_in_dir_togl_dedicate.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    cfg_in_dir_togl_dedicate.num_rot_cell = 1
    #
    cfg_in_dir_togl_dedicate.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    ###
    target_cfg_dict[cfg_name] = (cfg_in_dir_togl_dedicate, run_opts)

    # 5-2. DIR_TOGL + DEDICATE CELL2
    cfg_name = "DIR_TOGL+DEDICATE_CELL2"
    cfg_in_dir_togl_dedicate = experiment_config()
    cfg_in_dir_togl_dedicate.code_dist = code_dist
    ###
    cfg_in_dir_togl_dedicate.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl_dedicate.rot_sched_opt = RotSchedOpt.DISTRIBUTE # NOTE
    ###
    cfg_in_dir_togl_dedicate.cell_size = CellSize.SMALLEST
    cfg_in_dir_togl_dedicate.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl_dedicate.refl_type_h = None
    cfg_in_dir_togl_dedicate.refl_type_d = None
    cfg_in_dir_togl_dedicate.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    cfg_in_dir_togl_dedicate.num_rot_cell = 2
    #
    cfg_in_dir_togl_dedicate.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    ###
    target_cfg_dict[cfg_name] = (cfg_in_dir_togl_dedicate, run_opts)

    # 5-3. DIR_TOGL + DEDICATE CELL3
    cfg_name = "DIR_TOGL+DEDICATE_CELL3"
    cfg_in_dir_togl_dedicate = experiment_config()
    cfg_in_dir_togl_dedicate.code_dist = code_dist
    ###
    cfg_in_dir_togl_dedicate.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_togl_dedicate.rot_sched_opt = RotSchedOpt.DISTRIBUTE # NOTE
    ###
    cfg_in_dir_togl_dedicate.cell_size = CellSize.SMALLEST
    cfg_in_dir_togl_dedicate.rot_type = RotType.DIR_TOGL
    cfg_in_dir_togl_dedicate.refl_type_h = None
    cfg_in_dir_togl_dedicate.refl_type_d = None
    cfg_in_dir_togl_dedicate.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    cfg_in_dir_togl_dedicate.num_rot_cell = 3
    #
    cfg_in_dir_togl_dedicate.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    ###
    target_cfg_dict[cfg_name] = (cfg_in_dir_togl_dedicate, run_opts)


    # 6. DIR_IDEAL
    cfg_name = "DIR_IDEAL"
    cfg_in_dir_ideal = experiment_config()
    cfg_in_dir_ideal.code_dist = code_dist
    ###
    cfg_in_dir_ideal.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg_in_dir_ideal.rot_sched_opt = RotSchedOpt.AGGREGATE
    ###
    cfg_in_dir_ideal.cell_size = CellSize.DOUBLE_DIR
    cfg_in_dir_ideal.rot_type = RotType.DIR_IDEAL
    cfg_in_dir_ideal.refl_type_h = None
    cfg_in_dir_ideal.refl_type_d = None
    cfg_in_dir_ideal.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg_in_dir_ideal.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    ###
    target_cfg_dict[cfg_name] = (cfg_in_dir_ideal, run_opts)

    # ### Output directories
    if use_naive_mapping:
        mapping_prefix = "naive_mapping_"
    elif use_precomputed_mapping:
        mapping_prefix = "precomputed_mapping_"
    else:
        mapping_prefix = ""

    base_dir = os.path.join(curr_dir, f"output/d3rot_opt_evaluation_fixed_d/{mapping_prefix}{ham_name}_distance{code_dist}")
    os.makedirs(base_dir, exist_ok=True)
    #
    outdir_dict = dict()
    for qc_name in input_qc_dict.keys():
        if not qc_name in outdir_dict.keys():
            outdir_dict[qc_name] = dict()
        for cfg_name in target_cfg_dict.keys():
            outdir_dict[qc_name][cfg_name] = os.path.join(base_dir, f"{qc_name}_{cfg_name}")
    print(outdir_dict)

    # ### Naive qubit mapping
    width, height = get_rectangle(n=qc_in.num_qubits, max_diff=2)
    naive_map = dict()
    index = 0
    for h in range(height):
        for w in range(width):
            naive_map[f'Q{index}'] = (h+2, w)
            index += 1
            if index >= qc_in.num_qubits:
                break

    # ### Run

    ############
    if use_precomputed_mapping:
        qmap_in = load_precomputed_qmaps(
            script_name="d3rot_opt",
            ham_name=ham_name,
            target_cfg_dict=target_cfg_dict,
            mapping_dir=precomputed_mapping_dir,
        )
        print(f"[INFO] use precomputed mappings from {precomputed_mapping_dir}")
    elif use_naive_mapping:
        qmap_in = naive_map
    else:
        qmap_in = None

    experiment_runner(
        input_qc_dict,
        target_cfg_dict,
        outdir_dict,
        num_threads=num_threads,
        qmap_in=qmap_in,
    )

if __name__ == '__main__':
    args = parse_args()
    evaluate_qpe(args.ham_name,
                 code_dist=args.code_distance,
                 use_naive_mapping=args.use_naive_mapping,
                 num_threads=args.num_threads,
                 use_precomputed_mapping=args.use_precomputed_mapping,
                 precomputed_mapping_dir=args.precomputed_mapping_dir)
