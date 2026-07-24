
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
from itertools import product
import qiskit.qasm2
from qiskit import QuantumCircuit, transpile


from config import experiment_config
from plane_initializer import get_rectangle
from precomputed_mapping_io import (
    DEFAULT_PRECOMPUTED_MAPPING_DIR,
    load_precomputed_qmaps,
)
from misc import getJsonData

import argparse


HWCFG_NAME_LIST = [
    "sensitivity_hwcfg_togl1_trf40.json",
    "sensitivity_hwcfg_togl10_trf40.json",
    "sensitivity_hwcfg_togl1_trf120.json",
    "sensitivity_hwcfg_togl10_trf120.json",
]


def normalize_ham_name(ham_name):
    ham_name = os.path.basename(ham_name)
    if ham_name.endswith(".qasm"):
        return ham_name[:-len(".qasm")]
    return ham_name


def find_qasm_path(ham_name):
    ham_name = normalize_ham_name(ham_name)
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


def load_qc_for_ham(ham_name):
    ### load QASM file
    qc_orig = qiskit.qasm2.load(find_qasm_path(ham_name), custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    basis_gates = ['cx', 'h', 's', 't', 'x', 'y', 'z']
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

    return qc_with_meas_reset


def make_target_cfg_dict(code_dist):
    target_cfg_dict = dict()
    run_opts = []
    run_opts.append(RunOpt.IGNORE_NONE)

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
    d3rot_cfg.rot_sched_opt = RotSchedOpt.DISTRIBUTE # NOTE
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


# ## 1. Check each benchmark one by one

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

    ham_name = normalize_ham_name(ham_name)
    qc_name = ham_name
    qc_in = load_qc_for_ham(ham_name)

    # ### Input quantum circuits
    input_qc_dict = dict()
    input_qc_dict[qc_name] = qc_in

    # ### Target configurations
    target_cfg_dict = make_target_cfg_dict(code_dist)

    # ### Output directories
    if use_naive_mapping:
        mapping_prefix = "naive_mapping_"
    elif use_precomputed_mapping:
        mapping_prefix = "precomputed_mapping_"
    else:
        mapping_prefix = ""

    base_dir = os.path.join(curr_dir, f"output/sensitivity_test_evaluation_fixed_d/{mapping_prefix}{ham_name}_distance{code_dist}")
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
            script_name="sensitivity_test",
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
