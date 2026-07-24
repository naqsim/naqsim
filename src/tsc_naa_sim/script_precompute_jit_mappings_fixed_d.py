import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent
QASM_ROOT = SRC_DIR / "benchmarks" / "qasm"
ORIGINAL_CWD = Path.cwd()

os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

import qiskit.qasm2
import zstandard as zstd
from qiskit import QuantumCircuit, transpile

from config import experiment_config
from misc import getJsonData
from macro import *
from plane_initializer import plane_initializer
from tsc_inst_scheduler import tsc_inst_scheduler
from tsc_inst_translator import tsc_inst_translator
from tsc_qubit_mapper_jit import NUMBA_AVAILABLE, sa_mapper

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_HAM_NAMES = [
    "SELECT_6_FermiHubbard2D_cylinder_0_0",
    "SELECT_2_Jellium3D_OBC_0_0",
    "SELECT_4_Hydrogen2x2_OBC_0_0",
    "SELECT_12_Heisenberg2D_cylinder_0.5_0.5",
    "adder_64q_basis_err1em3",
    "ising_1d_64q_basis_err1em3",
    "ising_2d_64q_basis_err1em3",
    "qft_29q_basis_err1em3",
]

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "precomputed_jit_mappings_fixed_d"
SENSITIVITY_HWCFG_NAME_LIST = [
    "sensitivity_hwcfg_togl1_trf40.json",
    "sensitivity_hwcfg_togl10_trf40.json",
    "sensitivity_hwcfg_togl1_trf120.json",
    "sensitivity_hwcfg_togl10_trf120.json",
]


@contextmanager
def suppress_stdout(enabled=True):
    if not enabled:
        yield
        return

    stdout = sys.stdout
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout


def enum_name(value):
    return value.name if hasattr(value, "name") else str(value)


def qasm_sha256(qasm_path):
    digest = hashlib.sha256()
    with open(qasm_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_qasm_path(ham_name):
    matches = sorted(QASM_ROOT.rglob(f"{ham_name}.qasm"))
    if not matches:
        raise FileNotFoundError(f"QASM file for {ham_name} was not found under {QASM_ROOT}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"QASM file for {ham_name} is ambiguous: {paths}")
    return matches[0]


def load_qc_for_ham(ham_name):
    qasm_path = find_qasm_path(ham_name)
    qc_orig = qiskit.qasm2.load(
        str(qasm_path),
        custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )

    basis_gates = ["cx", "h", "s", "t", "x", "y", "z"]
    if "err1em3" in ham_name:
        qc_decomposed = qc_orig
    else:
        qc_decomposed = transpile(qc_orig, basis_gates=basis_gates)

    reset_qc = QuantumCircuit(qc_decomposed.num_qubits, qc_decomposed.num_clbits)
    reset_qc.reset(range(reset_qc.num_qubits))
    qc_with_init = reset_qc.compose(qc_decomposed)

    qc_with_meas_reset = QuantumCircuit(qc_with_init.num_qubits, qc_with_init.num_clbits)
    for inst in qc_with_init.data:
        qc_with_meas_reset.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == "measure":
            for qubit in inst.qubits:
                qc_with_meas_reset.reset(qubit)

    return qc_with_meas_reset, qasm_path


def build_comp_opt_cfgs(code_distance):
    cfgs = OrderedDict()

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfgs["Baseline"] = cfg

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfgs["Skip"] = cfg

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfgs["Skip+Aggr"] = cfg

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg.rot_sched_opt = RotSchedOpt.DISTRIBUTE
    cfgs["Skip+Dist"] = cfg

    return cfgs


def build_d3rot_cfgs(code_distance):
    cfgs = OrderedDict()

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg.cell_size = CellSize.SMALLEST
    cfg.rot_type = RotType.REFL
    cfg.refl_type_h = ReflType.STATIC_SE
    cfg.refl_type_d = ReflType.STATIC_SE
    cfg.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg.aod_sched_opt = AodSchedOpt.NAIVE_DRAIN
    cfgs["REFL_SE"] = cfg

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg.cell_size = CellSize.DOUBLE_TE
    cfg.rot_type = RotType.REFL
    cfg.refl_type_h = ReflType.STATIC_TE
    cfg.refl_type_d = ReflType.STATIC_TE
    cfg.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg.aod_sched_opt = AodSchedOpt.NAIVE_DRAIN
    cfgs["REFL_TE"] = cfg

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.cell_size = CellSize.DOUBLE_DIR
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg.rot_type = RotType.DIR_CHANGE
    cfg.refl_type_h = None
    cfg.refl_type_d = None
    cfg.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    cfgs["DIR_CHANGE"] = cfg

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg.cell_size = CellSize.DOUBLE_DIR
    cfg.rot_type = RotType.DIR_TOGL
    cfg.refl_type_h = None
    cfg.refl_type_d = None
    cfg.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    cfgs["DIR_TOGL"] = cfg

    for num_rot_cell in [1, 2, 3]:
        cfg = experiment_config()
        cfg.code_dist = code_distance
        cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
        cfg.rot_sched_opt = RotSchedOpt.DISTRIBUTE
        cfg.cell_size = CellSize.SMALLEST
        cfg.rot_type = RotType.DIR_TOGL
        cfg.refl_type_h = None
        cfg.refl_type_d = None
        cfg.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
        cfg.num_rot_cell = num_rot_cell
        cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
        cfgs[f"DIR_TOGL+DEDICATE_CELL{num_rot_cell}"] = cfg

    cfg = experiment_config()
    cfg.code_dist = code_distance
    cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    cfg.rot_sched_opt = RotSchedOpt.AGGREGATE
    cfg.cell_size = CellSize.DOUBLE_DIR
    cfg.rot_type = RotType.DIR_IDEAL
    cfg.refl_type_h = None
    cfg.refl_type_d = None
    cfg.rot_plane_opt = RotPlaneOpt.ALL_ROT
    cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    cfgs["DIR_IDEAL"] = cfg

    return cfgs


def build_sensitivity_test_cfgs(code_distance):
    cfgs = OrderedDict()

    arch_cfg_list = []

    base_cfg = experiment_config()
    base_cfg.code_dist = code_distance
    base_cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    arch_cfg_list.append(("Base", base_cfg))

    d3rot_cfg = experiment_config()
    d3rot_cfg.code_dist = code_distance
    d3rot_cfg.rot_trans_opt = RotTransOpt.NAIVE_SKIP
    d3rot_cfg.rot_sched_opt = RotSchedOpt.DISTRIBUTE
    d3rot_cfg.cell_size = CellSize.SMALLEST
    d3rot_cfg.rot_type = RotType.DIR_TOGL
    d3rot_cfg.refl_type_h = None
    d3rot_cfg.refl_type_d = None
    d3rot_cfg.rot_plane_opt = RotPlaneOpt.DEDICATED_ROT
    num_rot_cell = 3
    d3rot_cfg.num_rot_cell = num_rot_cell
    d3rot_cfg.aod_sched_opt = AodSchedOpt.FIRST_FINISH
    arch_cfg_list.append((f"D3ROT_CELL{num_rot_cell}", d3rot_cfg))

    for (arch_cfg_name, arch_cfg), hwcfg_name in product(arch_cfg_list, SENSITIVITY_HWCFG_NAME_LIST):
        hwcfg_fn, _ = os.path.splitext(hwcfg_name)
        temp_name = arch_cfg_name + "_" + "_".join(hwcfg_fn.split("_")[-2:])
        temp_cfg = deepcopy(arch_cfg)
        temp_cfg.hw_cfg = getJsonData(hwcfg_name)

        if temp_cfg.rot_type == RotType.REFL:
            aod_settings = [
                ("aod2", False, False, 1, 1, 1),
                ("aod4", False, False, 2, 2, 1),
                ("aod8", False, False, 4, 4, 1),
                ("aodinf", True, True, None, None, None),
            ]
        elif temp_cfg.rot_type == RotType.DIR_TOGL:
            aod_settings = [
                ("aod2", False, False, 1, 1, 1),
                ("aod4", False, False, 2, 1, 2),
                ("aod8", False, False, 6, 1, 2),
                ("aodinf", True, True, None, None, None),
            ]
        else:
            raise ValueError(f"unsupported rot_type for sensitivity cfg: {temp_cfg.rot_type}")

        for suffix, is_infinite, skip_grouping, num_aodh, num_aodd, num_aodr in aod_settings:
            cfg = deepcopy(temp_cfg)
            cfg.is_aod_infinite = is_infinite
            cfg.skip_uop_grouping = skip_grouping
            cfg.num_aodh_max = num_aodh
            cfg.num_aodd_max = num_aodd
            cfg.num_aodr_max = num_aodr
            cfgs[f"{temp_name}_{suffix}"] = cfg

    return cfgs


def build_script_cfgs(code_distance):
    return {
        "comp_opt": build_comp_opt_cfgs(code_distance),
        "d3rot_opt": build_d3rot_cfgs(code_distance),
        "sensitivity_test": build_sensitivity_test_cfgs(code_distance),
    }


def mapping_signature(ham_name, cfg):
    return {
        "ham_name": ham_name,
        "plane_type": enum_name(cfg.plane_type),
        "rot_trans_opt": enum_name(cfg.rot_trans_opt),
        "inst_sched_opt": enum_name(cfg.inst_sched_opt),
        "rot_sched_opt": enum_name(cfg.rot_sched_opt),
    }


def signature_key(signature):
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in text)


def ham_manifest_path(output_dir, ham_name):
    return output_dir / safe_name(ham_name) / "manifest.json"


def mapping_file_name(signature):
    readable = "__".join(
        [
            safe_name(signature["ham_name"]),
            f"rottrans-{safe_name(signature['rot_trans_opt'])}",
            f"inst-{safe_name(signature['inst_sched_opt'])}",
            f"rotsched-{safe_name(signature['rot_sched_opt'])}",
            f"plane-{safe_name(signature['plane_type'])}",
        ]
    )
    digest = hashlib.sha1(signature_key(signature).encode("utf-8")).hexdigest()[:12]
    return f"{readable}__{digest}.zst"


def atomic_dump_zst(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    compressed = zstd.ZstdCompressor(level=10).compress(pickle.dumps(obj))
    with open(tmp_path, "wb") as f:
        f.write(compressed)
    os.replace(tmp_path, path)


def atomic_dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def load_zst(path):
    with open(path, "rb") as f:
        return pickle.loads(zstd.ZstdDecompressor().decompress(f.read()))


def build_schedule_for_mapping(qc_in, cfg):
    inst_translator = tsc_inst_translator(
        qc_qiskit_in=qc_in,
        rot_trans_opt=cfg.rot_trans_opt,
        s_trans_opt=cfg.s_trans_opt,
    )
    inst_translator.run()
    inst_dag = deepcopy(inst_translator.sc_dag)

    plane_init = plane_initializer(
        num_lq=qc_in.num_qubits,
        plane_type=cfg.plane_type,
    )
    plane_init.run()
    plane_char = plane_init.plane_char

    inst_scheduler = tsc_inst_scheduler(
        sc_dag_in=deepcopy(inst_dag),
        plane_char_in=plane_char,
        inst_sched_opt=cfg.inst_sched_opt,
        rot_sched_opt=cfg.rot_sched_opt,
        s_trans_opt=cfg.s_trans_opt,
        skip_h=cfg.skip_h,
    )
    inst_scheduler.run()
    return plane_char, inst_scheduler.inst_schedule_trace, inst_scheduler.req_schedule_trace


def precompute_one_mapping(job):
    (
        ham_name,
        signature,
        cfg,
        mapping_path,
        force,
        verbose,
    ) = job

    start = time.perf_counter()
    mapping_path = Path(mapping_path)
    if mapping_path.is_file() and not force:
        payload = load_zst(mapping_path)
        if payload.get("mapping_signature") != signature:
            raise ValueError(
                f"existing mapping signature mismatch: path={mapping_path}, "
                f"expected={signature}, payload={payload.get('mapping_signature')}"
            )
        return {
            "status": "skipped",
            "ham_name": ham_name,
            "signature": signature,
            "mapping_path": str(mapping_path),
            "wall_s": 0.0,
            "num_qubits": payload.get("num_qubits"),
            "best_cost": payload.get("best_cost"),
            "current_cost": payload.get("current_cost"),
        }

    with suppress_stdout(not verbose):
        qc_in, qasm_path = load_qc_for_ham(ham_name)
        plane_char, inst_schedule_trace, req_schedule_trace = build_schedule_for_mapping(qc_in, cfg)
        mapper = sa_mapper(
            num_lq=qc_in.num_qubits,
            inst_schedule=inst_schedule_trace,
            plane_char=plane_char,
            use_jit=True,
        )
        mapper.run()

    payload = {
        "schema_version": 1,
        "mapper": "tsc_qubit_mapper_jit.sa_mapper",
        "numba_available": NUMBA_AVAILABLE,
        "created_at_unix": time.time(),
        "ham_name": ham_name,
        "qasm_path": str(qasm_path),
        "qasm_sha256": qasm_sha256(qasm_path),
        "num_qubits": qc_in.num_qubits,
        "mapping_signature": signature,
        "qmap_init": mapper.sa_mapping,
        "random_mapping": mapper.random_mapping,
        "best_cost": mapper.best_cost,
        "current_cost": mapper.current_cost,
        "plane_shape": [len(plane_char), len(plane_char[0]) if plane_char else 0],
        "inst_schedule_layers": len(inst_schedule_trace),
        "req_schedule_layers": len(req_schedule_trace),
    }
    atomic_dump_zst(payload, mapping_path)

    return {
        "status": "created",
        "ham_name": ham_name,
        "signature": signature,
        "mapping_path": str(mapping_path),
        "wall_s": time.perf_counter() - start,
        "num_qubits": qc_in.num_qubits,
        "best_cost": mapper.best_cost,
    }


def build_work_items(ham_names, script_names, code_distance, output_dir, cfg_name_filter):
    script_cfgs = build_script_cfgs(code_distance)
    selected_script_cfgs = {
        script_name: script_cfgs[script_name]
        for script_name in script_names
    }

    unique_jobs = OrderedDict()
    cfg_refs = defaultdict(lambda: defaultdict(dict))
    for ham_name in ham_names:
        for script_name, cfgs in selected_script_cfgs.items():
            for cfg_name, cfg in cfgs.items():
                if cfg_name_filter and cfg_name not in cfg_name_filter:
                    continue
                signature = mapping_signature(ham_name, cfg)
                key = signature_key(signature)
                mapping_path = output_dir / "unique" / mapping_file_name(signature)
                cfg_refs[script_name][ham_name][cfg_name] = {
                    "mapping_signature": signature,
                    "mapping_path": str(mapping_path),
                }
                if key not in unique_jobs:
                    unique_jobs[key] = {
                        "ham_name": ham_name,
                        "signature": signature,
                        "cfg": cfg,
                        "mapping_path": mapping_path,
                    }

    return list(unique_jobs.values()), cfg_refs


def write_cfg_refs(cfg_refs, output_dir):
    by_cfg_dir = output_dir / "by_cfg"
    for script_name, ham_dict in cfg_refs.items():
        for ham_name, cfg_dict in ham_dict.items():
            for cfg_name, ref in cfg_dict.items():
                ref_path = by_cfg_dir / script_name / safe_name(ham_name) / f"{safe_name(cfg_name)}.json"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ref_path, "w", encoding="utf-8") as f:
                    json.dump(ref, f, indent=2, sort_keys=True)


def cfg_refs_for_ham(cfg_refs, ham_name):
    ret = {}
    for script_name, ham_dict in cfg_refs.items():
        if ham_name in ham_dict:
            ret[script_name] = {ham_name: ham_dict[ham_name]}
    return ret


def write_manifest_for_ham(output_dir, args, ham_name, unique_items, cfg_refs, results):
    ham_cfg_refs = cfg_refs_for_ham(cfg_refs, ham_name)
    ham_unique_items = [
        item
        for item in unique_items
        if item["ham_name"] == ham_name
    ]
    ham_results = [
        result
        for result in results
        if result["ham_name"] == ham_name
    ]
    manifest = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "mapper": "tsc_qubit_mapper_jit.sa_mapper",
        "numba_available": NUMBA_AVAILABLE,
        "code_distance": args.code_distance,
        "ham_name": ham_name,
        "ham_names": [ham_name],
        "scripts": args.scripts,
        "unique_mapping_count": len(ham_unique_items),
        "cfg_reference_count": sum(
            len(cfg_dict)
            for ham_dict in ham_cfg_refs.values()
            for cfg_dict in ham_dict.values()
        ),
        "output_dir": str(output_dir),
        "cfg_refs": ham_cfg_refs,
        "results": ham_results,
    }
    manifest_path = ham_manifest_path(output_dir, ham_name)
    atomic_dump_json(manifest, manifest_path)
    return manifest_path


def write_manifests(output_dir, args, unique_items, cfg_refs, results):
    manifest_paths = []
    for ham_name in args.ham_names:
        manifest_paths.append(
            write_manifest_for_ham(output_dir, args, ham_name, unique_items, cfg_refs, results)
        )

    index = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "output_dir": str(output_dir),
        "ham_names": args.ham_names,
        "scripts": args.scripts,
        "manifest_paths": {
            ham_name: str(ham_manifest_path(output_dir, ham_name))
            for ham_name in args.ham_names
        },
    }
    index_path = output_dir / "manifest_index.json"
    atomic_dump_json(index, index_path)
    return manifest_paths, index_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute reusable JIT SA mappings needed by "
            "script_comp_opt_evaluation_fixed_d.py and "
            "script_d3rot_opt_evaluation_fixed_d.py, and "
            "script_sensitivity_test_evaluation_fixed_d.py."
        )
    )
    parser.add_argument(
        "--ham-name",
        dest="ham_names",
        action="append",
        help="Hamiltonian/circuit name. Can be specified multiple times. Defaults to the fixed-d batch.",
    )
    parser.add_argument(
        "--script",
        dest="scripts",
        action="append",
        choices=["comp_opt", "d3rot_opt", "sensitivity_test", "all"],
        default=None,
        help="Target evaluation script family. Defaults to all.",
    )
    parser.add_argument(
        "--cfg-name",
        dest="cfg_names",
        action="append",
        default=None,
        help="Optional cfg-name filter. Can be specified multiple times for testing.",
    )
    parser.add_argument(
        "--code-distance",
        type=int,
        default=25,
        help="Code distance used when reconstructing evaluation cfgs. Mapping keys do not depend on it.",
    )
    parser.add_argument(
        "--num-threads",
        "--num_threads",
        dest="num_threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of parallel workers. Process workers are used by default.",
    )
    parser.add_argument(
        "--parallel-kind",
        choices=["process", "thread"],
        default="process",
        help="Use process workers or thread workers. Process is safer for Python-heavy preprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for unique mapping files, per-cfg references, and manifest.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute mappings even when the unique mapping file already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show internal translator/scheduler/mapper output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deduplicated mapping jobs without running mapper.",
    )

    args, unknown = parser.parse_known_args()
    unknown = [item for item in unknown if item != "-"]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    args.ham_names = list(OrderedDict.fromkeys(args.ham_names or list(DEFAULT_HAM_NAMES)))
    if args.scripts is None or "all" in args.scripts:
        args.scripts = ["comp_opt", "d3rot_opt", "sensitivity_test"]
    else:
        args.scripts = list(OrderedDict.fromkeys(args.scripts))
    args.cfg_names = set(args.cfg_names) if args.cfg_names else None
    if args.output_dir.is_absolute():
        args.output_dir = args.output_dir.resolve()
    else:
        args.output_dir = (ORIGINAL_CWD / args.output_dir).resolve()
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_items, cfg_refs = build_work_items(
        ham_names=args.ham_names,
        script_names=args.scripts,
        code_distance=args.code_distance,
        output_dir=output_dir,
        cfg_name_filter=args.cfg_names,
    )

    print(f"[INFO] target cfg references: {sum(len(c) for h in cfg_refs.values() for c in h.values())}")
    print(f"[INFO] deduplicated mapping jobs: {len(unique_items)}")
    print(f"[INFO] output dir: {output_dir}")
    print(f"[INFO] numba available: {NUMBA_AVAILABLE}")

    if args.dry_run:
        for item in unique_items:
            print(json.dumps(item["signature"], sort_keys=True))
        write_cfg_refs(cfg_refs, output_dir)
        manifest_paths, index_path = write_manifests(output_dir, args, unique_items, cfg_refs, [])
        print(f"[INFO] dry-run manifests saved: {len(manifest_paths)} ham-name manifest(s)")
        print(f"[INFO] manifest index saved to {index_path}")
        return

    jobs = [
        (
            item["ham_name"],
            item["signature"],
            item["cfg"],
            item["mapping_path"],
            args.force,
            args.verbose,
        )
        for item in unique_items
    ]

    results = []
    worker_count = max(1, min(args.num_threads, len(jobs) or 1))
    executor_cls = ProcessPoolExecutor if args.parallel_kind == "process" else ThreadPoolExecutor

    if worker_count == 1:
        iterator = (precompute_one_mapping(job) for job in jobs)
        progress = tqdm(total=len(jobs), desc="precompute mappings", unit="map") if tqdm else None
        for result in iterator:
            results.append(result)
            if progress:
                progress.update(1)
                progress.set_postfix(status=result["status"], ham=result["ham_name"])
            else:
                print(f"[INFO] {result['status']}: {result['mapping_path']}")
        if progress:
            progress.close()
    else:
        progress = tqdm(total=len(jobs), desc="precompute mappings", unit="map") if tqdm else None
        with executor_cls(max_workers=worker_count) as executor:
            future_to_job = {
                executor.submit(precompute_one_mapping, job): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                if progress:
                    progress.update(1)
                    progress.set_postfix(status=result["status"], ham=result["ham_name"])
                else:
                    print(f"[INFO] {result['status']}: {result['mapping_path']}")
        if progress:
            progress.close()

    write_cfg_refs(cfg_refs, output_dir)
    manifest_paths, index_path = write_manifests(output_dir, args, unique_items, cfg_refs, results)
    created = sum(1 for result in results if result["status"] == "created")
    skipped = sum(1 for result in results if result["status"] == "skipped")
    print(f"[INFO] created mappings: {created}, skipped mappings: {skipped}")
    print(f"[INFO] manifests saved: {len(manifest_paths)} ham-name manifest(s)")
    print(f"[INFO] manifest index saved to {index_path}")


if __name__ == "__main__":
    main()
