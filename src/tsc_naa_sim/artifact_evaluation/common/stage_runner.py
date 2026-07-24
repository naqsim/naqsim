#!/usr/bin/env python3
"""Run AE compilation and execution as separately cached stages.

The accepted simulator sources live above ``artifact_evaluation`` and are kept
unchanged.  This adapter lets their evaluation scripts build the circuits and
configuration dictionaries as before, but replaces the final experiment loop
with a content-addressed, memory-aware AE loop.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import zstandard as zstd

try:
    import fcntl
except ImportError:  # Windows: a single AE pipeline still remains race-free.
    fcntl = None


AE_DIR = Path(__file__).resolve().parents[1]
TSC_DIR = AE_DIR.parent
DEFAULT_CACHE_DIR = AE_DIR / ".cache"
CACHE_SCHEMA = 1
GIB = 1024**3

if str(TSC_DIR) not in sys.path:
    sys.path.insert(0, str(TSC_DIR))

import run_analysis_scripts as simulator  # noqa: E402


COMPILER_FIELDS = (
    "rot_trans_opt",
    "s_trans_opt",
    "plane_type",
    "inst_sched_opt",
    "rot_sched_opt",
    "skip_h",
)


@dataclass
class CompilationGroup:
    key: str
    cfg_in: Any
    qc_in: Any
    qmap_in: Any
    destinations: list[Path]
    cache_file: Path
    cache_manifest: Path
    operation_count: int


@dataclass
class ExecutionGroup:
    key: str
    cfg_in: Any
    run_opt: Any
    comp_key: str
    comp_file: Path
    destinations: list[Path]
    contexts: list[tuple[str, str]]
    cache_file: Path
    cache_manifest: Path
    operation_count: int


@dataclass
class CompilationTask:
    group: CompilationGroup
    existing_file: Path | None
    compact_existing: bool


@dataclass
class ExecutionTask:
    group: ExecutionGroup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one simulator evaluation script through the shared AE cache."
    )
    parser.add_argument("--stage", required=True, choices=("compilation", "execution"))
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        default=None,
        help="Maximum aggregate worker-memory budget; defaults to 75%% of available memory.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected simulator evaluation script.",
    )
    args = parser.parse_args()
    if args.script_args[:1] == ["--"]:
        args.script_args = args.script_args[1:]
    if args.memory_budget_gib is not None and args.memory_budget_gib <= 0:
        parser.error("--memory-budget-gib must be positive")
    return args


def canonical(value: Any) -> Any:
    """Convert simulator settings into stable JSON-compatible values."""

    if isinstance(value, Enum):
        cls = type(value)
        return {"enum": f"{cls.__module__}.{cls.__qualname__}", "name": value.name}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": canonical(vars(value)),
        }
    return {"pickle_sha256": hashlib.sha256(pickle.dumps(value)).hexdigest()}


def json_digest(value: Any) -> str:
    payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_fingerprint() -> str:
    """Invalidate caches whenever a simulator Python source changes."""

    digest = hashlib.sha256(f"ae-cache-schema:{CACHE_SCHEMA}\n".encode())
    for path in sorted(TSC_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def circuit_digest(qc_in: Any) -> str:
    """Hash a Qiskit circuit without relying on object identity or repr()."""

    try:
        import qiskit.qpy

        buffer = io.BytesIO()
        qiskit.qpy.dump(qc_in, buffer)
        payload = buffer.getvalue()
    except Exception:
        payload = pickle.dumps(qc_in, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def compilation_key(
    qc_hash: str,
    cfg_in: Any,
    qmap_in: Any,
    source_hash: str,
) -> str:
    compiler_cfg = {name: getattr(cfg_in, name) for name in COMPILER_FIELDS}
    return json_digest(
        {
            "schema": CACHE_SCHEMA,
            "source": source_hash,
            "circuit": qc_hash,
            "compiler_cfg": compiler_cfg,
            "qmap": qmap_in,
        }
    )


def execution_key(comp_key: str, cfg_in: Any, run_opt: Any, source_hash: str) -> str:
    # Using the complete experiment_config is deliberately conservative: it
    # may miss a harmless reuse, but it cannot merge configurations whose
    # execution behavior differs.
    return json_digest(
        {
            "schema": CACHE_SCHEMA,
            "source": source_hash,
            "compilation": comp_key,
            "execution_cfg": vars(cfg_in),
            "run_opt": run_opt,
        }
    )


def resolve_memory_limit() -> int:
    limits: list[int] = []
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
                break

    cgroup_max = Path("/sys/fs/cgroup/memory.max")
    cgroup_current = Path("/sys/fs/cgroup/memory.current")
    if cgroup_max.is_file():
        raw = cgroup_max.read_text().strip()
        if raw != "max":
            maximum = int(raw)
            current = int(cgroup_current.read_text()) if cgroup_current.is_file() else 0
            limits.append(max(maximum - current, 0))

    return min(limits) if limits else 8 * GIB


def memory_budget_bytes(override_gib: float | None) -> int:
    if override_gib is not None:
        return int(override_gib * GIB)
    return max(int(resolve_memory_limit() * 0.75), GIB)


def zstd_content_size(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
        size = zstd.frame_content_size(header)
    except (OSError, zstd.ZstdError):
        return None
    if size in (zstd.CONTENTSIZE_ERROR, zstd.CONTENTSIZE_UNKNOWN):
        return None
    return int(size)


def compilation_memory_estimate(operation_count: int) -> int:
    # AE disables the superlinear creg_names_cond propagation.  Keep a 2-GiB
    # floor and otherwise scale conservatively with circuit operation count.
    return max(2 * GIB, operation_count * 64 * 1024)


def execution_memory_estimate(group: ExecutionGroup) -> int:
    content_size = zstd_content_size(group.comp_file) or group.comp_file.stat().st_size * 8
    circuit_scale = max(group.operation_count, 22_000) / 22_000
    scheduler_estimate = int(2 * GIB * circuit_scale**1.5)
    return max(2 * GIB, int(content_size * 3.5), scheduler_estimate)


def worker_count(requested: int | None, estimates: Iterable[int], budget: int) -> int:
    estimates = list(estimates)
    if not estimates:
        return 1
    requested = max(int(requested or 1), 1)
    largest = max(estimates)
    admitted = max(budget // max(largest, 1), 1)
    return max(1, min(requested, len(estimates), admitted))


@contextlib.contextmanager
def cache_lock(cache_file: Path):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_file.with_suffix(cache_file.suffix + ".lock")
    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def dump_zst_atomic(obj: Any, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{cache_file.name}.", dir=cache_file.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink()
    generated = Path(f"{temp_name}.zst")
    try:
        simulator.dump_zst(obj, temp_name)
        os.replace(generated, cache_file)
    finally:
        for candidate in (temp_path, generated):
            if candidate.exists():
                candidate.unlink()


def link_or_copy(source: Path, destination: Path) -> None:
    """Materialize a cache entry without replacing a usable existing result.

    A prior cache cleanup can leave ``destination`` as a dangling symlink.
    ``Path.exists()`` is false for that case, so let the atomic ``os.replace``
    below retarget it to the current cache entry.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        relative = os.path.relpath(source, destination.parent)
        temp.symlink_to(relative)
        os.replace(temp, destination)
    finally:
        if temp.exists() or temp.is_symlink():
            temp.unlink()


def adopt_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        try:
            os.link(source, temp)
        except OSError:
            shutil.copy2(source, temp)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def strip_condition_metadata(comp_out: Any) -> int:
    """Drop metadata unused by execution and every current AE analyzer."""

    removed = 0
    inst_schedule_trace = comp_out[2]
    for layer in inst_schedule_trace:
        for inst in layer:
            names = getattr(inst, "creg_names_cond", None)
            if names:
                removed += len(names)
                inst.creg_names_cond = []
    return removed


def load_zst_path(path: Path) -> Any:
    if path.suffix != ".zst":
        raise ValueError(f"Expected a .zst file: {path}")
    return simulator.load_zst(str(path.with_suffix("")))


def compile_task(task: CompilationTask) -> dict[str, Any]:
    group = task.group
    with cache_lock(group.cache_file):
        if group.cache_file.is_file():
            return {"key": group.key, "status": "race-reuse"}

        if task.existing_file is not None and not task.compact_existing:
            adopt_file(task.existing_file, group.cache_file)
            removed = 0
            origin = "adopted-existing"
            metadata_stripped = False
        else:
            with simulator.suppress_stdout():
                if task.existing_file is not None:
                    comp_out = load_zst_path(task.existing_file)
                    origin = "compacted-existing"
                else:
                    comp_out = simulator.compilation_stage(
                        qc_in=group.qc_in,
                        cfg_in=group.cfg_in,
                        qmap_in=group.qmap_in,
                        propagate_creg_names_cond=False,
                    )
                    origin = "compiled"
            removed = strip_condition_metadata(comp_out)
            metadata_stripped = True
            dump_zst_atomic(comp_out, group.cache_file)

        manifest = {
            "schema": CACHE_SCHEMA,
            "key": group.key,
            "origin": origin,
            "condition_references_removed": removed,
            "condition_metadata_stripped": metadata_stripped,
            "operation_count": group.operation_count,
            "compressed_bytes": group.cache_file.stat().st_size,
            "uncompressed_bytes": zstd_content_size(group.cache_file),
        }
        write_json_atomic(group.cache_manifest, manifest)
        return {"key": group.key, "status": origin, "removed": removed}


def execute_task(task: ExecutionTask) -> dict[str, Any]:
    group = task.group
    with cache_lock(group.cache_file):
        if group.cache_file.is_file():
            return {"key": group.key, "status": "race-reuse"}
        with simulator.suppress_stdout():
            comp_out = load_zst_path(group.comp_file)
            exec_out = simulator.execution_stage(comp_out, group.cfg_in, group.run_opt)
        dump_zst_atomic(exec_out, group.cache_file)
        manifest = {
            "schema": CACHE_SCHEMA,
            "key": group.key,
            "compilation_key": group.comp_key,
            "run_opt": group.run_opt.name,
            "origin": "executed",
            "compressed_bytes": group.cache_file.stat().st_size,
            "uncompressed_bytes": zstd_content_size(group.cache_file),
        }
        write_json_atomic(group.cache_manifest, manifest)
        return {"key": group.key, "status": "executed"}


def run_isolated_batches(
    function: Callable[[Any], dict[str, Any]],
    tasks: list[Any],
    workers: int,
    error_callback: Callable[[Any, Exception], None] | None = None,
) -> list[dict[str, Any]]:
    """Give each memory-heavy task a fresh process, in admitted-size batches.

    Compilation callers leave ``error_callback`` unset so that a missing cache
    entry remains fatal. Execution callers use it to preserve the evaluation
    scripts' per-configuration failure isolation.
    """

    results: list[dict[str, Any]] = []
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else methods[0])
    for offset in range(0, len(tasks), workers):
        batch = tasks[offset : offset + workers]
        with ProcessPoolExecutor(max_workers=len(batch), mp_context=context) as executor:
            task_by_future = {
                executor.submit(function, task): task for task in batch
            }
            for future in as_completed(task_by_future):
                try:
                    results.append(future.result())
                except Exception as exc:
                    if error_callback is None:
                        raise
                    error_callback(task_by_future[future], exc)
    return results


def cache_paths(cache_dir: Path, kind: str, key: str) -> tuple[Path, Path]:
    root = cache_dir / kind / key[:2] / key
    filename = "comp_out.zst" if kind == "compilation" else "exec_out.zst"
    return root / filename, root / "manifest.json"


def build_compilation_groups(
    input_qc_dict: dict[str, Any],
    target_cfg_dict: dict[str, tuple[Any, list[Any]]],
    outdir_dict: dict[str, dict[str, str]],
    qmap_in: Any,
    cache_dir: Path,
    source_hash: str,
) -> tuple[dict[tuple[str, str], str], dict[str, CompilationGroup]]:
    groups: dict[str, CompilationGroup] = {}
    cfg_comp_keys: dict[tuple[str, str], str] = {}
    circuit_hashes: dict[int, str] = {}

    for qc_name, qc_in in input_qc_dict.items():
        qc_hash = circuit_hashes.setdefault(id(qc_in), circuit_digest(qc_in))
        operation_count = len(qc_in.data)
        for cfg_name, (cfg_in, _) in target_cfg_dict.items():
            destination = Path(outdir_dict[qc_name][cfg_name])
            destination.mkdir(parents=True, exist_ok=True)
            resolved_qmap = simulator.resolve_qmap_in(qmap_in, qc_name, cfg_name)
            key = compilation_key(qc_hash, cfg_in, resolved_qmap, source_hash)
            cfg_comp_keys[(qc_name, cfg_name)] = key
            cache_file, manifest = cache_paths(cache_dir, "compilation", key)
            if key not in groups:
                groups[key] = CompilationGroup(
                    key=key,
                    cfg_in=cfg_in,
                    qc_in=qc_in,
                    qmap_in=resolved_qmap,
                    destinations=[],
                    cache_file=cache_file,
                    cache_manifest=manifest,
                    operation_count=operation_count,
                )
            groups[key].destinations.append(destination)
            # Keep the original per-configuration static-input contract used by
            # the existing analysis programs.
            if not (destination / "cfg_in.zst").is_file():
                simulator.dump_zst(cfg_in, str(destination / "cfg_in"))
            if not (destination / "qc_in.zst").is_file():
                simulator.dump_zst(qc_in, str(destination / "qc_in"))

    return cfg_comp_keys, groups


def prepare_compilations(
    groups: dict[str, CompilationGroup],
    requested_workers: int | None,
    budget: int,
) -> None:
    tasks: list[CompilationTask] = []
    for group in groups.values():
        if group.cache_file.is_file():
            print(f"[cache] compilation {group.key[:12]}", flush=True)
            continue

        existing = next(
            (
                destination / "comp_out.zst"
                for destination in group.destinations
                if (destination / "comp_out.zst").is_file()
            ),
            None,
        )
        compact = False
        if existing is not None:
            content_size = zstd_content_size(existing)
            # Leave very large legacy results untouched: loading them merely to
            # compact metadata can itself exceed memory.  Newly compiled cache
            # entries are always stripped.
            compact_limit = min(4 * GIB, max(budget // 4, GIB))
            compact = content_size is not None and content_size <= compact_limit
        tasks.append(CompilationTask(group, existing, compact))

    estimates = []
    for task in tasks:
        if task.existing_file is not None:
            content_size = zstd_content_size(task.existing_file)
            estimates.append(max(2 * GIB, int((content_size or 0) * 3.5)))
        else:
            estimates.append(compilation_memory_estimate(task.group.operation_count))
    workers = worker_count(requested_workers, estimates, budget)
    if tasks:
        largest = max(estimates)
        print(
            f"[memory] compilation workers={workers}, budget={budget / GIB:.1f} GiB, "
            f"largest estimate={largest / GIB:.1f} GiB",
            flush=True,
        )
        if largest > budget:
            print(
                "[memory] warning: one compilation is estimated to exceed the budget; "
                "it will run alone.",
                flush=True,
            )
        results = run_isolated_batches(compile_task, tasks, workers)
        for result in results:
            print(f"[done] compilation {result['key'][:12]}: {result['status']}", flush=True)

    for group in groups.values():
        if not group.cache_file.is_file():
            raise FileNotFoundError(f"Missing compilation cache entry: {group.cache_file}")
    materialize_compilations(groups)


def materialize_compilations(groups: dict[str, CompilationGroup]) -> None:
    for group in groups.values():
        for destination in group.destinations:
            link_or_copy(group.cache_file, destination / "comp_out.zst")


def build_execution_groups(
    input_qc_dict: dict[str, Any],
    target_cfg_dict: dict[str, tuple[Any, list[Any]]],
    outdir_dict: dict[str, dict[str, str]],
    cfg_comp_keys: dict[tuple[str, str], str],
    cache_dir: Path,
    source_hash: str,
) -> dict[str, ExecutionGroup]:
    groups: dict[str, ExecutionGroup] = {}
    for qc_name, qc_in in input_qc_dict.items():
        operation_count = len(qc_in.data)
        for cfg_name, (cfg_in, run_opts) in target_cfg_dict.items():
            comp_key = cfg_comp_keys[(qc_name, cfg_name)]
            comp_file, _ = cache_paths(cache_dir, "compilation", comp_key)
            if not comp_file.is_file():
                raise FileNotFoundError(
                    f"Compilation cache {comp_key[:12]} is missing. Run --stage compilation first."
                )
            output_dir = Path(outdir_dict[qc_name][cfg_name])
            for run_opt in run_opts:
                key = execution_key(comp_key, cfg_in, run_opt, source_hash)
                cache_file, manifest = cache_paths(cache_dir, "execution", key)
                if key not in groups:
                    groups[key] = ExecutionGroup(
                        key=key,
                        cfg_in=cfg_in,
                        run_opt=run_opt,
                        comp_key=comp_key,
                        comp_file=comp_file,
                        destinations=[],
                        contexts=[],
                        cache_file=cache_file,
                        cache_manifest=manifest,
                        operation_count=operation_count,
                    )
                groups[key].destinations.append(output_dir)
                groups[key].contexts.append((qc_name, cfg_name))
    return groups


def write_execution_failure_logs(
    failures: list[tuple[ExecutionTask, Exception]],
) -> None:
    failures_by_log: dict[Path, list[str]] = {}
    for task, exc in failures:
        group = task.group
        formatted = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        for destination, (qc_name, cfg_name) in zip(
            group.destinations, group.contexts, strict=True
        ):
            section = (
                f"===== qc={qc_name}, cfg={cfg_name}, "
                f"stage=execution/{group.run_opt.name} =====\n"
                f"{formatted}\n"
            )
            failures_by_log.setdefault(
                destination.parent / "failed_stages.log", []
            ).append(section)

    for log_path, sections in failures_by_log.items():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("".join(sections), encoding="utf-8")


def prepare_executions(
    groups: dict[str, ExecutionGroup],
    requested_workers: int | None,
    budget: int,
) -> None:
    tasks: list[ExecutionTask] = []
    failures: list[tuple[ExecutionTask, Exception]] = []
    for group in groups.values():
        if group.cache_file.is_file():
            print(f"[cache] execution {group.key[:12]} ({group.run_opt.name})", flush=True)
            continue
        existing = next(
            (
                destination / f"exec_out_{group.run_opt.name}.zst"
                for destination in group.destinations
                if (destination / f"exec_out_{group.run_opt.name}.zst").is_file()
            ),
            None,
        )
        if existing is not None:
            with cache_lock(group.cache_file):
                adopt_file(existing, group.cache_file)
                if not group.cache_manifest.is_file():
                    write_json_atomic(
                        group.cache_manifest,
                        {
                            "schema": CACHE_SCHEMA,
                            "key": group.key,
                            "compilation_key": group.comp_key,
                            "run_opt": group.run_opt.name,
                            "origin": "adopted-existing",
                            "compressed_bytes": group.cache_file.stat().st_size,
                            "uncompressed_bytes": zstd_content_size(group.cache_file),
                        },
                    )
            print(f"[adopt] execution {group.key[:12]} ({group.run_opt.name})", flush=True)
        else:
            tasks.append(ExecutionTask(group))

    estimates = [execution_memory_estimate(task.group) for task in tasks]
    workers = worker_count(requested_workers, estimates, budget)
    if tasks:
        largest = max(estimates)
        print(
            f"[memory] execution workers={workers}, budget={budget / GIB:.1f} GiB, "
            f"largest estimate={largest / GIB:.1f} GiB",
            flush=True,
        )
        if largest > budget:
            print(
                "[memory] warning: one execution is estimated to exceed the budget; "
                "it will run alone.",
                flush=True,
            )
        # The original evaluation scripts deliberately continue after an
        # execution failure because some inputs do not support every optional
        # configuration. Keep that contract when stages are cache-separated.
        def record_failure(task: ExecutionTask, exc: Exception) -> None:
            failures.append((task, exc))
            contexts = ", ".join(
                f"{qc_name}/{cfg_name}"
                for qc_name, cfg_name in task.group.contexts
            )
            print(
                f"[WARN] execution failed ({task.group.run_opt.name}): "
                f"{contexts}: {exc}",
                flush=True,
            )

        for result in run_isolated_batches(
            execute_task,
            tasks,
            workers,
            error_callback=record_failure,
        ):
            print(f"[done] execution {result['key'][:12]}: {result['status']}", flush=True)

    if failures:
        write_execution_failure_logs(failures)
        print(
            f"[WARN] {len(failures)} unique execution(s) failed; "
            "continuing with the successful configurations. "
            "Details are saved in failed_stages.log.",
            flush=True,
        )

    for group in groups.values():
        if not group.cache_file.is_file():
            continue
        for destination in group.destinations:
            link_or_copy(
                group.cache_file,
                destination / f"exec_out_{group.run_opt.name}.zst",
            )


def run_cached_experiments(
    stage: str,
    cache_dir: Path,
    budget: int,
    input_qc_dict: dict[str, Any],
    target_cfg_dict: dict[str, tuple[Any, list[Any]]],
    outdir_dict: dict[str, dict[str, str]],
    num_threads: int | None,
    qmap_in: Any = None,
) -> None:
    source_hash = source_fingerprint()
    cfg_comp_keys, compilation_groups = build_compilation_groups(
        input_qc_dict,
        target_cfg_dict,
        outdir_dict,
        qmap_in,
        cache_dir,
        source_hash,
    )
    print(
        f"[plan] {len(target_cfg_dict)} configurations -> "
        f"{len(compilation_groups)} unique compilations",
        flush=True,
    )
    if stage == "compilation":
        prepare_compilations(
            compilation_groups,
            num_threads,
            budget,
        )
        return

    missing = [
        group.key for group in compilation_groups.values() if not group.cache_file.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} compilation cache entries are missing. "
            "Run the same command with --stage compilation first."
        )
    # A cache populated by another figure/table is sufficient for execution,
    # but the original analyzers still expect comp_out under every cfg outdir.
    materialize_compilations(compilation_groups)
    execution_groups = build_execution_groups(
        input_qc_dict,
        target_cfg_dict,
        outdir_dict,
        cfg_comp_keys,
        cache_dir,
        source_hash,
    )
    total_requested = sum(len(run_opts) for _, run_opts in target_cfg_dict.values())
    print(
        f"[plan] {total_requested} configuration/mode pairs -> "
        f"{len(execution_groups)} unique executions",
        flush=True,
    )
    prepare_executions(execution_groups, num_threads, budget)


def load_evaluation_module(script: Path) -> Any:
    script = script.expanduser().resolve()
    if script.parent != TSC_DIR or not script.name.endswith("_evaluation_fixed_d.py"):
        raise ValueError(f"Unsupported evaluation script: {script}")
    module_name = f"_ae_wrapped_{script.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    budget = memory_budget_bytes(args.memory_budget_gib)
    module = load_evaluation_module(args.script)

    def adapter(
        input_qc_dict: dict[str, Any],
        target_cfg_dict: dict[str, tuple[Any, list[Any]]],
        outdir_dict: dict[str, dict[str, str]],
        num_threads: int | None,
        qmap_in: Any = None,
    ) -> None:
        run_cached_experiments(
            args.stage,
            cache_dir,
            budget,
            input_qc_dict,
            target_cfg_dict,
            outdir_dict,
            num_threads,
            qmap_in,
        )

    old_argv = sys.argv
    try:
        sys.argv = [str(args.script), *args.script_args]
        script_args = module.parse_args()
    finally:
        sys.argv = old_argv

    module.evaluate_qpe(
        script_args.ham_name,
        code_dist=script_args.code_distance,
        use_naive_mapping=script_args.use_naive_mapping,
        num_threads=script_args.num_threads,
        use_precomputed_mapping=script_args.use_precomputed_mapping,
        precomputed_mapping_dir=script_args.precomputed_mapping_dir,
        experiment_runner=adapter,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
