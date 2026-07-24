import json
import pickle
from pathlib import Path

import zstandard as zstd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRECOMPUTED_MAPPING_DIR = SCRIPT_DIR / "output" / "precomputed_jit_mappings_fixed_d"


def enum_name(value):
    return value.name if hasattr(value, "name") else str(value)


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in text)


def mapping_signature(ham_name, cfg):
    return {
        "ham_name": ham_name,
        "plane_type": enum_name(cfg.plane_type),
        "rot_trans_opt": enum_name(cfg.rot_trans_opt),
        "inst_sched_opt": enum_name(cfg.inst_sched_opt),
        "rot_sched_opt": enum_name(cfg.rot_sched_opt),
    }


def _load_zst(path):
    with open(path, "rb") as f:
        return pickle.loads(zstd.ZstdDecompressor().decompress(f.read()))


def _resolve_mapping_dir(mapping_dir):
    if mapping_dir is None:
        return DEFAULT_PRECOMPUTED_MAPPING_DIR
    return Path(mapping_dir).expanduser().resolve()


def _resolve_mapping_path(mapping_dir, ref):
    stored_path = Path(ref["mapping_path"]).expanduser()
    candidates = []
    if stored_path.is_absolute():
        candidates.append(stored_path)
    else:
        candidates.append(mapping_dir / stored_path)
    candidates.append(mapping_dir / "unique" / stored_path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _resolve_manifest_path(mapping_dir, ham_name):
    candidates = [
        mapping_dir / safe_name(ham_name) / "manifest.json",
        mapping_dir / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_precomputed_qmaps(script_name, ham_name, target_cfg_dict, mapping_dir=None):
    mapping_dir = _resolve_mapping_dir(mapping_dir)
    manifest_path = _resolve_manifest_path(mapping_dir, ham_name)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"precomputed mapping manifest was not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    try:
        cfg_refs = manifest["cfg_refs"][script_name][ham_name]
    except KeyError as exc:
        raise KeyError(
            f"precomputed mapping manifest has no entry for script={script_name}, "
            f"ham_name={ham_name}"
        ) from exc

    qmap_by_cfg = {}
    loaded_payloads = {}
    for cfg_name, (cfg_in, _) in target_cfg_dict.items():
        if cfg_name not in cfg_refs:
            raise KeyError(
                f"precomputed mapping manifest has no entry for script={script_name}, "
                f"ham_name={ham_name}, cfg={cfg_name}"
            )

        ref = cfg_refs[cfg_name]
        expected_signature = mapping_signature(ham_name, cfg_in)
        if ref.get("mapping_signature") != expected_signature:
            raise ValueError(
                f"precomputed mapping signature mismatch for cfg={cfg_name}: "
                f"expected={expected_signature}, manifest={ref.get('mapping_signature')}"
            )

        mapping_path = _resolve_mapping_path(mapping_dir, ref)
        if mapping_path not in loaded_payloads:
            loaded_payloads[mapping_path] = _load_zst(mapping_path)
        payload = loaded_payloads[mapping_path]

        if payload.get("mapping_signature") != expected_signature:
            raise ValueError(
                f"precomputed mapping payload mismatch for cfg={cfg_name}: "
                f"expected={expected_signature}, payload={payload.get('mapping_signature')}"
            )

        qmap_init = payload.get("qmap_init")
        if not isinstance(qmap_init, dict):
            raise ValueError(f"precomputed mapping payload has no qmap_init dict: {mapping_path}")
        qmap_by_cfg[cfg_name] = qmap_init

    return {ham_name: qmap_by_cfg}
