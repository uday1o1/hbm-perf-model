"""Hashing, manifests, run directories, and run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 20260928
SCHEMA_VERSION = 1


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dumps(obj) -> str:
    """Deterministic JSON: sorted keys, two-space indent, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=2, default=_default) + "\n"


def _default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj))


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_state() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def cpu_brand() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                timeout=5,
            ).stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def mem_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def llm_model_sha256() -> str:
    return sha256_file(Path(__file__).with_name("llm.py"))


def run_meta(config: dict, seed: int | None = None, tool_versions: dict | None = None) -> dict:
    from hbmperf import __version__

    sha, dirty = git_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "hbmperf_version": __version__,
        "git_sha": sha,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_brand": cpu_brand(),
        "mem_bytes": mem_bytes(),
        "tool_versions": tool_versions or {},
        "seed": seed,
        "config": config,
        "config_sha256": sha256_bytes(dumps(config).encode()),
        "llm_model_sha256": llm_model_sha256(),
        "started_utc": utc_now(),
    }


def new_run_dir(kind: str, config: dict, base: Path | None = None) -> Path:
    """Create results/<kind>/<run_id>/ without clobbering an existing directory."""
    base = (base or ROOT / "results").resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{sha256_bytes(dumps(config).encode())[:8]}"
    path = (base / kind / run_id).resolve()
    if base not in path.parents:
        raise ProvenanceError(f"refusing to write outside {base}: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_inside(path: Path, allowed: Path) -> Path:
    path = path.resolve()
    if allowed.resolve() not in (path, *path.parents):
        raise ProvenanceError(f"path {path} is outside {allowed}")
    return path


def verify_manifest_entry(path: Path, expected_sha256: str) -> None:
    got = sha256_file(path)
    if got != expected_sha256:
        raise ProvenanceError(f"hash mismatch for {path}: expected {expected_sha256}, got {got}")
