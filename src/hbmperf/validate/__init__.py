"""Validation of model predictions against measurements (claims C1, C2, and invariant I1)."""

from __future__ import annotations

import json
from pathlib import Path

from hbmperf.provenance import ROOT, dumps, sha256_bytes, sha256_file

VALIDATION_DIR = ROOT / "results" / "validation"
SRC = ROOT / "src" / "hbmperf"


def claim_input_sha256(modules: list[str], catalog_rows: list[dict], files: list[Path]) -> str:
    """Hash of everything a claim's result depends on: module sources, the exact catalog rows
    used, and the committed measurement or snapshot files read."""
    parts = {
        "modules": {m: sha256_file(SRC / m) for m in sorted(modules)},
        # documentation fields do not affect results, so editing them does not stale a claim
        "catalog_rows": sorted(dumps({k: v for k, v in r.items()
                                      if k not in ("notes", "source_url")})
                               for r in catalog_rows),
        "files": {str(Path(f).resolve().relative_to(ROOT)) if Path(f).resolve().is_relative_to(ROOT)
                  else Path(f).name: sha256_file(Path(f)) for f in sorted(files)},
    }
    return sha256_bytes(dumps(parts).encode())


def staleness(report_path: Path, current: str) -> str:
    """'CURRENT', 'STALE', or 'MISSING' for a committed report against a recomputed hash."""
    if not report_path.exists():
        return "MISSING"
    recorded = json.loads(report_path.read_text()).get("claim_input_sha256")
    return "CURRENT" if recorded == current else "STALE"
