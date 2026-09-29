"""Download public datasets into dated, hash-verified snapshots.

Raw files go to data/external/<source>/<YYYY-MM-DD>/ (ignored by Git); a manifest with URL,
retrieval time, size, SHA-256, and license goes to data/manifests/<source>-<YYYY-MM-DD>.json.
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from hbmperf.provenance import (
    ROOT,
    ProvenanceError,
    sha256_bytes,
    utc_now,
    verify_manifest_entry,
    write_json,
)

EXTERNAL = ROOT / "data" / "external"
MANIFESTS = ROOT / "data" / "manifests"
MAX_BYTES = 300 * 1024 * 1024
TIMEOUT_S = 120

SOURCES = {
    "epoch": {
        "files": {"ml_hardware.csv": "https://epoch.ai/data/ml_hardware.csv"},
        "license": "Creative Commons Attribution",
        "license_url": "https://epoch.ai/data/machine-learning-hardware",
    },
    "inferencex": {
        "files": {
            "llama70b_benchmarks.json": "https://inferencex.semianalysis.com/api/v1/benchmarks"
                                        "?model=Llama-3.3-70B-Instruct-FP8",
        },
        "license": "Repository Apache-2.0; API data has no explicit license (raw snapshot kept "
                   "local, only derived statistics committed)",
        "license_url": "https://github.com/SemiAnalysisAI/InferenceX",
    },
    "mlperf": {
        "files": {
            f"summary_results_v{v}.json": "https://raw.githubusercontent.com/mlcommons/"
                                          f"inference_results_v{v}/main/summary_results.json"
            for v in ("5.0", "5.1", "6.0")
        },
        "license": "Apache-2.0 (v5.1 and v6.0 repositories; v5.0 repository has no license file)",
        "license_url": "https://github.com/mlcommons/inference_results_v5.1/blob/main/LICENSE.md",
    },
}


def _download(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ProvenanceError(f"refusing non-HTTPS URL {url}")
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip",
                                               "User-Agent": "hbmperf/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = resp.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ProvenanceError(f"{url} exceeds {MAX_BYTES} bytes")
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
            if len(data) > MAX_BYTES:
                raise ProvenanceError(f"{url} exceeds {MAX_BYTES} bytes after decompression")
    return data


def fetch(source: str, date: str | None = None) -> Path:
    """Download every file of `source` and write its manifest; never overwrite a snapshot."""
    spec = SOURCES[source]
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    manifest_path = MANIFESTS / f"{source}-{date}.json"
    if manifest_path.exists():
        raise ProvenanceError(f"snapshot {manifest_path.name} already exists")
    out_dir = EXTERNAL / source / date
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, url in spec["files"].items():
        data = _download(url)
        (out_dir / name).write_bytes(data)
        files[name] = {"url": url, "bytes": len(data), "sha256": sha256_bytes(data),
                       "retrieved_utc": utc_now()}
    write_json(manifest_path, {"source": source, "date": date, "license": spec["license"],
                               "license_url": spec["license_url"], "files": files})
    return manifest_path


def latest_manifest(source: str) -> dict | None:
    paths = sorted(MANIFESTS.glob(f"{source}-*.json"))
    return json.loads(paths[-1].read_text()) if paths else None


def snapshot_path(source: str, name: str, verify: bool = True) -> Path:
    """Path of a file in the latest snapshot of `source`, verified against its manifest."""
    m = latest_manifest(source)
    if m is None:
        raise FileNotFoundError(f"no snapshot for {source}; run `hbmperf fetch {source}`")
    path = EXTERNAL / source / m["date"] / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `hbmperf fetch {source}` again")
    if verify:
        verify_manifest_entry(path, m["files"][name]["sha256"])
    return path
