"""Local measurements: C host microbenchmarks and llama-bench decode runs for claim C1."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from hbmperf import catalog, gguf, llm
from hbmperf.provenance import (
    DEFAULT_SEED,
    ROOT,
    ProvenanceError,
    new_run_dir,
    run_meta,
    sha256_file,
    utc_now,
    write_json,
)

BENCH_DIR = ROOT / "bench" / "host"
MANIFEST = ROOT / "data" / "manifests" / "c1_models.json"
C1_DIR = ROOT / "data" / "measurements" / "c1"
MAX_GGUF_BYTES = 4_000_000_000
MAX_OTHER_CPU_PERCENT = 200.0  # two of eight cores busy with other processes
CONTENTION_WAITS = 6  # 10-second waits before marking a configuration invalid
MAX_CV = 0.15


class Blocked(RuntimeError):
    """An external prerequisite (tool, file, or network) is unavailable."""


# ---------------------------------------------------------------- host microbenchmarks

def build_host_bench() -> Path:
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "clang")) is None:
        raise Blocked("make and a C compiler (clang) are required for host benchmarks")
    subprocess.run(["make", "-C", str(BENCH_DIR)], check=True, capture_output=True, timeout=300)
    return BENCH_DIR / "build"


def _json_lines(cmd: list[str], timeout: int = 1800) -> list[dict]:
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip().startswith("{")]


def measure_host(sizes: list[int] | None = None, threads: list[int] | None = None) -> Path:
    build = build_host_bench()
    sizes = sizes or [1 << k for k in range(14, 29)]  # 16 KiB .. 256 MiB
    threads = threads or [1, 2, 4, 8]
    config = {"kind": "host", "sizes": sizes, "threads": threads}
    out = new_run_dir("host", config)
    meta = run_meta(config, seed=DEFAULT_SEED)
    latency = _json_lines([str(build / "latency"), *map(str, sizes)])
    bandwidth = _json_lines([str(build / "bandwidth"), str(1 << 28), *map(str, threads)])
    meta["finished_utc"] = utc_now()
    write_json(out / "meta.json", meta)
    write_json(out / "host.json", {"latency": latency, "bandwidth": bandwidth})
    return out


# ---------------------------------------------------------------- llama.cpp decode (C1)

def model_dir() -> Path:
    return Path(os.environ.get("HBMPERF_MODEL_DIR", Path.home() / ".cache" / "hbmperf" / "models"))


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def llama_bench() -> str:
    exe = shutil.which("llama-bench")
    if exe is None:
        raise Blocked("llama-bench not found; install llama.cpp (for example `brew install "
                      "llama.cpp`) and retry")
    return exe


def probe() -> dict:
    exe = llama_bench()
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    lines = (out.stdout + out.stderr).splitlines()
    keep = [ln.strip() for ln in lines if ln.startswith(("version:", "built with"))]
    return {"llama_bench": Path(exe).name, "version": " | ".join(keep)}


def ensure_model(entry: dict) -> Path:
    """Download a manifest file if absent, then verify size and SHA-256."""
    if entry["bytes"] > MAX_GGUF_BYTES:
        raise ProvenanceError(f"{entry['filename']} exceeds the {MAX_GGUF_BYTES}-byte cap")
    path = model_dir() / entry["filename"]
    if not path.exists():
        if not entry["url"].startswith("https://huggingface.co/"):
            raise ProvenanceError(f"unexpected model URL {entry['url']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(entry["url"], timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
        except OSError as e:
            raise Blocked(f"download failed for {entry['filename']}: {e}") from e
        tmp.rename(path)
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ProvenanceError(f"{path.name} does not match the manifest size or SHA-256")
    return path


def depths_for(model_id: str, depths: list[int], n_gen: int) -> list[int]:
    max_ctx = {"qwen2.5-0.5b": 32768, "qwen2.5-1.5b": 32768, "smollm2-360m": 8192,
               "smollm2-1.7b": 8192, "tinyllama-1.1b": 2048}[model_id]
    return [d for d in depths if d + n_gen <= max_ctx]


def gguf_facts(entry: dict, path: Path) -> dict:
    """GGUF inventory plus guard (b): model decode bytes within 1 percent of GGUF read bytes."""
    facts = gguf.decode_read_bytes(gguf.read_header(path))
    model = catalog.get("models", entry["model_id"])
    step = llm.step(model, llm.device("apple_m2", entry["precision"]), entry["precision"],
                    phase="decode", batch=1, context=0)
    facts["model_decode_weight_bytes"] = step.bytes_weights
    facts["guard_b_rel_error"] = step.bytes_weights / facts["decode_read_bytes"] - 1
    facts["guard_b_pass"] = abs(facts["guard_b_rel_error"]) <= 0.01
    return facts


def _load_avg() -> float:
    return os.getloadavg()[0]


def other_cpu_percent() -> float:
    """CPU percent used by processes other than llama-bench and this Python process."""
    out = subprocess.run(["ps", "-Ao", "pid,pcpu,comm"], capture_output=True, text=True,
                         timeout=30).stdout.splitlines()[1:]
    me = os.getpid()
    total = 0.0
    for line in out:
        parts = line.split(None, 2)
        if len(parts) < 3 or int(parts[0]) == me or "llama-bench" in parts[2]:
            continue
        total += float(parts[1])
    return total


def wait_for_quiet() -> float:
    """Wait (bounded) until other processes use at most MAX_OTHER_CPU_PERCENT; return the level."""
    import time

    level = other_cpu_percent()
    for _ in range(CONTENTION_WAITS):
        if level <= MAX_OTHER_CPU_PERCENT:
            break
        time.sleep(10)
        level = other_cpu_percent()
    return level


def run_llama_bench(path: Path, depth: int, n_gen: int, reps: int) -> dict:
    cmd = [llama_bench(), "-m", str(path), "-p", "0", "-n", str(n_gen), "-d", str(depth),
           "-r", str(reps), "-o", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"status": "invalid", "reason": "timeout"}
    if proc.returncode != 0:
        return {"status": "invalid", "reason": f"exit {proc.returncode}: {proc.stderr[-300:]}"}
    try:
        rows = json.loads(proc.stdout)
        row = next(r for r in rows if r.get("n_gen") == n_gen)
    except (json.JSONDecodeError, StopIteration):
        return {"status": "invalid", "reason": "unparseable llama-bench output"}
    samples = row.get("samples_ns")
    if not samples or len(samples) != reps:
        return {"status": "invalid", "reason": "samples_ns missing or incomplete"}
    step_s = np.asarray(samples, dtype=float) / n_gen / 1e9
    cv = float(step_s.std(ddof=1) / step_s.mean()) if len(step_s) > 1 else 0.0
    return {
        "status": "ok", "step_time_s": float(np.median(step_s)), "step_times_s": step_s.tolist(),
        "cv": cv, "model_size": row.get("model_size"), "build_commit": row.get("build_commit"),
        "build_number": row.get("build_number"), "backends": row.get("backends"),
        "n_gpu_layers": row.get("n_gpu_layers"), "flash_attn": row.get("flash_attn"),
    }


def measure_config(entry: dict, path: Path, depth: int, n_gen: int, reps: int) -> dict:
    other = wait_for_quiet()
    rec = {"filename": entry["filename"], "model_id": entry["model_id"],
           "precision": entry["precision"], "split": entry["split"], "depth": depth,
           "n_gen": n_gen, "load_avg_1m": _load_avg(), "other_cpu_percent": other,
           "measured_utc": utc_now()}
    if other > MAX_OTHER_CPU_PERCENT:
        return {**rec, "status": "invalid",
                "reason": f"other processes use {other:.0f}% CPU > {MAX_OTHER_CPU_PERCENT:.0f}%"}
    res = run_llama_bench(path, depth, n_gen, reps)
    if res["status"] == "ok" and res["cv"] > MAX_CV:
        res = run_llama_bench(path, depth, n_gen, reps)  # one automatic retry
        if res["status"] == "ok" and res["cv"] > MAX_CV:
            res = {**res, "status": "invalid", "reason": f"cv {res['cv']:.3f} > {MAX_CV}"}
    return {**rec, **res}


def download_split(split: str) -> list[dict]:
    """Download and hash-verify a split's files without benchmarking them."""
    out = []
    for e in (e for e in manifest()["files"] if e["split"] == split):
        path = ensure_model(e)
        facts = gguf_facts(e, path)
        out.append({"filename": e["filename"], "verified": True, **facts})
    return out


def measure_split(split: str, *, out: Path | None = None, seed: int = DEFAULT_SEED) -> Path:
    """Measure every (file, depth) of a split with session anchors; resumable via `out`."""
    m = manifest()
    probe_info = probe()
    entries = [e for e in m["files"] if e["split"] == split]
    anchor = next(e for e in m["files"] if e["filename"] == m["anchor"])
    if not entries:
        raise ValueError(f"no files for split {split!r}")
    C1_DIR.mkdir(parents=True, exist_ok=True)
    out = out or C1_DIR / f"{split}-{utc_now().replace(':', '').replace('-', '')}.json"
    session = json.loads(out.read_text()) if out.exists() else {
        "split": split, "seed": seed, "probe": probe_info, "started_utc": utc_now(),
        "records": [], "anchors": [], "files": {},
    }
    paths = {e["filename"]: ensure_model(e) for e in [anchor, *entries]}
    for e in [anchor, *entries]:
        if e["filename"] not in session["files"]:
            facts = gguf_facts(e, paths[e["filename"]])
            session["files"][e["filename"]] = {"sha256": e["sha256"], **facts}

    def save():
        write_json(out, session)

    if not session["anchors"]:
        session["anchors"].append(measure_config(anchor, paths[anchor["filename"]], 0,
                                                 m["n_gen"], m["repetitions"]))
        save()
    keys = [(e["filename"], d) for e in entries
            for d in depths_for(e["model_id"], m["depths"], m["n_gen"])]
    rng = np.random.Generator(np.random.PCG64(seed))
    order = [keys[i] for i in rng.permutation(len(keys))]
    done = {(r["filename"], r["depth"]) for r in session["records"]}
    by_name = {e["filename"]: e for e in entries}
    for name, depth in order:
        if (name, depth) in done:
            continue
        rec = measure_config(by_name[name], paths[name], depth, m["n_gen"], m["repetitions"])
        facts = session["files"][name]
        if rec.get("model_size") is not None and rec["model_size"] != facts["file_tensor_bytes"]:
            rec = {**rec, "status": "invalid", "reason": "guard (a): GGUF bytes != model_size"}
        session["records"].append(rec)
        save()
    if len(session["anchors"]) < 2:
        session["anchors"].append(measure_config(anchor, paths[anchor["filename"]], 0,
                                                 m["n_gen"], m["repetitions"]))
    session["finished_utc"] = utc_now()
    session["meta"] = run_meta({"kind": "c1", "split": split}, seed=seed,
                               tool_versions=probe_info)
    save()
    return out
