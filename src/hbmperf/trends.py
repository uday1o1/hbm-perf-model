"""Hardware trend fits and architectural intercepts over Epoch AI ML Hardware data.

Fits log2(metric) = a + b * (year - 2016) by ordinary least squares; the doubling time is 1/b.
Intervals come from a nonparametric bootstrap over devices (2000 resamples, fixed seed), and
intercept years are recomputed inside every resample. Intercepts are extrapolations, not
forecasts.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np

from hbmperf import catalog, fetch, llm
from hbmperf.provenance import DEFAULT_SEED, ROOT, run_meta, utc_now, write_json

DERIVED = ROOT / "data" / "derived" / "epoch_accelerators.csv"
RESULTS = ROOT / "results" / "trends"
BASE_YEAR = 2016
MIN_DEVICES = 8
N_RESAMPLES = 2000
YEAR_RANGE = (2010, 2040)
ATTRIBUTION = ("# Derived from Epoch AI, 'Machine Learning Hardware' "
               "(https://epoch.ai/data/machine-learning-hardware), Creative Commons Attribution\n")
COLUMNS = ["name", "manufacturer", "type", "release_date", "year", "bf16_flops",
           "mem_bw_bytes_per_s", "mem_bytes"]


def _year(iso: str) -> float:
    d = date.fromisoformat(iso[:10])
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def _num(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def build_derived(out: Path = DERIVED) -> Path:
    """Filter the latest Epoch snapshot to dated devices with bandwidth or tensor data."""
    src = fetch.snapshot_path("epoch", "ml_hardware.csv")
    rows = []
    with open(src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["Release date"] or r["Type"] in ("Hybrid CPU", ""):
                continue
            bw = _num(r["Memory bandwidth (byte/s)"])
            fl = _num(r["Tensor-FP16/BF16 performance (FLOP/s)"])
            mem = _num(r["Memory (bytes)"])
            if bw is None and fl is None:
                continue
            rows.append({"name": r["Hardware name"], "manufacturer": r["Manufacturer"],
                         "type": r["Type"], "release_date": r["Release date"][:10],
                         "year": round(_year(r["Release date"]), 4), "bf16_flops": fl or "",
                         "mem_bw_bytes_per_s": bw or "", "mem_bytes": mem or ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write(ATTRIBUTION)
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["release_date"], r["name"])):
            w.writerow(r)
    return out


def load_devices(path: Path = DERIVED) -> list[dict]:
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    out = []
    for r in csv.DictReader(lines):
        out.append({"name": r["name"], "year": float(r["year"]),
                    "bf16_flops": _num(r["bf16_flops"]), "mem_bw": _num(r["mem_bw_bytes_per_s"]),
                    "mem_bytes": _num(r["mem_bytes"])})
    return out


def _fit(years: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(years - BASE_YEAR, np.log2(values), 1)
    return float(a), float(b)


def _crossing(a: float, b: float, target: float) -> float | None:
    """Year at which a + b * (year - BASE_YEAR) = log2(target); None if out of range."""
    if b <= 0:
        return None
    y = BASE_YEAR + (np.log2(target) - a) / b
    return float(y) if YEAR_RANGE[0] <= y <= YEAR_RANGE[1] else None


def fit_metric(devices: list[dict], key: str, target: float | None = None,
               seed: int = DEFAULT_SEED) -> dict:
    pts = [(d["year"], d[key]) for d in devices if d[key]]
    if len(pts) < MIN_DEVICES:
        return {"metric": key, "n_devices": len(pts), "verdict": "INSUFFICIENT_EVIDENCE"}
    yrs, vals = (np.array(x, dtype=float) for x in zip(*pts, strict=True))
    a, b = _fit(yrs, vals)
    rng = np.random.Generator(np.random.PCG64(seed))
    boot_b, boot_x = [], []
    for _ in range(N_RESAMPLES):
        i = rng.integers(0, len(yrs), len(yrs))
        if np.ptp(yrs[i]) == 0:
            continue
        ab, bb = _fit(yrs[i], vals[i])
        boot_b.append(bb)
        if target is not None:
            boot_x.append(_crossing(ab, bb, target))
    lo, hi = np.percentile(boot_b, [2.5, 97.5])
    out = {"metric": key, "n_devices": len(pts), "log2_intercept": a, "log2_per_year": b,
           "growth_per_year": 2**b, "doubling_years": 1 / b if b > 0 else None,
           "doubling_years_ci": [1 / hi, 1 / lo] if lo > 0 else None,
           "year_range": [float(yrs.min()), float(yrs.max())]}
    if target is not None:
        cross = _crossing(a, b, target)
        valid = [x for x in boot_x if x is not None]
        out["target"] = target
        out["intercept_year"] = cross
        out["intercept_year_ci"] = (np.percentile(valid, [2.5, 97.5]).tolist()
                                    if len(valid) >= 0.95 * len(boot_x) else None)
        out["intercept_note"] = ("no intercept within range" if cross is None else
                                 "extrapolation of a log-linear fit, not a forecast")
    return out


def targets(model_id: str = "llama-3.3-70b", capacity_batch: int = 32,
            capacity_context: int = 8192, tpot_target_s: float = 0.005,
            tpot_context: int = 1024) -> dict:
    """Illustrative requirement lines for intercepts, derived from the LLM model."""
    m = catalog.get("models", model_id)
    w, kv = llm.footprint_bytes(m, "bf16", batch=capacity_batch, max_context=capacity_context)
    p = llm.params(m)
    read = (m["n_layers"] * (p.attn_per_layer + p.norm_per_layer + p.mlp_dense_per_layer)
            + p.lm_head) * 1.0 + tpot_context * llm.kv_bytes_per_token(m, 8)  # FP8: 1 B/weight
    return {
        "capacity_bytes": w + kv,
        "capacity_desc": f"{model_id} BF16 weights plus {capacity_batch} sequences x "
                         f"{capacity_context} tokens of BF16 KV on one device",
        "bandwidth_bytes_per_s": read / tpot_target_s,
        "bandwidth_desc": f"{model_id} FP8 batch-1 decode at {tpot_context} tokens of context "
                          f"with TPOT {tpot_target_s * 1e3:.0f} ms on one device "
                          "(memory-bound, 100% bandwidth efficiency)",
    }


def analyze(devices: list[dict] | None = None, seed: int = DEFAULT_SEED) -> dict:
    devices = devices if devices is not None else load_devices()
    t = targets()
    ridge = [{"name": d["name"], "year": d["year"], "ridge": d["bf16_flops"] / d["mem_bw"]}
             for d in devices if d["bf16_flops"] and d["mem_bw"]]
    fits = {
        "mem_bw": fit_metric(devices, "mem_bw", t["bandwidth_bytes_per_s"], seed),
        "bf16_flops": fit_metric(devices, "bf16_flops", None, seed),
        "mem_bytes": fit_metric(devices, "mem_bytes", t["capacity_bytes"], seed),
        "ridge": fit_metric([{"year": r["year"], "ridge": r["ridge"]} for r in ridge], "ridge",
                            None, seed),
    }
    return {"fits": fits, "targets": t, "ridge_points": ridge,
            "notes": "Ridge point = dense BF16 FLOP/s divided by memory bandwidth; for BF16 "
                     "weights it approximates the decode batch at which a short-context dense "
                     "model turns compute-bound."}


def write_results() -> Path:
    if not DERIVED.exists():
        build_derived()
    res = analyze()
    res["meta"] = run_meta({"kind": "trends", "resamples": N_RESAMPLES,
                            "min_devices": MIN_DEVICES, "base_year": BASE_YEAR})
    res["meta"]["finished_utc"] = utc_now()
    path = RESULTS / "trends.json"
    write_json(path, res)
    return path
