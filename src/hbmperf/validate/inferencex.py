"""Claim C2: calibrated TPOT predictions for held-out accelerators versus InferenceX.

Calibration uses FP8 rows of H100, H200, and MI300X; evaluation uses B200, MI325X, and
MI355X. FP4 rows (NVFP4 on NVIDIA, MXFP4 on AMD) are reported separately as a precision
extrapolation with no threshold claim (BUILD_PLAN sections 1.6, 8.5, 11).
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict

import numpy as np

from hbmperf import catalog, fetch, llm, stats
from hbmperf.provenance import ROOT, run_meta, utc_now, write_json
from hbmperf.validate import VALIDATION_DIR, claim_input_sha256

MODEL = "llama-3.3-70b"
SNAPSHOT = "llama70b_benchmarks.json"
HARDWARE = {"h100": "nvidia_h100_sxm", "h200": "nvidia_h200_sxm", "mi300x": "amd_mi300x",
            "b200": "nvidia_b200", "mi325x": "amd_mi325x", "mi355x": "amd_mi355x"}
CALIBRATION_HW = ("h100", "h200", "mi300x")
EVALUATION_HW = ("b200", "mi325x", "mi355x")
KV_BITS = 8.0  # labeled assumption: FP8 KV cache
RULES = dict(threshold=0.35, upper=0.50, min_clusters=6, min_examples=60)
MODULES = ["llm.py", "stats.py", "validate/__init__.py", "validate/inferencex.py"]
CAL_PATH = VALIDATION_DIR / "c2_calibration.json"
REPORT_PATH = VALIDATION_DIR / "c2.json"
DERIVED = ROOT / "data" / "derived" / "inferencex_c2_predictions.csv"

# calibration grid (bounded ranges; deterministic)
GRID = {
    "eta_m": np.round(np.arange(0.30, 1.001, 0.025), 3),
    "eta_c": np.round(np.arange(0.10, 1.001, 0.05), 3),
    "alpha": np.array([1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]),
    "t_fixed": np.array([0, 0.25e-3, 0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3]),
}


def precision_for(row: dict) -> str | None:
    if row["precision"] == "fp8":
        return "fp8"
    if row["precision"] == "fp4":
        return "nvfp4" if HARDWARE[row["hardware"]].startswith("nvidia") else "mxfp4"
    return None


def load_rows() -> tuple[list[dict], dict[str, int]]:
    """Single-node, non-disaggregated Llama-3.3-70B rows with components for fast prediction."""
    data = json.loads(fetch.snapshot_path("inferencex", SNAPSHOT).read_text())
    model = catalog.get("models", MODEL)
    rows, excluded = [], defaultdict(int)
    for r in data:
        if r.get("disagg") or r.get("is_multinode") or r.get("hardware") not in HARDWARE:
            excluded["multinode, disaggregated, or unmapped hardware"] += 1
            continue
        prec = precision_for(r)
        tpot = (r.get("metrics") or {}).get("median_tpot")
        if prec is None or not tpot:
            excluded["unsupported precision or missing TPOT"] += 1
            continue
        tp = int(r["decode_tp"])
        replicas = max(1, int(r.get("num_decode_gpu") or tp) // tp)
        batch = max(1, round(int(r["conc"]) / replicas))
        dev = llm.device(HARDWARE[r["hardware"]], prec)
        if dev.peak_flops is None:
            excluded[f"no published dense peak for {prec} on {r['hardware']}"] += 1
            continue
        ctx = int(r["isl"]) + int(r["osl"]) / 2
        s = llm.step(model, dev, prec, phase="decode", batch=batch, context=ctx, tp=tp,
                     kv_bits=KV_BITS)
        # decompose so calibration can scan knobs without recomputing the model
        hops = 2 * model["n_layers"] * 2 * (tp - 1) if tp > 1 else 0
        comm_bw = s.t_comm - hops * llm.DEFAULT_KNOBS.alpha
        rows.append({
            "hardware": r["hardware"], "framework": r["framework"], "precision": prec,
            "tp": tp, "conc": int(r["conc"]), "batch": batch, "isl": int(r["isl"]),
            "osl": int(r["osl"]), "measured_tpot_s": float(tpot), "a": s.t_mem,
            "c": s.t_compute, "comm_bw": comm_bw, "hops": hops,
            "peak_bw": dev.mem_bw,
            "cluster": f"{r['hardware']}/{r['framework']}/{prec}/tp{tp}",
        })
    return rows, dict(excluded)


def _predict(rows: list[dict], k: dict) -> np.ndarray:
    a = np.array([r["a"] for r in rows])
    c = np.array([r["c"] for r in rows])
    comm = np.array([r["comm_bw"] + r["hops"] * k["alpha"] for r in rows])
    return np.maximum(a / k["eta_m"], c / k["eta_c"]) + comm + k["t_fixed"]


def calibrate(write: bool = True) -> dict:
    rows, excluded = load_rows()
    cal = [r for r in rows if r["hardware"] in CALIBRATION_HW and r["precision"] == "fp8"]
    if not cal:
        raise ValueError("no calibration rows in the InferenceX snapshot")
    meas = np.array([r["measured_tpot_s"] for r in cal])
    a = np.array([r["a"] for r in cal])
    c = np.array([r["c"] for r in cal])
    comm_bw = np.array([r["comm_bw"] for r in cal])
    hops = np.array([r["hops"] for r in cal])
    best, best_k = np.inf, None
    for em, ec, al, t0 in itertools.product(*GRID.values()):
        pred = np.maximum(a / em, c / ec) + comm_bw + hops * al + t0
        loss = float(np.mean(np.log(pred / meas) ** 2))  # symmetric relative error
        if loss < best:
            best, best_k = loss, {"eta_m": float(em), "eta_c": float(ec), "alpha": float(al),
                                  "t_fixed": float(t0)}
    pred = _predict(cal, best_k)
    result = {"claim": "C2", "n_examples": len(cal), "knobs": best_k, "loss": best,
              "calibration_mdape": float(np.median(stats.ape(pred, meas))),
              "grid": {k: v.tolist() for k, v in GRID.items()}, "excluded": excluded,
              "meta": run_meta({"claim": "C2", "stage": "calibration"})}
    result["meta"]["finished_utc"] = utc_now()
    if write:
        write_json(CAL_PATH, result)
    return result


def _baseline(rows: list[dict], cal_rows: list[dict]) -> list[float | None]:
    """Bandwidth-scaled calibration TPOT for the matching configuration, if any."""
    index = defaultdict(list)
    for r in cal_rows:
        index[(r["framework"], r["tp"], r["conc"], r["isl"], r["osl"])].append(r)
    out = []
    for r in rows:
        m = index.get((r["framework"], r["tp"], r["conc"], r["isl"], r["osl"]))
        out.append(float(np.mean([x["measured_tpot_s"] * x["peak_bw"] / r["peak_bw"]
                                  for x in m])) if m else None)
    return out


def _slices(rows, pred) -> dict:
    groups = defaultdict(list)
    for r, p in zip(rows, pred, strict=True):
        e = abs(p - r["measured_tpot_s"]) / r["measured_tpot_s"]
        groups[f"isl{r['isl']}_osl{r['osl']}"].append(e)
        groups["conc<=16" if r["conc"] <= 16 else "conc>16"].append(e)
        groups[f"framework={r['framework']}"].append(e)
        groups[f"hardware={r['hardware']}"].append(e)
    return {k: {"n": len(v), "mdape": float(np.median(v))} for k, v in sorted(groups.items())}


def _compare(rows, cal_rows, knobs, rules, label):
    pred = _predict(rows, knobs)
    base = _baseline(rows, cal_rows)
    keep = [i for i, b in enumerate(base) if b is not None]
    meas = np.array([r["measured_tpot_s"] for r in rows])
    comp = stats.compare([rows[i]["cluster"] for i in keep], pred[keep],
                         [base[i] for i in keep], meas[keep], **rules)
    spec = _predict(rows, {"eta_m": 1.0, "eta_c": 1.0, "alpha": llm.DEFAULT_KNOBS.alpha,
                           "t_fixed": 0.0})
    return {
        "label": label, "verdict": comp.verdict, "reasons": comp.reasons,
        "comparison": comp.to_dict(), "fallback_claim": stats.fallback_claim(comp),
        "n_rows": len(rows), "n_without_baseline_match": len(rows) - len(keep),
        "mdape_all_rows": float(np.median(stats.ape(pred, meas))) if len(rows) else None,
        "spec_peak_mdape": float(np.median(stats.ape(spec, meas))) if len(rows) else None,
        "slices": _slices(rows, pred),
    }, pred


def evaluate(write: bool = True) -> dict:
    if not CAL_PATH.exists():
        raise FileNotFoundError("no calibration; run `hbmperf validate inferencex --calibrate`")
    knobs = json.loads(CAL_PATH.read_text())["knobs"]
    rows, excluded = load_rows()
    cal_rows = [r for r in rows if r["hardware"] in CALIBRATION_HW and r["precision"] == "fp8"]
    ev8 = [r for r in rows if r["hardware"] in EVALUATION_HW and r["precision"] == "fp8"]
    ev4 = [r for r in rows if r["hardware"] in EVALUATION_HW and r["precision"] != "fp8"]
    main, pred8 = _compare(ev8, cal_rows, knobs, RULES, "C2 (FP8, held-out hardware)")
    fp4, pred4 = _compare(ev4, cal_rows, knobs, RULES, "C2-FP4 (precision extrapolation)")
    fp4["verdict_note"] = "reported only; no threshold claim (no FP4 calibration hardware)"
    rep = {"claim": "C2", "verdict": main["verdict"], "reasons": main["reasons"],
           "fallback_claim": main["fallback_claim"], "comparison": main["comparison"],
           "fp8": main, "fp4": fp4, "knobs": knobs, "kv_bits_assumption": KV_BITS,
           "excluded": excluded, "rules": RULES,
           "source": "SemiAnalysis InferenceX public API snapshot "
                     f"({fetch.latest_manifest('inferencex')['date']}); repository Apache-2.0"}
    snap = fetch.snapshot_path("inferencex", SNAPSHOT)
    used = [catalog.get("accelerators", HARDWARE[h]) for h in sorted(HARDWARE)]
    rep["claim_input_sha256"] = claim_input_sha256(MODULES, [*used, catalog.get("models", MODEL)],
                                                   [snap, CAL_PATH])
    rep["meta"] = run_meta({"claim": "C2", "stage": "evaluation"})
    rep["meta"]["finished_utc"] = utc_now()
    if write:
        write_json(REPORT_PATH, rep)
        _write_derived([*ev8, *ev4], np.concatenate([pred8, pred4]))
    return rep


def _write_derived(rows, pred) -> None:
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    cols = ["hardware", "framework", "precision", "tp", "conc", "isl", "osl",
            "measured_tpot_s", "predicted_tpot_s"]
    with open(DERIVED, "w", newline="") as f:
        f.write("# Derived from the SemiAnalysis InferenceX public API "
                "(https://github.com/SemiAnalysisAI/InferenceX); measured values are "
                "InferenceX median TPOT\n")
        w = csv.DictWriter(f, cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r, p in sorted(zip(rows, pred, strict=True),
                           key=lambda x: (x[0]["cluster"], x[0]["conc"], x[0]["isl"])):
            w.writerow({**r, "predicted_tpot_s": float(p)})
