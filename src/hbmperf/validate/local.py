"""Claim C1: calibrated decode step-time predictions versus llama-bench on the local Apple M2.

Workflow (BUILD_PLAN section 13.1 and 13.4):
1. `hbmperf measure llamacpp --split calibration` writes a calibration session.
2. `hbmperf validate local --calibrate` fits eta_m and t_fixed (and the file-size baseline) on
   calibration examples only and writes results/validation/c1_calibration.json.
3. Tag `eval-c1-v1`, then `hbmperf measure llamacpp --split evaluation`.
4. `hbmperf validate local` evaluates once and writes results/validation/c1.json.
Everything needed is in committed files, so step 4 reproduces without model files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hbmperf import catalog, llm, stats
from hbmperf.measure import C1_DIR
from hbmperf.provenance import run_meta, utc_now, write_json
from hbmperf.validate import VALIDATION_DIR, claim_input_sha256, staleness

DEVICE = "apple_m2"
MODULES = ["llm.py", "gguf.py", "stats.py", "validate/local.py", "validate/__init__.py"]
RULES = dict(threshold=0.20, upper=0.30, min_clusters=4, min_examples=24)
ANCHOR_TOLERANCE = 0.10
CAL_PATH = VALIDATION_DIR / "c1_calibration.json"
REPORT_PATH = VALIDATION_DIR / "c1.json"


def sessions(split: str, directory: Path = C1_DIR) -> list[Path]:
    return sorted(directory.glob(f"{split}-*.json"))


def examples(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        s = json.loads(path.read_text())
        for r in s["records"]:
            if r["status"] != "ok":
                continue
            f = s["files"][r["filename"]]
            rows.append({**r, "file_tensor_bytes": f["file_tensor_bytes"],
                         "guard_b_pass": f["guard_b_pass"], "session": path.name})
    return rows


def anchor_step(paths: list[Path]) -> float | None:
    vals = [a["step_time_s"] for p in paths for a in json.loads(p.read_text())["anchors"]
            if a["status"] == "ok"]
    return float(np.median(vals)) if vals else None


def unit_memory_time(ex: dict, weights_only: bool = False) -> float:
    """Predicted batch-1 decode memory time at eta_m = 1 (seconds)."""
    model = catalog.get("models", ex["model_id"])
    dev = llm.device(DEVICE, ex["precision"])
    s = llm.step(model, dev, ex["precision"], phase="decode", batch=1,
                 context=ex["depth"] + ex["n_gen"] / 2,
                 knobs=llm.Knobs(weights_only=weights_only))
    return s.t_mem


def calibrate(directory: Path = C1_DIR) -> dict:
    cal_paths = sessions("calibration", directory)
    ex = examples(cal_paths)
    if not ex:
        raise ValueError("no valid calibration examples; run `hbmperf measure llamacpp`")
    if not all(e["guard_b_pass"] for e in ex):
        raise ValueError("GGUF byte-accounting guard (b) failed on a calibration file")
    y = [e["step_time_s"] for e in ex]
    a, b = stats.fit_linear([unit_memory_time(e) for e in ex], y)
    aw, bw = stats.fit_linear([unit_memory_time(e, True) for e in ex], y)
    fa, fb = stats.fit_linear([e["file_tensor_bytes"] for e in ex], y)
    result = {
        "claim": "C1", "device": DEVICE, "n_examples": len(ex),
        "sessions": [p.name for p in cal_paths],
        "candidate": {"eta_m": 1 / a, "t_fixed_s": b},
        "ablation_weights_only": {"eta_m": 1 / aw, "t_fixed_s": bw},
        "baseline_file_bytes": {"s_per_byte": fa, "t0_s": fb},
        "anchor_step_time_s": anchor_step(cal_paths),
        "meta": run_meta({"claim": "C1", "stage": "calibration"}),
    }
    result["meta"]["finished_utc"] = utc_now()
    write_json(CAL_PATH, result)
    return result


def _input_hash(eval_paths: list[Path], cal_paths: list[Path], model_ids: set[str]) -> str:
    rows = [catalog.get("models", m) for m in sorted(model_ids)]
    rows.append(catalog.get("accelerators", DEVICE))
    return claim_input_sha256(MODULES, rows, [*cal_paths, *eval_paths, CAL_PATH])


def evaluate(directory: Path = C1_DIR, write: bool = True) -> dict:
    if not CAL_PATH.exists():
        raise FileNotFoundError("no calibration; run `hbmperf validate local --calibrate`")
    cal = json.loads(CAL_PATH.read_text())
    cal_paths = sessions("calibration", directory)
    eval_paths = sessions("evaluation", directory)
    ex = examples(eval_paths)
    k = llm.Knobs(eta_m=cal["candidate"]["eta_m"], t_fixed=cal["candidate"]["t_fixed_s"])
    kw = cal["ablation_weights_only"]
    meas = np.array([e["step_time_s"] for e in ex])
    unit = np.array([unit_memory_time(e) for e in ex])
    unit_w = np.array([unit_memory_time(e, True) for e in ex])
    pred = unit / k.eta_m + k.t_fixed
    base = np.array([cal["baseline_file_bytes"]["s_per_byte"] * e["file_tensor_bytes"]
                     + cal["baseline_file_bytes"]["t0_s"] for e in ex])
    clusters = [e["filename"] for e in ex]

    anchor_cal = cal["anchor_step_time_s"]
    anchor_eval = anchor_step(eval_paths)
    drift = None if not (anchor_cal and anchor_eval) else anchor_eval / anchor_cal - 1
    anchor_ok = drift is not None and abs(drift) <= ANCHOR_TOLERANCE
    guards_ok = all(e["guard_b_pass"] for e in ex)

    report: dict = {"claim": "C1", "device": DEVICE, "rules": RULES,
                    "calibration": cal["candidate"], "anchor_drift": drift,
                    "anchor_within_tolerance": anchor_ok, "guards_pass": guards_ok,
                    "sessions": [p.name for p in eval_paths]}
    if len(ex) == 0:
        report.update(verdict="INSUFFICIENT_EVIDENCE", reasons=["no evaluation examples"])
    else:
        comp = stats.compare(clusters, pred, base, meas, **RULES)
        verdict, reasons = comp.verdict, list(comp.reasons)
        if not anchor_ok:
            verdict = "INSUFFICIENT_EVIDENCE"
            reasons.append(f"anchor drift {drift} outside {ANCHOR_TOLERANCE}")
        if not guards_ok:
            verdict = "INSUFFICIENT_EVIDENCE"
            reasons.append("GGUF byte-accounting guard failed")
        scale = (anchor_cal / anchor_eval) if (anchor_cal and anchor_eval) else 1.0
        report.update(
            verdict=verdict, reasons=reasons, comparison=comp.to_dict(),
            fallback_claim=stats.fallback_claim(comp) if verdict != "INSUFFICIENT_EVIDENCE"
            else "none: insufficient evidence",
            spec_peak_mdape=float(np.median(stats.ape(unit, meas))),
            ablation_weights_only_mdape=float(np.median(stats.ape(
                unit_w / kw["eta_m"] + kw["t_fixed_s"], meas))),
            drift_normalized_mdape=float(np.median(stats.ape(pred, meas * scale))),
            examples=[{"filename": e["filename"], "model_id": e["model_id"],
                       "precision": e["precision"], "depth": e["depth"],
                       "measured_s": float(m), "predicted_s": float(p),
                       "baseline_s": float(bb), "spec_peak_s": float(u)}
                      for e, m, p, bb, u in zip(ex, meas, pred, base, unit, strict=True)],
        )
    report["claim_input_sha256"] = _input_hash(eval_paths, cal_paths,
                                               {e["model_id"] for e in ex})
    report["meta"] = run_meta({"claim": "C1", "stage": "evaluation"})
    report["meta"]["finished_utc"] = utc_now()
    if write:
        write_json(REPORT_PATH, report)
    return report


def status(directory: Path = C1_DIR) -> str:
    """STALE check for the committed C1 report."""
    if not REPORT_PATH.exists():
        return "MISSING"
    rep = json.loads(REPORT_PATH.read_text())
    ids = {e["model_id"] for e in rep.get("examples", [])}
    return staleness(REPORT_PATH, _input_hash(sessions("evaluation", directory),
                                              sessions("calibration", directory), ids))
