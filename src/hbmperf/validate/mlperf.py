"""Invariant I1: roofline compute bound versus MLPerf Inference Llama2-70B Offline throughput.

For every mapped closed-division, available-category datacenter row, the uncalibrated
output-token compute bound P / (2 * active_params) per accelerator must be at least the
measured throughput per accelerator. The informative output is the distribution of
measured-over-bound ratios (achieved fraction of peak) by accelerator and round.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from hbmperf import catalog, fetch, llm
from hbmperf.provenance import ROOT, run_meta, utc_now, write_json
from hbmperf.validate import VALIDATION_DIR, claim_input_sha256

ROUNDS = ("5.0", "5.1", "6.0")
MODEL = "llama-2-70b"
MIN_ROWS = 30
DERIVED = ROOT / "data" / "derived" / "mlperf_llama2_70b_offline.csv"
REPORT_PATH = VALIDATION_DIR / "i1.json"

# ordered (pattern, catalog id); first match wins; unmatched accelerators are listed
ACCELERATOR_MAP = [
    (r"GB200", "nvidia_gb200_per_gpu"),
    (r"GH200.*144", "nvidia_gh200_hbm3e"),
    (r"GH200.*96", "nvidia_gh200_hbm3"),
    (r"H200-SXM", "nvidia_h200_sxm"),
    (r"H100-SXM", "nvidia_h100_sxm"),
    (r"H100-PCIe", "nvidia_h100_pcie"),
    (r"\bB200\b", "nvidia_b200"),
    (r"MI355X", "amd_mi355x"),
    (r"MI325X", "amd_mi325x"),
    (r"MI300X", "amd_mi300x"),
    (r"Gaudi ?3", "intel_gaudi3"),
    (r"Gaudi ?2", "intel_gaudi2"),
]
CSP_SUBMITTERS = ("Azure", "Microsoft", "Google", "Oracle", "CoreWeave", "Lambda", "Nebius",
                  "Crusoe", "Vultr", "AWS", "Amazon")


def map_accelerator(name: str) -> str | None:
    for pattern, acc_id in ACCELERATOR_MAP:
        if re.search(pattern, name):
            return acc_id
    return None


def peak_precision(weight_types: str) -> str:
    """Dense peak column for the row's self-reported weight precision (FP8 when unclear)."""
    w = weight_types.lower()
    if "fp4" in w:
        return "nvfp4"
    if "bf16" in w or "fp16" in w:
        return "bf16"
    return "fp8"


def load_rows() -> list[dict]:
    """Deduplicated llama2-70b Offline closed available datacenter rows from all rounds."""
    out, seen = [], set()
    for v in ROUNDS:
        path = fetch.snapshot_path("mlperf", f"summary_results_v{v}.json")
        for r in json.loads(path.read_text()):
            if not (str(r.get("Model", "")).startswith("llama2-70b")
                    and r.get("Scenario") == "Offline" and r.get("Suite") == "datacenter"
                    and r.get("Category") == "closed" and r.get("Availability") == "available"
                    and r.get("Performance_Units") == "Tokens/s"):
                continue
            key = (v, r["Submitter"], r["System"], r.get("Platform"))
            if key in seen:  # -99 and -99.9 variants report identical runs
                continue
            seen.add(key)
            try:
                n = int(float(r["a#"])) * int(float(r.get("Nodes") or 1))
                perf = float(r["Performance_Result"])
            except (TypeError, ValueError):
                continue
            out.append({"round": v, "submitter": r["Submitter"], "system": r["System"],
                        "accelerator": r["Accelerator"], "accelerators_total": n,
                        "weight_data_types": r.get("weight_data_types") or "",
                        "tokens_per_s": perf,
                        "csp": any(c.lower() in r["Submitter"].lower() for c in CSP_SUBMITTERS)})
    return out


def evaluate(write: bool = True) -> dict:
    model = catalog.get("models", MODEL)
    p = llm.params(model)
    active = model["n_layers"] * p.layer_matmul_active() + p.lm_head
    rows, unmapped = [], defaultdict(int)
    used = set()
    for r in load_rows():
        acc = map_accelerator(r["accelerator"])
        prec = peak_precision(r["weight_data_types"])
        dev = llm.device(acc, prec) if acc else None
        if dev is None or dev.peak_flops is None:
            unmapped[r["accelerator"]] += 1
            continue
        used.add(acc)
        bound = dev.peak_flops / (2 * active)
        measured = r["tokens_per_s"] / r["accelerators_total"]
        rows.append({**r, "accelerator_id": acc, "peak_precision": prec,
                     "bound_tokens_per_s_per_acc": bound,
                     "measured_tokens_per_s_per_acc": measured, "ratio": measured / bound})
    ratios = np.array([r["ratio"] for r in rows])
    violations = [r for r in rows if r["ratio"] > 1]
    by = defaultdict(list)
    for r in rows:
        by[(r["accelerator_id"], r["round"])].append(r["ratio"])
    verdict = ("INSUFFICIENT_EVIDENCE" if len(rows) < MIN_ROWS
               else "FAIL" if violations else "PASS")
    rep = {
        "claim": "I1", "verdict": verdict, "n_rows": len(rows), "n_violations": len(violations),
        "violations": [{k: v[k] for k in ("round", "submitter", "system", "accelerator", "ratio")}
                       for v in violations],
        "ratio_quantiles": dict(zip(("min", "p25", "median", "p75", "max"),
                                    np.quantile(ratios, [0, .25, .5, .75, 1]).tolist(),
                                    strict=True)) if len(rows) else {},
        "by_accelerator_round": [{"accelerator_id": a, "round": v, "n": len(x),
                                  "median_ratio": float(np.median(x)), "max_ratio": max(x)}
                                 for (a, v), x in sorted(by.items())],
        "csp_rows": sum(r["csp"] for r in rows),
        "unmapped_accelerators": dict(sorted(unmapped.items())),
        "source": "MLCommons MLPerf Inference Datacenter results v5.0, v5.1, v6.0 "
                  "(Apache-2.0; the v5.0 repository has no license file)",
    }
    files = [fetch.snapshot_path("mlperf", f"summary_results_v{v}.json") for v in ROUNDS]
    rows_used = [catalog.get("accelerators", a) for a in sorted(used)] + [model]
    rep["claim_input_sha256"] = claim_input_sha256(
        ["llm.py", "validate/__init__.py", "validate/mlperf.py"], rows_used, files)
    rep["meta"] = run_meta({"claim": "I1"})
    rep["meta"]["finished_utc"] = utc_now()
    if write:
        write_json(REPORT_PATH, rep)
        write_derived(rows, DERIVED)
    return rep


def write_derived(rows: list[dict], path: Path) -> None:
    cols = ["round", "submitter", "system", "accelerator", "accelerator_id",
            "accelerators_total", "weight_data_types", "peak_precision", "csp", "tokens_per_s",
            "measured_tokens_per_s_per_acc", "bound_tokens_per_s_per_acc", "ratio"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("# Derived from MLCommons MLPerf Inference Datacenter results (Apache-2.0); "
                "https://github.com/mlcommons\n")
        w = csv.DictWriter(f, cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["round"], r["submitter"], r["system"])):
            w.writerow(r)
