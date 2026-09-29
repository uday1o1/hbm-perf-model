"""Memory-hierarchy and cluster trade-offs (model-only analyses; not validated).

KV-cache (or weight) offload to a second tier: a fraction `f` of the offloaded bytes is read
from the tier each decode step. The memory time lies between perfect overlap,
max(B1/W1, B2/W2), and full serialization, B1/W1 + B2/W2; both bounds are reported.
Offload frees HBM capacity, which raises the largest feasible batch.
"""

from __future__ import annotations

import math

from hbmperf import catalog, llm

DEFAULT_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)


def tier_bandwidth(tier: dict) -> float:
    return (tier["measured_bw_gb_per_s"] or tier["peak_bw_gb_per_s"]) * 1e9


def max_feasible_batch(model: dict, dev: llm.Device, precision: str, *, max_context: int,
                       tp: int = 1, kv_fraction_in_hbm: float = 1.0,
                       weight_fraction_in_hbm: float = 1.0, kv_bits: float | None = None,
                       usable: float = 0.9) -> int:
    w, kv1 = llm.footprint_bytes(model, precision, batch=1, max_context=max_context,
                                 kv_bits=kv_bits)
    free = dev.mem_bytes * usable - w * weight_fraction_in_hbm / tp
    per_seq = kv1 * kv_fraction_in_hbm / tp
    if free <= 0:
        return 0
    return math.inf if per_seq == 0 else int(free // per_seq)


def offload_sweep(model_id: str, device_id: str, precision: str, tier_id: str, *,
                  offload: str = "kv", fractions=DEFAULT_FRACTIONS, batch: int | None = None,
                  isl: int = 1024, osl: int = 1024, tp: int = 1, kv_bits: float | None = None,
                  max_batch_cap: int = 1024) -> list[dict]:
    """TPOT bounds and throughput versus offload fraction.

    With `batch=None` each row uses the largest feasible batch (capped), showing the
    capacity-versus-bandwidth trade-off; otherwise the batch is fixed.
    """
    if offload not in ("kv", "weights"):
        raise ValueError("offload must be 'kv' or 'weights'")
    model = catalog.get("models", model_id)
    dev = llm.device(device_id, precision)
    tier = catalog.get("tiers", tier_id)
    w2 = tier_bandwidth(tier)
    ctx = isl + osl / 2
    rows = []
    for f in fractions:
        kv_in, w_in = (1 - f, 1.0) if offload == "kv" else (1.0, 1 - f)
        bmax = max_feasible_batch(model, dev, precision, max_context=isl + osl, tp=tp,
                                  kv_fraction_in_hbm=kv_in, weight_fraction_in_hbm=w_in,
                                  kv_bits=kv_bits)
        b = batch if batch is not None else min(bmax, max_batch_cap)
        if b < 1:
            rows.append({"fraction": f, "max_feasible_batch": bmax, "batch": 0,
                         "feasible": False})
            continue
        s = llm.step(model, dev, precision, phase="decode", batch=b, context=ctx, tp=tp,
                     kv_bits=kv_bits)
        off = s.bytes_kv * f if offload == "kv" else s.bytes_weights * f
        hbm = s.bytes_total - off
        t_lo = max(hbm / dev.mem_bw, off / w2)
        t_hi = hbm / dev.mem_bw + off / w2
        tc = s.t_compute or 0.0
        tpot_lo, tpot_hi = max(t_lo, tc) + s.t_comm, max(t_hi, tc) + s.t_comm
        tier_cap = (tier["capacity_gb"] or math.inf) * 1e9
        w, kv = llm.footprint_bytes(model, precision, batch=b, max_context=isl + osl,
                                    kv_bits=kv_bits)
        resident_in_tier = (kv if offload == "kv" else w) * f / tp
        rows.append({
            "fraction": f, "max_feasible_batch": bmax, "batch": b,
            "feasible": b <= bmax and resident_in_tier <= tier_cap,
            "tier_bytes_per_step": off, "hbm_bytes_per_step": hbm,
            "tpot_s_lower": tpot_lo, "tpot_s_upper": tpot_hi,
            "tok_per_s_upper": b / tpot_lo, "tok_per_s_lower": b / tpot_hi,
        })
    return rows


def cluster_table(model_id: str = "llama-3.3-70b", precision: str = "fp8", isl: int = 1024,
                  osl: int = 1024, max_batch_cap: int = 256) -> list[dict]:
    """Per CSP instance: aggregate HBM, whether the model fits at TP = accelerator count, and
    predicted decode throughput at the largest feasible batch (capped)."""
    model = catalog.get("models", model_id)
    rows = []
    for inst in catalog.load("csp_instances").values():
        acc = catalog.get("accelerators", inst["accelerator_id"])
        n = inst["accelerator_count"]
        dev = llm.device(acc["id"], precision)
        row = {"instance": inst["id"], "csp": inst["csp"], "accelerator_id": acc["id"],
               "accelerators": n, "hbm_total_gb": acc["memory_gb"] * n,
               "hbm_bw_total_gb_per_s": acc["mem_bw_gb_per_s"] * n,
               "scaleout_gbit_per_s": inst["scaleout_gbit_per_s"]}
        if dev.peak_flops is None or (n > 1 and dev.link_bw is None):
            rows.append({**row, "feasible": False, "reason": "missing peak or link spec"})
            continue
        bmax = max_feasible_batch(model, dev, precision, max_context=isl + osl, tp=n)
        b = min(bmax, max_batch_cap)
        if b < 1:
            rows.append({**row, "feasible": False, "reason": "model does not fit"})
            continue
        r = llm.predict(model_id, acc["id"], precision, batch=b, isl=isl, osl=osl, tp=n)
        rows.append({**row, "feasible": True, "max_feasible_batch": bmax, "batch": b,
                     "tpot_s": r.tpot_s, "tok_per_s": r.output_tok_per_s,
                     "tok_per_s_per_accelerator": r.output_tok_per_s_per_device,
                     "bound": r.bound})
    return sorted(rows, key=lambda r: r["instance"])


SWEEPS = [  # (model, device, precision, tier, offload)
    ("llama-3.3-70b", "nvidia_gh200_hbm3e", "fp8", "gh200_c2c_lpddr5x", "kv"),
    ("llama-3.3-70b", "nvidia_h100_sxm", "fp8", "host_ddr5_pcie_gen5_x16", "kv"),
    ("llama-3.3-70b", "nvidia_h100_sxm", "fp8", "host_ddr5_pcie_gen4_x16", "kv"),
    ("llama-3.3-70b", "nvidia_gh200_hbm3e", "fp8", "gh200_c2c_lpddr5x", "weights"),
]


def write_results() -> str:
    from hbmperf.provenance import ROOT, run_meta, utc_now, write_json

    sweeps = [{"model": m, "device": d, "precision": p, "tier": t, "offload": o,
               "rows": offload_sweep(m, d, p, t, offload=o, isl=8192, osl=1024)}
              for m, d, p, t, o in SWEEPS]
    data = {"sweeps": sweeps, "cluster": cluster_table(),
            "note": "model-only analyses; not validated against measurement",
            "meta": run_meta({"sweeps": SWEEPS, "isl": 8192, "osl": 1024})}
    data["meta"]["finished_utc"] = utc_now()
    path = ROOT / "results" / "analysis" / "hierarchy.json"
    write_json(path, data)
    return str(path.relative_to(ROOT))
