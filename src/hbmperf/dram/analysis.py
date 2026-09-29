"""Simulator studies: loaded-latency curves, pattern efficiency tables, KV block-size sweep."""

from __future__ import annotations

from hbmperf.dram import sim, timing, traffic
from hbmperf.provenance import DEFAULT_SEED, ROOT, run_meta, utc_now, write_json

LOAD_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1)
EFFICIENCY_PATTERNS = ("stream", "random", "strided", "mix", "kv_gather", "llm_decode")


def _run(p, pattern, requests, seed, block_bytes=1024, rate=None, **cfg):
    addr, writes = traffic.addresses(p, pattern, requests, seed=seed, block_bytes=block_bytes)
    src = (traffic.open_loop(p, addr, writes, rate, seed) if rate
           else traffic.closed_loop(addr, writes))
    return sim.simulate(p, src, sim.Config(**cfg))


def loaded_latency_curve(preset_id: str, pattern: str, *, requests: int = 20000,
                         seed: int = DEFAULT_SEED, block_bytes: int = 1024) -> list[dict]:
    """Latency versus offered load, as fractions of the closed-loop saturation bandwidth."""
    p = timing.load(preset_id)
    sat = _run(p, pattern, requests, seed, block_bytes).achieved_gb_per_s
    rows = []
    for f in LOAD_FRACTIONS:
        r = _run(p, pattern, requests, seed, block_bytes, rate=f * sat)
        rows.append({"preset": preset_id, "pattern": pattern, "load_fraction": f,
                     "offered_gb_per_s": f * sat, "achieved_gb_per_s": r.achieved_gb_per_s,
                     "read_latency_ns_mean": r.read_latency_ns_mean,
                     "read_latency_ns_p50": r.read_latency_ns_p50,
                     "read_latency_ns_p99": r.read_latency_ns_p99})
    return rows


def efficiency_table(requests: int = 40000, seed: int = DEFAULT_SEED) -> dict:
    out = {}
    for pid in timing.all_ids():
        p = timing.load(pid)
        out[pid] = {"peak_gb_per_s": p.peak_gb_per_s, "stack_peak_gb_per_s": p.stack_peak_gb_per_s}
        for pattern in EFFICIENCY_PATTERNS:
            for q in (32, 64):
                r = _run(p, pattern, requests, seed, queue_depth=q)
                key = pattern if q == 32 else f"{pattern}_q64"
                out[pid][key] = {"efficiency": r.efficiency, "row_hit_rate": r.row_hit_rate,
                                 "read_latency_ns_mean": r.read_latency_ns_mean,
                                 "read_latency_ns_p99": r.read_latency_ns_p99}
    return out


def write_efficiency_table(requests: int = 40000) -> str:
    config = {"requests": requests, "patterns": EFFICIENCY_PATTERNS, "queue_depths": [32, 64],
              "llm_decode": {"weight_fraction": 0.8, "block_bytes": 1024}}
    data = {"meta": run_meta(config, seed=DEFAULT_SEED), "presets": efficiency_table(requests)}
    data["meta"]["finished_utc"] = utc_now()
    path = ROOT / "results" / "dram" / "efficiency.json"
    write_json(path, data)
    return str(path.relative_to(ROOT))


def kv_block_sweep(preset_id: str = "hbm3_6400", kv_bytes_per_token_layer: int = 4096,
                   block_tokens=(1, 4, 16, 64, 256), requests: int = 40000,
                   seed: int = DEFAULT_SEED) -> list[dict]:
    """Effective bandwidth of paged KV gathers versus block size.

    A paged-attention block of `t` tokens stores `t * kv_bytes_per_token_layer` contiguous bytes
    per layer (default 4096 bytes: Llama-3.3-70B K plus V, 8 KV heads x 128 dims x 2 bytes x 2).
    """
    p = timing.load(preset_id)
    rows = []
    for t in block_tokens:
        block = t * kv_bytes_per_token_layer
        r = _run(p, "kv_gather", requests, seed, block_bytes=block)
        rows.append({"preset": preset_id, "block_tokens": t, "block_bytes": block,
                     "efficiency": r.efficiency, "achieved_gb_per_s": r.achieved_gb_per_s,
                     "row_hit_rate": r.row_hit_rate,
                     "read_latency_ns_mean": r.read_latency_ns_mean})
    return rows


def write_studies(requests: int = 20000) -> list[str]:
    """Write loaded-latency curves and the KV block-size sweep to results/dram/."""
    curves = [row for pid in timing.all_ids() for pattern in ("stream", "random")
              for row in loaded_latency_curve(pid, pattern, requests=requests)]
    kv = [row for pid in timing.all_ids() for row in kv_block_sweep(pid, requests=requests)]
    written = []
    for name, rows, cfg in (("latency_curves", curves, {"requests": requests,
                                                         "load_fractions": LOAD_FRACTIONS}),
                            ("kv_block_sweep", kv, {"requests": requests,
                                                    "kv_bytes_per_token_layer": 4096})):
        data = {"meta": run_meta(cfg, seed=DEFAULT_SEED), "rows": rows}
        data["meta"]["finished_utc"] = utc_now()
        path = ROOT / "results" / "dram" / f"{name}.json"
        write_json(path, data)
        written.append(str(path.relative_to(ROOT)))
    return written
