"""C4: cross-check the hbmperf simulator against Ramulator 2 on identical traces.

Run from the repository with `uv run python tools/ramulator_crosscheck/run.py` after
`tools/ramulator_crosscheck/build.sh`. Ramulator 2 runs in a separate interpreter
($RAMULATOR_PYTHON, default python3.14) because its binding is built for that Python.

Matched configuration (BUILD_PLAN Milestone 8): Ramulator HBM3_16Gb_4hi + HBM3_6400Mbps,
HBM34 controller (32-entry read and write buffers), FR-FCFS, open page, all-bank refresh,
RoBaRaCoCh mapping; hbmperf preset hbm3_6400_4hi with mapping RoBaBgSiCo and 32-entry
queues. Traces target pseudo-channel 0 only: hbmperf addresses gain a zero pseudo-channel bit
above the column field, and every address is checked to decode to the same tuple on both sides.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from hbmperf.dram import sim, timing, traffic
from hbmperf.dram.mapping import Mapper
from hbmperf.provenance import ROOT, run_meta, utc_now, write_json

PRESET = "hbm3_6400_4hi"
MAPPING = "RoBaBgSiCo"
REQUESTS = 40000
BW_TOL, HIT_TOL = 0.15, 0.05
SUITE = [  # (label, pattern, kwargs)
    ("stream", "stream", {}),
    ("random", "random", {}),
    ("strided_4KiB", "strided", {"stride": 4096}),
    ("mix_30pct_writes", "mix", {"write_fraction": 0.3}),
    ("kv_gather_16tok", "kv_gather", {"block_bytes": 16 * 4096}),
]
DIFFERENCES = [
    "Ramulator 2 protects an open row only while the request that opened it is unserved; "
    "hbmperf never precharges a bank with any queued row hit in the active queue.",
    "Ramulator 2 shares its controller buffers across the two pseudo-channels of a channel; "
    "traces target pseudo-channel 0 only so the second pseudo-channel is idle.",
    "Ramulator 2's trace frontend issues at most one request per cycle; hbmperf admits "
    "requests whenever a queue slot is free.",
    "Ramulator 2 models nPPD and command-bus occupancy classes; hbmperf allows one row and one "
    "column command per cycle.",
    "Ramulator 2's PassThroughChannelMapper crashes (SIGSEGV) at the pinned commit with trace "
    "frontends; CacheLineInterleave is used instead, which passes the address through unchanged "
    "for a single channel (cache_line_interleave.cpp line 38).",
    "Ramulator 2's AllBank refresh manager and hbmperf's all-bank refresh may schedule the "
    "precharge-all differently.",
]

RAMULATOR_SCRIPT = r"""
import json, sys, ramulator
trace = sys.argv[1]
fe = ramulator.frontend.LoadStoreTrace(clock_ratio=1, path=trace)
dram = ramulator.dram.HBM3(org_preset="HBM3_16Gb_4hi", timing_preset="HBM3_6400Mbps")
ctrl = ramulator.controller.HBM34(dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(), row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())
mem = ramulator.memory_system.GenericDRAM(clock_ratio=1, controllers=[ctrl],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
s = ramulator.Simulation(fe, mem)
s.run()
c = s.stats["memory_system"]["controller"]
print(json.dumps({k: v for k, v in c.items() if isinstance(v, (int, float))}))
"""


def ramulator_address(p, mapper: Mapper, addr: int) -> int:
    """Insert a zero pseudo-channel bit between the stack-ID field and the column field."""
    low_bits = mapper.offset_bits + mapper.widths["co"]
    low = addr & ((1 << low_bits) - 1)
    return ((addr >> low_bits) << (low_bits + 1)) | low


def ramulator_decode(p, addr: int) -> tuple[int, int, int, int, int, int]:
    """Python mirror of Ramulator 2 RoBaRaCoCh for HBM3 (pc, sid, bg, bank, row, col)."""
    x = addr >> (p.burst_bytes.bit_length() - 1)
    co = x & (p.bursts_per_row - 1)
    x >>= p.bursts_per_row.bit_length() - 1
    out = []
    for n in (2, p.sids, p.bank_groups, p.banks_per_group):  # PseudoChannel, Sid, BG, Bank
        w = n.bit_length() - 1
        out.append(x & ((1 << w) - 1))
        x >>= w
    return (*out, x % p.rows, co)


def main() -> int:
    p = timing.load(PRESET)
    m = Mapper(p, MAPPING)
    py = os.environ.get("RAMULATOR_PYTHON", "python3.14")
    ram_dir = Path(os.environ.get("RAMULATOR_DIR", ROOT / ".cache" / "ramulator2"))
    out_dir = ROOT / "results" / "crosscheck"
    report = {"claim": "C4", "preset": PRESET, "mapping": MAPPING, "requests": REQUESTS,
              "tolerances": {"bandwidth_rel": BW_TOL, "row_hit_abs": HIT_TOL},
              "documented_differences": DIFFERENCES, "patterns": []}
    if not (ram_dir / "python" / "ramulator").exists():
        report.update(verdict="BLOCKED", reasons=[f"Ramulator 2 not built at {ram_dir.name}; "
                                                  "run tools/ramulator_crosscheck/build.sh"])
        write_json(out_dir / "c4.json", report)
        print(report["reasons"][0])
        return 2
    reasons = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, pattern, kw in SUITE:
            addr, writes = traffic.addresses(p, pattern, REQUESTS, **kw)
            ram_addrs = [ramulator_address(p, m, int(a)) for a in addr]
            for a, ra in zip(addr.tolist(), ram_addrs, strict=True):
                pc, *rest = ramulator_decode(p, ra)
                if pc != 0 or tuple(rest) != m.decode(a):
                    raise SystemExit(f"decode mismatch for {pattern} address {a}")
            trace = Path(tmp) / f"{label}.trace"
            trace.write_text("".join(f"{'ST' if w else 'LD'} {ra}\n"
                                     for ra, w in zip(ram_addrs, writes.tolist(), strict=True)))
            h = sim.simulate(p, traffic.closed_loop(addr, writes), sim.Config(mapping=MAPPING))
            env = {**os.environ, "PYTHONPATH": str(ram_dir / "python")}
            proc = subprocess.run([py, "-c", RAMULATOR_SCRIPT, str(trace)], capture_output=True,
                                  text=True, env=env, timeout=1800)
            if proc.returncode != 0:
                raise SystemExit(f"Ramulator 2 failed on {label}: {proc.stderr[-500:]}")
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            n_r = r["num_read_reqs_served"] + r["num_write_reqs_served"]
            r_bw = r["total_throughput_MBps"] * 1e6 / 1e9
            r_hit = r["row_hits"] / max(r["row_hits"] + r["row_misses"] + r["row_conflicts"], 1)
            row = {"pattern": label, "requests": REQUESTS,
                   "hbmperf_gb_per_s": h.achieved_gb_per_s, "ramulator_gb_per_s": r_bw,
                   "bandwidth_rel_diff": h.achieved_gb_per_s / r_bw - 1,
                   "hbmperf_row_hit_rate": h.row_hit_rate, "ramulator_row_hit_rate": r_hit,
                   "row_hit_abs_diff": h.row_hit_rate - r_hit,
                   "hbmperf_read_latency_ns": h.read_latency_ns_mean,
                   "ramulator_read_latency_ns": r["avg_read_latency"] * p.tck_ps / 1000,
                   "ramulator_requests_served": n_r}
            row["pass"] = (abs(row["bandwidth_rel_diff"]) <= BW_TOL
                           and abs(row["row_hit_abs_diff"]) <= HIT_TOL)
            if not row["pass"]:
                reasons.append(f"{label} outside tolerance")
            report["patterns"].append(row)
            print(f"{label:<18} hbmperf {h.achieved_gb_per_s:6.2f} GB/s hit {h.row_hit_rate:.3f}"
                  f" | ramulator {r_bw:6.2f} GB/s hit {r_hit:.3f} | "
                  f"{'ok' if row['pass'] else 'OUT'}")
    report["verdict"] = "FAIL" if reasons else "PASS"
    report["reasons"] = reasons
    report["ramulator_commit"] = "72427a1bba3771564c4fb0e494ba02242fd1eaa7"
    report["meta"] = run_meta({"claim": "C4", "suite": [s[0] for s in SUITE]})
    report["meta"]["finished_utc"] = utc_now()
    write_json(out_dir / "c4.json", report)
    with open(out_dir / "c4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(report["patterns"][0]), lineterminator="\n")
        w.writeheader()
        w.writerows(report["patterns"])
    print(f"C4: {report['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
