"""Command-line interface: `hbmperf <group> <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hbmperf.provenance import DEFAULT_SEED, dumps

EXIT_ERROR, EXIT_BLOCKED, EXIT_FAIL = 1, 2, 3


def _print(obj, as_json: bool, text: str | None = None) -> None:
    if as_json or text is None:
        sys.stdout.write(dumps(obj))
    else:
        print(text)


# ------------------------------------------------------------------ catalog

def cmd_catalog_list(a):
    from hbmperf import catalog

    name = {"memory": "memory_standards", "timing": "dram_timing",
            "instances": "csp_instances"}.get(a.name, a.name)
    rows = catalog.load(name)
    if a.json:
        return _print(list(rows.values()), True)
    keys = {"accelerators": ["id", "year", "memory_type", "memory_gb", "mem_bw_gb_per_s",
                             "bf16_dense_tflops", "fp8_dense_tflops"],
            "memory_standards": ["id", "year", "interface_bits_per_stack",
                                 "max_pin_rate_gbit_per_s", "peak_bw_per_stack_gb_per_s"],
            "dram_timing": ["id", "data_rate_mbit_per_s", "tck_ps", "n_rcd_rd_ck", "n_cl_ck",
                            "n_rp_ck", "n_faw_ck", "n_rfc_ck"],
            "models": ["id", "n_layers", "d_model", "n_heads", "n_kv_heads", "published_params"],
            "tiers": ["id", "link", "peak_bw_gb_per_s", "capacity_gb"],
            "csp_instances": ["id", "accelerator_id", "accelerator_count",
                              "scaleout_gbit_per_s"]}[name]
    widths = [max(len(k), *(len(str(r[k])) for r in rows.values())) for k in keys]
    print("  ".join(k.ljust(w) for k, w in zip(keys, widths, strict=True)))
    for r in rows.values():
        print("  ".join(str(r[k] if r[k] is not None else "-").ljust(w)
                        for k, w in zip(keys, widths, strict=True)))


def cmd_catalog_validate(a):
    from hbmperf import catalog

    counts = catalog.validate_all()
    _print(counts, a.json, "catalog OK: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


# ------------------------------------------------------------------ llm

def _knobs(a):
    from hbmperf import llm

    eta_m = a.eta_m
    if a.efficiency_source == "sim":
        eta_m = sim_efficiency(a.sim_preset)
    return llm.Knobs(eta_m=eta_m, eta_c=a.eta_c, t_fixed=a.t_fixed_us * 1e-6)


def sim_efficiency(preset: str) -> float:
    """Byte-weighted efficiency from committed simulator results for the llm_decode pattern."""
    from hbmperf.provenance import ROOT

    path = ROOT / "results" / "dram" / "efficiency.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; run `hbmperf dram efficiency` first")
    data = json.loads(path.read_text())
    return data["presets"][preset]["llm_decode"]["efficiency"]


def cmd_llm_predict(a):
    from hbmperf import llm

    r = llm.predict(a.model, a.device, a.precision, batch=a.batch, isl=a.isl, osl=a.osl, tp=a.tp,
                    kv_bits=a.kv_bits, knobs=_knobs(a))
    text = (
        f"{a.model} on {a.device} ({a.precision}, tp={a.tp}, batch={a.batch}, "
        f"isl={a.isl}, osl={a.osl})\n"
        f"  feasible:          {r.feasible}{'' if r.feasible else ' - ' + r.reason}\n"
        f"  weights/device:    {r.weight_bytes_per_device / 1e9:.2f} GB, "
        f"KV/device {r.kv_bytes_per_device / 1e9:.2f} GB, "
        f"usable {r.capacity_bytes_per_device / 1e9:.1f} GB\n"
        f"  TTFT:              {r.ttft_s * 1e3:.2f} ms\n"
        f"  TPOT:              {r.tpot_s * 1e3:.3f} ms ({r.bound}-bound)\n"
        f"  throughput:        {r.output_tok_per_s:.1f} tok/s total, "
        f"{r.output_tok_per_s_per_device:.1f} tok/s/device, {r.output_tok_per_s_per_seq:.1f} "
        f"tok/s/sequence\n"
        f"  bandwidth use:     {r.mem_bw_utilization * 100:.1f}% of peak\n"
        f"  arith. intensity:  {r.arithmetic_intensity:.1f} FLOP/byte (ridge "
        f"{'n/a' if r.ridge_point is None else f'{r.ridge_point:.0f}'})\n"
        f"  critical batch B*: {r.critical_batch:.1f}"
    )
    _print(r.to_dict(), a.json, text)


def cmd_llm_sweep(a):
    from hbmperf import llm

    lo, hi = (int(x) for x in a.batch_range.split(":"))
    rows = []
    b = lo
    while b <= hi:
        r = llm.predict(a.model, a.device, a.precision, batch=b, isl=a.isl, osl=a.osl, tp=a.tp,
                        kv_bits=a.kv_bits, knobs=_knobs(a))
        rows.append({"batch": b, "tpot_ms": r.tpot_s * 1e3, "tok_per_s": r.output_tok_per_s,
                     "bound": r.bound, "feasible": r.feasible,
                     "mem_bw_utilization": r.mem_bw_utilization})
        b *= 2
    text = "batch  TPOT_ms   tok/s      bound    feasible\n" + "\n".join(
        f"{r['batch']:<6} {r['tpot_ms']:<9.3f} {r['tok_per_s']:<10.1f} {r['bound']:<8} "
        f"{r['feasible']}" for r in rows)
    _print(rows, a.json, text)


# ------------------------------------------------------------------ dram

def _traffic(a, p):
    from hbmperf.dram import traffic

    addr, writes = traffic.addresses(p, a.pattern, a.requests, seed=a.seed,
                                     block_bytes=a.kv_block_bytes)
    if a.rate_gb_per_s:
        return traffic.open_loop(p, addr, writes, a.rate_gb_per_s, a.seed)
    return traffic.closed_loop(addr, writes)


def _dram_config(a, log=False):
    from hbmperf.dram import sim

    mapping = "RoSiBaBgCo" if a.pattern == "single_bank" else a.mapping
    return sim.Config(queue_depth=a.queue_depth, mapping=mapping, page_policy=a.page_policy,
                      refresh=not a.no_refresh, log_commands=log)


def cmd_dram_sim(a):
    from hbmperf.dram import sim, timing

    p = timing.load(a.preset)
    r = sim.simulate(p, _traffic(a, p), _dram_config(a, log=bool(a.log)))
    if a.log:
        Path(a.log).write_text(dumps({"preset": a.preset, "log": r.command_log}))
    d = r.to_dict()
    d["stack_achieved_gb_per_s"] = r.achieved_gb_per_s * p.pcs_per_stack
    text = (
        f"{a.preset} {a.pattern}: {r.requests} requests, {r.cycles} cycles\n"
        f"  bandwidth:  {r.achieved_gb_per_s:.2f} GB/s per pseudo-channel "
        f"({r.efficiency * 100:.1f}% of {r.peak_gb_per_s:.1f}); "
        f"{d['stack_achieved_gb_per_s']:.0f} GB/s per stack\n"
        f"  row buffer: {r.row_hit_rate * 100:.1f}% hits, {r.row_misses} misses, "
        f"{r.row_conflicts} conflicts\n"
        f"  read latency ns: mean {r.read_latency_ns_mean:.1f}, p50 {r.read_latency_ns_p50:.1f},"
        f" p95 {r.read_latency_ns_p95:.1f}, p99 {r.read_latency_ns_p99:.1f}\n"
        f"  refreshes:  {r.refreshes}"
    )
    _print(d, a.json, text)


def cmd_dram_curve(a):
    from hbmperf.dram import analysis

    rows = analysis.loaded_latency_curve(a.preset, a.pattern, requests=a.requests, seed=a.seed,
                                         block_bytes=a.kv_block_bytes)
    text = "offered_GB/s  achieved_GB/s  mean_ns  p99_ns\n" + "\n".join(
        f"{r['offered_gb_per_s']:<13.2f} {r['achieved_gb_per_s']:<14.2f} "
        f"{r['read_latency_ns_mean']:<8.1f} {r['read_latency_ns_p99']:.1f}" for r in rows)
    _print(rows, a.json, text)


def cmd_dram_check(a):
    from hbmperf.dram import checker, timing

    data = json.loads(Path(a.log).read_text())
    rep = checker.check_log(timing.load(data["preset"]), [tuple(e) for e in data["log"]])
    _print(rep, a.json, f"{rep['commands']} commands, {rep['violations']} violations "
                        f"{rep['by_constraint'] or ''}")
    return EXIT_FAIL if rep["violations"] else 0


def cmd_dram_efficiency(a):
    from hbmperf.dram import analysis

    path = analysis.write_efficiency_table(requests=a.requests)
    print(f"wrote {path}")


# ------------------------------------------------------------------ data and measurement

def cmd_fetch(a):
    from hbmperf import fetch

    print(f"wrote {fetch.fetch(a.source, a.date)}")


def cmd_measure_host(a):
    from hbmperf import measure

    out = measure.measure_host()
    host = json.loads((out / "host.json").read_text())
    for r in host["latency"]:
        print(f"  {r['working_set_bytes'] / 1024:>10.0f} KiB  {r['latency_ns']:7.2f} ns")
    for r in host["bandwidth"]:
        print(f"  {r['threads']} threads: {r['read_gb_per_s']:.1f} GB/s read")
    print(f"wrote {out}")


def cmd_measure_llamacpp(a):
    from hbmperf import measure

    if a.probe:
        return _print(measure.probe(), True)
    if a.download_only:
        rows = measure.download_split(a.split)
        return _print(rows, a.json, "\n".join(
            f"verified {r['filename']}: guard (b) {r['guard_b_rel_error'] * 100:+.3f}%"
            for r in rows))
    out = measure.measure_split(a.split, out=Path(a.resume) if a.resume else None)
    print(f"wrote {out}")


# ------------------------------------------------------------------ validation

def _verdict_text(rep: dict) -> str:
    lines = [f"{rep['claim']}: {rep['verdict']}"]
    c = rep.get("comparison")
    if c:
        lines.append(f"  MdAPE {c['mdape'] * 100:.1f}% (95% CI {c['mdape_ci'][0] * 100:.1f}-"
                     f"{c['mdape_ci'][1] * 100:.1f}%), baseline {c['baseline_mdape'] * 100:.1f}%, "
                     f"{c['n_clusters']} clusters / {c['n_examples']} examples")
    for r in rep.get("reasons", []):
        lines.append(f"  reason: {r}")
    if rep.get("fallback_claim"):
        lines.append(f"  claim tree: {rep['fallback_claim']}")
    return "\n".join(lines)


def cmd_validate_local(a):
    from hbmperf.validate import local

    if a.calibrate:
        cal = local.calibrate()
        return _print(cal, a.json, f"calibrated on {cal['n_examples']} examples: eta_m="
                      f"{cal['candidate']['eta_m']:.3f}, t_fixed="
                      f"{cal['candidate']['t_fixed_s'] * 1e3:.3f} ms")
    if a.status:
        return print(local.status())
    rep = local.evaluate()
    _print(rep, a.json, _verdict_text(rep))
    return EXIT_FAIL if a.strict and rep["verdict"] != "PASS" else 0


def cmd_validate_inferencex(a):
    from hbmperf.validate import inferencex

    if a.calibrate:
        cal = inferencex.calibrate()
        return _print(cal, a.json, f"calibrated on {cal['n_examples']} rows: {cal['knobs']}, "
                      f"calibration MdAPE {cal['calibration_mdape'] * 100:.1f}%")
    rep = inferencex.evaluate()
    text = _verdict_text(rep) + (
        f"\n  C2-FP4 (reported only): MdAPE "
        f"{rep['fp4']['comparison']['mdape'] * 100:.1f}% over {rep['fp4']['n_rows']} rows")
    _print(rep, a.json, text)
    return EXIT_FAIL if a.strict and rep["verdict"] != "PASS" else 0


def cmd_validate_mlperf(a):
    from hbmperf.validate import mlperf

    rep = mlperf.evaluate()
    q = rep["ratio_quantiles"]
    _print(rep, a.json, f"I1: {rep['verdict']} over {rep['n_rows']} rows, "
                        f"{rep['n_violations']} violations; measured/bound median "
                        f"{q.get('median', 0):.3f}, max {q.get('max', 0):.3f}; "
                        f"unmapped {rep['unmapped_accelerators']}")
    return EXIT_FAIL if a.strict and rep["verdict"] != "PASS" else 0


# ------------------------------------------------------------------ analyses

def cmd_hierarchy_sweep(a):
    from hbmperf import hierarchy

    rows = hierarchy.offload_sweep(a.model, a.device, a.precision, a.tier, offload=a.offload,
                                   batch=a.batch, isl=a.isl, osl=a.osl, tp=a.tp)
    text = "fraction  max_batch  batch  TPOT_ms(overlap..serial)  tok/s(serial..overlap)\n" + \
        "\n".join(f"{r['fraction']:<9} {r['max_feasible_batch']:<10} {r['batch']:<6} "
                  f"{r['tpot_s_lower'] * 1e3:8.2f}..{r['tpot_s_upper'] * 1e3:<8.2f}        "
                  f"{r['tok_per_s_lower']:8.0f}..{r['tok_per_s_upper']:<8.0f}"
                  + ("" if r["feasible"] else "  (exceeds tier capacity)")
                  if r.get("batch") else f"{r['fraction']:<9} {0:<10} no batch fits in HBM"
                  for r in rows)
    _print(rows, a.json, text + "\n(model-only analysis; not validated)")


def cmd_cluster_table(a):
    from hbmperf import hierarchy

    rows = hierarchy.cluster_table(a.model, a.precision)
    text = "\n".join(f"{r['instance']:<38} {r['accelerators']}x {r['accelerator_id']:<22} "
                     + (f"batch {r['batch']:<4} {r['tok_per_s']:9.0f} tok/s ({r['bound']})"
                        if r["feasible"] else r["reason"]) for r in rows)
    _print(rows, a.json, text)


def cmd_trends(a):
    from hbmperf import trends

    if a.cmd == "fit":
        path = trends.write_results()
        print(f"wrote {path}")
    res = trends.analyze()
    lines = []
    for key, f in res["fits"].items():
        if f.get("doubling_years"):
            ci = f.get("doubling_years_ci") or [float("nan")] * 2
            lines.append(f"{key:<11} {f['n_devices']:>3} devices, doubles every "
                         f"{f['doubling_years']:.2f} y (95% CI {ci[0]:.2f}-{ci[1]:.2f})")
        if "intercept_year" in f:
            yr, ci = f["intercept_year"], f.get("intercept_year_ci")
            desc = res["targets"]["bandwidth_desc" if key == "mem_bw" else "capacity_desc"]
            lines.append(f"  intercept: {desc}: "
                         + ("none in range" if yr is None else f"{yr:.1f}")
                         + (f" (95% CI {ci[0]:.1f}-{ci[1]:.1f})" if ci else ""))
    _print(res["fits"], a.json, "\n".join(lines))


def cmd_dram_studies(a):
    from hbmperf.dram import analysis

    for path in analysis.write_studies(requests=a.requests):
        print(f"wrote {path}")


def cmd_analysis_hierarchy(a):
    from hbmperf import hierarchy

    print(f"wrote {hierarchy.write_results()}")


def cmd_dashboard(a):
    from hbmperf import dashboard

    print(f"wrote {dashboard.build(Path(a.out))}")


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hbmperf", description=__doc__)
    sub = ap.add_subparsers(dest="group", required=True)

    def add(parent, name, func, help_):
        sp = parent.add_parser(name, help=help_)
        sp.set_defaults(func=func)
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        return sp

    cat = sub.add_parser("catalog", help="cited input catalogs").add_subparsers(
        dest="cmd", required=True)
    sp = add(cat, "list", cmd_catalog_list, "list a catalog")
    sp.add_argument("name", choices=["accelerators", "memory", "timing", "models", "tiers",
                                     "instances"])
    add(cat, "validate", cmd_catalog_validate, "validate every catalog")

    lg = sub.add_parser("llm", help="first-order LLM inference model").add_subparsers(
        dest="cmd", required=True)
    for name, func in (("predict", cmd_llm_predict), ("sweep", cmd_llm_sweep)):
        sp = add(lg, name, func, f"{name} decode and prefill performance")
        sp.add_argument("--model", required=True)
        sp.add_argument("--device", required=True)
        sp.add_argument("--precision", required=True,
                        choices=["bf16", "f16", "q8_0", "fp8", "nvfp4", "mxfp4"])
        sp.add_argument("--tp", type=int, default=1)
        sp.add_argument("--isl", type=int, default=1024)
        sp.add_argument("--osl", type=int, default=128)
        sp.add_argument("--kv-bits", type=float)
        sp.add_argument("--efficiency-source", choices=["peak", "sim"], default="peak")
        sp.add_argument("--sim-preset", default="hbm3_6400")
        sp.add_argument("--eta-m", type=float, default=1.0)
        sp.add_argument("--eta-c", type=float, default=1.0)
        sp.add_argument("--t-fixed-us", type=float, default=0.0)
        if name == "predict":
            sp.add_argument("--batch", type=int, default=1)
        else:
            sp.add_argument("--batch-range", default="1:256")

    dr = sub.add_parser("dram", help="HBM pseudo-channel simulator").add_subparsers(
        dest="cmd", required=True)
    for name, func in (("sim", cmd_dram_sim), ("curve", cmd_dram_curve)):
        sp = add(dr, name, func, "simulate a traffic pattern" if name == "sim"
                 else "loaded-latency curve")
        sp.add_argument("--preset", default="hbm3_6400")
        sp.add_argument("--pattern", default="stream",
                        choices=["stream", "random", "strided", "mix", "kv_gather",
                                 "llm_decode", "single_bank"])
        sp.add_argument("--requests", type=int, default=20000)
        sp.add_argument("--queue-depth", type=int, default=32)
        sp.add_argument("--mapping", default="RoCoSiBaBg")
        sp.add_argument("--page-policy", choices=["open", "closed"], default="open")
        sp.add_argument("--no-refresh", action="store_true")
        sp.add_argument("--kv-block-bytes", type=int, default=1024)
        sp.add_argument("--rate-gb-per-s", type=float)
        sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
        sp.add_argument("--log", help="write the command log to this JSON file")
    sp = add(dr, "check", cmd_dram_check, "verify a command log with the timing checker")
    sp.add_argument("--log", required=True)
    sp = add(dr, "efficiency", cmd_dram_efficiency, "write results/dram/efficiency.json")
    sp.add_argument("--requests", type=int, default=40000)

    sp = add(sub, "fetch", cmd_fetch, "download a public dataset snapshot")
    sp.add_argument("source", choices=["epoch", "inferencex", "mlperf"])
    sp.add_argument("--date")

    ms = sub.add_parser("measure", help="local measurements").add_subparsers(
        dest="cmd", required=True)
    add(ms, "host", cmd_measure_host, "C latency and bandwidth microbenchmarks")
    sp = add(ms, "llamacpp", cmd_measure_llamacpp, "llama-bench decode measurements (C1)")
    sp.add_argument("--split", choices=["calibration", "evaluation"], default="calibration")
    sp.add_argument("--probe", action="store_true")
    sp.add_argument("--download-only", action="store_true",
                    help="download and verify files without benchmarking")
    sp.add_argument("--resume", help="continue an interrupted session file")
    va = sub.add_parser("validate", help="compare predictions with measurements").add_subparsers(
        dest="cmd", required=True)
    sp = add(va, "local", cmd_validate_local, "claim C1: llama-bench on the local machine")
    sp.add_argument("--calibrate", action="store_true", help="fit on the calibration split")
    sp.add_argument("--status", action="store_true", help="report CURRENT, STALE, or MISSING")
    sp.add_argument("--strict", action="store_true", help="exit 3 unless the verdict is PASS")
    sp = add(va, "inferencex", cmd_validate_inferencex, "claim C2: held-out accelerators")
    sp.add_argument("--calibrate", action="store_true")
    sp.add_argument("--strict", action="store_true")
    sp = add(va, "mlperf", cmd_validate_mlperf, "invariant I1: MLPerf compute bound")
    sp.add_argument("--strict", action="store_true")

    hi = sub.add_parser("hierarchy", help="memory-tier offload trade-offs").add_subparsers(
        dest="cmd", required=True)
    sp = add(hi, "sweep", cmd_hierarchy_sweep, "offload fraction sweep")
    sp.add_argument("--model", default="llama-3.3-70b")
    sp.add_argument("--device", default="nvidia_gh200_hbm3e")
    sp.add_argument("--precision", default="fp8")
    sp.add_argument("--tier", default="gh200_c2c_lpddr5x")
    sp.add_argument("--offload", choices=["kv", "weights"], default="kv")
    sp.add_argument("--batch", type=int, help="fixed batch (default: largest feasible)")
    sp.add_argument("--isl", type=int, default=8192)
    sp.add_argument("--osl", type=int, default=1024)
    sp.add_argument("--tp", type=int, default=1)
    add(hi, "write", cmd_analysis_hierarchy, "write results/analysis/hierarchy.json")

    cl = sub.add_parser("cluster", help="CSP instance analysis").add_subparsers(
        dest="cmd", required=True)
    sp = add(cl, "table", cmd_cluster_table, "predicted throughput per CSP instance")
    sp.add_argument("--model", default="llama-3.3-70b")
    sp.add_argument("--precision", default="fp8")

    tr = sub.add_parser("trends", help="hardware trends and intercepts").add_subparsers(
        dest="cmd", required=True)
    add(tr, "fit", cmd_trends, "fit trends and write results/trends/trends.json")
    add(tr, "intercepts", cmd_trends, "print fits and intercepts from committed data")

    sp = add(dr, "studies", cmd_dram_studies, "write latency curves and KV block sweep")
    sp.add_argument("--requests", type=int, default=20000)

    db = sub.add_parser("dashboard", help="static HTML dashboard").add_subparsers(
        dest="cmd", required=True)
    sp = add(db, "build", cmd_dashboard, "build the dashboard from committed results")
    sp.add_argument("--out", default="docs/dashboard/index.html")
    return ap


def main(argv: list[str] | None = None) -> int:
    from hbmperf.catalog import CatalogError
    from hbmperf.measure import Blocked
    from hbmperf.provenance import ProvenanceError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except Blocked as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return EXIT_BLOCKED
    except (CatalogError, ProvenanceError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
