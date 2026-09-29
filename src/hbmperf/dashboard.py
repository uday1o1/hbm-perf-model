"""Static HTML dashboard built only from committed results and catalogs.

Every panel lists its sources; a panel whose input is missing says "not available" instead of
failing, so the page always builds.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import plotly.graph_objects as go

from hbmperf import catalog, llm
from hbmperf.provenance import ROOT, utc_now

RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "dashboard" / "index.html"
LAYOUT = dict(template="plotly_white", autosize=True, height=420,
              margin=dict(l=60, r=20, t=40, b=50),
              legend=dict(orientation="h", y=-0.2))


def _load(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def _csv(rel: str) -> list[dict] | None:
    import csv

    p = ROOT / rel
    if not p.exists():
        return None
    lines = [ln for ln in p.read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


class Panel:
    def __init__(self, pid: str, title: str, sources: list[str], note: str = ""):
        self.pid, self.title, self.sources, self.note = pid, title, sources, note
        self.body = ""

    def figure(self, fig: go.Figure, **layout) -> Panel:
        fig.update_layout(**{**LAYOUT, **layout})
        self.body = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig-{self.pid}",
                                default_width="100%", default_height="420px",
                                config={"displaylogo": False, "responsive": True})
        return self

    def table(self, header: list[str], rows: list[list]) -> Panel:
        th = "".join(f"<th>{html.escape(h)}</th>" for h in header)
        tr = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
                     for r in rows)
        self.body = f"<div class='scroll'><table><thead><tr>{th}</tr></thead>" \
                    f"<tbody>{tr}</tbody></table></div>"
        return self

    def missing(self, what: str) -> Panel:
        self.body = f"<p class='na'>not available: {html.escape(what)}</p>"
        return self

    def render(self) -> str:
        src = "".join(f"<li>{html.escape(s)}</li>" for s in self.sources)
        note = f"<p class='note'>{html.escape(self.note)}</p>" if self.note else ""
        return (f"<section class='panel' id='{self.pid}'><h2>{html.escape(self.title)}</h2>"
                f"{note}{self.body}<details><summary>Sources</summary><ul>{src}</ul></details>"
                f"</section>")


def _fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


# ------------------------------------------------------------------ panels

def panel_trends() -> list[Panel]:
    devs = _csv("data/derived/epoch_accelerators.csv")
    res = _load("results/trends/trends.json")
    src = ["Epoch AI, Machine Learning Hardware (CC BY)", "results/trends/trends.json"]
    out = []
    p = Panel("trend-growth", "Accelerator bandwidth, compute, and capacity over time", src,
              "Log-scale; lines are log-linear fits with bootstrap doubling-time intervals.")
    if not devs or not res:
        out.append(p.missing("trend data; run `hbmperf trends fit`"))
    else:
        fig = go.Figure()
        for key, col, label, scale in (("mem_bw", "mem_bw_bytes_per_s", "Memory bandwidth (TB/s)",
                                        1e12),
                                       ("bf16_flops", "bf16_flops", "BF16 dense (PFLOP/s)", 1e15),
                                       ("mem_bytes", "mem_bytes", "Memory capacity (100 GB)",
                                        1e11)):
            pts = [(float(d["year"]), float(d[col]) / scale, d["name"]) for d in devs if d[col]]
            f = res["fits"][key]
            dt = f.get("doubling_years")
            ci = f.get("doubling_years_ci")
            name = f"{label}: doubles every {dt:.2f} y" + (
                f" (95% CI {ci[0]:.2f}-{ci[1]:.2f})" if ci else "") if dt else label
            fig.add_scatter(x=[x[0] for x in pts], y=[x[1] for x in pts], mode="markers",
                            name=name, text=[x[2] for x in pts], hovertemplate="%{text}<br>"
                            "%{x:.1f}: %{y:.3g}<extra></extra>")
            if dt:
                xs = [f["year_range"][0], 2030]
                ys = [2 ** (f["log2_intercept"] + f["log2_per_year"] * (x - 2016)) / scale
                      for x in xs]
                fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(dash="dot"),
                                showlegend=False)
        out.append(p.figure(fig, yaxis_type="log", xaxis_title="release year"))
    q = Panel("trend-ridge", "Ridge point (FLOP per byte): the memory wall", src,
              "Dense BF16 FLOP/s divided by memory bandwidth. Rising ridge points mean more "
              "requests must be batched before decode stops being bandwidth-bound.")
    if not res:
        out.append(q.missing("trend data"))
    else:
        rp = res["ridge_points"]
        f = res["fits"]["ridge"]
        fig = go.Figure(go.Scatter(x=[r["year"] for r in rp], y=[r["ridge"] for r in rp],
                                   mode="markers", text=[r["name"] for r in rp],
                                   name="devices"))
        fig.update_layout(title=f"ridge grows {f.get('growth_per_year', 0):.2f}x per year "
                                f"({f['n_devices']} devices)")
        out.append(q.figure(fig, yaxis_type="log", xaxis_title="release year",
                            yaxis_title="FLOP per byte"))
    r = Panel("intercepts", "Architectural intercepts (extrapolations, not forecasts)", src)
    if not res:
        out.append(r.missing("trend data"))
    else:
        t = res["targets"]
        rows = []
        for key, desc, target in (("mem_bw", t["bandwidth_desc"], t["bandwidth_bytes_per_s"]),
                                  ("mem_bytes", t["capacity_desc"], t["capacity_bytes"])):
            f = res["fits"][key]
            yr, ci = f.get("intercept_year"), f.get("intercept_year_ci")
            rows.append([desc, f"{target / 1e12:.2f} TB/s" if key == "mem_bw"
                         else f"{target / 1e9:.0f} GB",
                         "none in 2010-2040" if yr is None else f"{yr:.1f}",
                         f"{ci[0]:.1f}-{ci[1]:.1f}" if ci else "n/a", f["n_devices"]])
        out.append(r.table(["Requirement", "Target", "Trend crosses in", "95% CI",
                            "Devices"], rows))
    return out


def panel_standards() -> Panel:
    rows = [r for r in catalog.load("memory_standards").values()
            if r["id"].startswith("hbm") and r["peak_bw_per_stack_gb_per_s"]]
    p = Panel("hbm-standards", "HBM generations: peak bandwidth per stack",
              ["data/catalog/memory_standards.csv (JEDEC press releases, vendor pages)"])
    fig = go.Figure(go.Scatter(x=[r["year"] for r in rows],
                               y=[r["peak_bw_per_stack_gb_per_s"] for r in rows],
                               mode="markers+text", text=[r["standard"].split(" (")[0]
                                                          for r in rows],
                               textposition="top center"))
    return p.figure(fig, yaxis_type="log", yaxis_title="GB/s per stack", xaxis_title="year")


def panel_roofline() -> Panel:
    p = Panel("roofline", "Roofline: Llama-3.3-70B FP8 decode operating points",
              ["data/catalog/accelerators.csv", "hbmperf.llm first-order model (spec peak)"],
              "Markers show decode steps at batch 1 to 512 with 2k context; the knee is the "
              "ridge point.")
    model = catalog.get("models", "llama-3.3-70b")
    fig = go.Figure()
    for dev_id in ("nvidia_h100_sxm", "nvidia_h200_sxm", "nvidia_b200", "amd_mi300x",
                   "amd_mi355x"):
        dev = llm.device(dev_id, "fp8")
        xs = [0.5, dev.peak_flops / dev.mem_bw, 1e4]
        fig.add_scatter(x=xs, y=[min(dev.peak_flops, x * dev.mem_bw) / 1e12 for x in xs],
                        mode="lines", name=dev_id)
        pts = []
        for b in (1, 4, 16, 64, 256, 512):
            s = llm.step(model, dev, "fp8", phase="decode", batch=b, context=2048)
            pts.append((s.flops / s.bytes_total, s.flops / max(s.t_mem, s.t_compute) / 1e12, b))
        fig.add_scatter(x=[x[0] for x in pts], y=[x[1] for x in pts], mode="markers",
                        text=[f"batch {x[2]}" for x in pts], showlegend=False)
    return p.figure(fig, xaxis_type="log", yaxis_type="log",
                    xaxis_title="arithmetic intensity (FLOP/byte)",
                    yaxis_title="attainable TFLOP/s")


def _scatter_validation(pid, title, rows, xkey, ykey, src, note, badge) -> Panel:
    p = Panel(pid, title, src, note)
    if not rows:
        return p.missing("validation report")
    fig = go.Figure()
    fig.add_scatter(x=[r[xkey] for r in rows], y=[r[ykey] for r in rows], mode="markers",
                    text=[r.get("label", "") for r in rows], name="examples")
    lo = min(min(r[xkey], r[ykey]) for r in rows)
    hi = max(max(r[xkey], r[ykey]) for r in rows)
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dot"), name="y = x")
    fig.update_layout(title=badge)
    return p.figure(fig, xaxis_type="log", yaxis_type="log", xaxis_title="measured (ms)",
                    yaxis_title="predicted (ms)")


def panel_validation() -> list[Panel]:
    out = []
    c1 = _load("results/validation/c1.json")
    rows = [{"m": e["measured_s"] * 1e3, "p": e["predicted_s"] * 1e3,
             "label": f"{e['filename']} depth {e['depth']}"} for e in (c1 or {}).get("examples",
                                                                                   [])]
    badge = "" if not c1 else (f"C1 {c1['verdict']}: MdAPE "
                               f"{_fmt_pct((c1.get('comparison') or {}).get('mdape'))} vs "
                               f"file-size baseline "
                               f"{_fmt_pct((c1.get('comparison') or {}).get('baseline_mdape'))}")
    out.append(_scatter_validation(
        "c1", "C1: decode step time on the local Apple M2 (held-out model families)", rows,
        "m", "p", ["results/validation/c1.json", "data/measurements/c1/ (llama-bench)"],
        "Calibrated on Qwen2.5; evaluated once on SmolLM2 and TinyLlama at depths up to 4096.",
        badge))
    c2 = _load("results/validation/c2.json")
    pr = _csv("data/derived/inferencex_c2_predictions.csv") or []
    rows = [{"m": float(r["measured_tpot_s"]) * 1e3, "p": float(r["predicted_tpot_s"]) * 1e3,
             "label": f"{r['hardware']} {r['framework']} {r['precision']} tp{r['tp']} "
                      f"conc {r['conc']} {r['isl']}/{r['osl']}"} for r in pr]
    badge = "" if not c2 else (f"C2 {c2['verdict']} (FP8): MdAPE "
                               f"{_fmt_pct(c2['comparison'].get('mdape'))}; FP4 extrapolation "
                               f"MdAPE {_fmt_pct(c2['fp4']['comparison'].get('mdape'))}")
    out.append(_scatter_validation(
        "c2", "C2: TPOT on held-out accelerators (B200, MI325X, MI355X)", rows, "m", "p",
        ["results/validation/c2.json", "SemiAnalysis InferenceX public API snapshot"],
        "Calibrated on H100, H200, and MI300X FP8 results only.", badge))
    i1 = _load("results/validation/i1.json")
    p = Panel("i1", "I1: MLPerf Llama2-70B Offline throughput as a fraction of the compute bound",
              ["results/validation/i1.json", "MLCommons MLPerf Inference v5.0-v6.0 (Apache-2.0)"],
              "Each bar is the median achieved fraction of dense peak per accelerator and round; "
              "a value above 1 would mean the model's bound is wrong.")
    if not i1:
        out.append(p.missing("I1 report"))
    else:
        by = i1["by_accelerator_round"]
        fig = go.Figure(go.Bar(x=[f"{b['accelerator_id']} v{b['round']}" for b in by],
                               y=[b["median_ratio"] for b in by],
                               text=[f"n={b['n']}" for b in by]))
        fig.update_layout(title=f"I1 {i1['verdict']}: {i1['n_rows']} rows, "
                                f"{i1['n_violations']} violations, {i1['csp_rows']} "
                                "CSP-submitted")
        out.append(p.figure(fig, yaxis_title="measured / bound"))
    return out


def panel_dram() -> list[Panel]:
    out = []
    eff = _load("results/dram/efficiency.json")
    p = Panel("dram-eff", "Simulated HBM efficiency by access pattern",
              ["results/dram/efficiency.json", "data/catalog/dram_timing.csv (Ramulator 2 "
               "presets; HBM2E and HBM3E derived)"],
              "Fraction of pseudo-channel peak bandwidth, closed-loop, 32-entry queues.")
    if not eff:
        out.append(p.missing("efficiency table"))
    else:
        fig = go.Figure()
        pats = ["stream", "llm_decode", "kv_gather", "random", "mix", "strided"]
        for pid, v in eff["presets"].items():
            fig.add_bar(x=pats, y=[v[k]["efficiency"] for k in pats], name=pid)
        out.append(p.figure(fig, barmode="group", yaxis_title="efficiency"))
    cur = _load("results/dram/latency_curves.json")
    p = Panel("dram-latency", "Loaded-latency curves (simulated)",
              ["results/dram/latency_curves.json"],
              "Mean read latency versus achieved bandwidth under Poisson arrivals.")
    if not cur:
        out.append(p.missing("latency curves"))
    else:
        fig = go.Figure()
        for pid in sorted({r["preset"] for r in cur["rows"]}):
            for pat in ("stream", "random"):
                rs = [r for r in cur["rows"] if r["preset"] == pid and r["pattern"] == pat]
                fig.add_scatter(x=[r["achieved_gb_per_s"] for r in rs],
                                y=[r["read_latency_ns_mean"] for r in rs],
                                mode="lines+markers", name=f"{pid} {pat}")
        out.append(p.figure(fig, yaxis_type="log", xaxis_title="achieved GB/s per pseudo-channel",
                            yaxis_title="mean read latency (ns)"))
    kv = _load("results/dram/kv_block_sweep.json")
    p = Panel("kv-block", "Paged KV-cache block size versus HBM efficiency (simulated)",
              ["results/dram/kv_block_sweep.json"],
              "Blocks of t tokens hold 4096 bytes per token per layer (Llama-3.3-70B, BF16 KV).")
    if not kv:
        out.append(p.missing("KV block sweep"))
    else:
        fig = go.Figure()
        for pid in sorted({r["preset"] for r in kv["rows"]}):
            rs = [r for r in kv["rows"] if r["preset"] == pid]
            fig.add_scatter(x=[r["block_tokens"] for r in rs], y=[r["efficiency"] for r in rs],
                            mode="lines+markers", name=pid)
        out.append(p.figure(fig, xaxis_type="log", xaxis_title="tokens per KV block",
                            yaxis_title="efficiency"))
    return out


def panel_hierarchy() -> list[Panel]:
    out = []
    h = _load("results/analysis/hierarchy.json")
    p = Panel("hierarchy", "Offload trade-off: throughput at the largest feasible batch",
              ["results/analysis/hierarchy.json", "data/catalog/tiers.csv"],
              "Model-only. Shaded band spans perfect overlap (upper) to full serialization "
              "(lower) of tier and HBM reads. Llama-3.3-70B FP8, 8k input, 1k output.")
    q = Panel("cluster", "CSP instances: Llama-3.3-70B FP8 decode (model prediction)",
              ["data/catalog/csp_instances.csv (AWS, Google Cloud, Azure documentation)",
               "results/analysis/hierarchy.json"],
              "Tensor parallel across all accelerators; 1k input, 1k output; batch capped at 256.")
    if not h:
        out += [p.missing("hierarchy analysis"), q.missing("cluster analysis")]
    else:
        fig = go.Figure()
        for s in h["sweeps"]:
            rs = [r for r in s["rows"] if r.get("batch")]
            name = f"{s['device']} {s['offload']} -> {s['tier']}"
            fig.add_scatter(x=[r["fraction"] for r in rs], y=[r["tok_per_s_upper"] for r in rs],
                            mode="lines+markers", name=name)
            fig.add_scatter(x=[r["fraction"] for r in rs], y=[r["tok_per_s_lower"] for r in rs],
                            mode="lines", line=dict(dash="dot"), showlegend=False)
        out.append(p.figure(fig, xaxis_title="offloaded fraction", yaxis_title="tokens/s"))
        rows = [[c["instance"], c["accelerators"], c["accelerator_id"],
                 f"{c['hbm_total_gb']:.0f}", f"{c['hbm_bw_total_gb_per_s'] / 1000:.1f}",
                 c.get("batch", "-"), f"{c.get('tok_per_s', 0):.0f}" if c["feasible"] else "-",
                 c.get("bound", c.get("reason", ""))] for c in h["cluster"]]
        out.append(q.table(["Instance", "GPUs", "Accelerator", "HBM GB", "HBM TB/s", "Batch",
                            "tok/s", "Bound"], rows))
    return out


def panel_accelerators() -> Panel:
    rows = [[r["name"], r["year"], r["memory_type"], r["memory_gb"], r["mem_bw_gb_per_s"],
             r["bf16_dense_tflops"] or "-", r["fp8_dense_tflops"] or "-",
             f"{r['bf16_dense_tflops'] * 1e12 / (r['mem_bw_gb_per_s'] * 1e9):.0f}"
             if r["bf16_dense_tflops"] else "-"]
            for r in sorted(catalog.load("accelerators").values(), key=lambda r: r["year"])]
    return Panel("accelerators", "Accelerator catalog (dense peaks)",
                 ["data/catalog/accelerators.csv (vendor datasheets; sparsity figures halved)"]
                 ).table(["Accelerator", "Year", "Memory", "GB", "GB/s", "BF16 TFLOPS",
                          "FP8 TFLOPS", "Ridge (FLOP/B)"], rows)


# ------------------------------------------------------------------ page

CSS = """
:root{color-scheme:light;--bg:#fafafa;--fg:#1b1f24;--muted:#5b6470;--card:#fff;--line:#e3e6ea;--accent:#2c5fb3}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px}
h1{margin:0 0 4px;font-size:26px}h2{font-size:18px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;
min-width:0}
.note{color:var(--muted);font-size:13px;margin:0 0 8px}.na{color:var(--muted)}
details{font-size:12px;color:var(--muted);margin-top:8px}
.scroll{overflow-x:auto}.plotly-graph-div{max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;white-space:nowrap}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px}
.badge{border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:13px;
background:var(--card)}
@media (max-width:600px){.grid{grid-template-columns:1fr}}
"""


def _badges() -> str:
    items = []
    for name, rel in (("C1", "results/validation/c1.json"), ("C2", "results/validation/c2.json"),
                      ("I1", "results/validation/i1.json"), ("C4", "results/crosscheck/c4.json")):
        d = _load(rel)
        items.append(f"<span class='badge'>{name}: {html.escape(d['verdict']) if d else 'n/a'}"
                     "</span>")
    return "<div class='badges'>" + "".join(items) + "</div>"


def build(out: Path = OUT) -> Path:
    panels = [*panel_trends(), panel_standards(), panel_roofline(), *panel_validation(),
              *panel_dram(), *panel_hierarchy(), panel_accelerators()]
    body = "".join(p.render() for p in panels)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="darkreader-lock">
<title>HBMPerfModel Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head><body><main>
<h1>HBMPerfModel</h1>
<p class="sub">HBM performance modeling and LLM inference workload analysis. Built {utc_now()}
from committed results; every panel lists its sources.</p>
{_badges()}<div class="grid">{body}</div></main>
<script>window.addEventListener("load", () => document.querySelectorAll(".plotly-graph-div")
  .forEach(d => Plotly.Plots.resize(d)));</script></body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


PANEL_IDS = ["trend-growth", "trend-ridge", "intercepts", "hbm-standards", "roofline", "c1", "c2",
             "i1", "dram-eff", "dram-latency", "kv-block", "hierarchy", "cluster",
             "accelerators"]
