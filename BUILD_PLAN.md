# HBMPerfModel Build Plan

HBMPerfModel: High-Bandwidth Memory Performance Modeling and LLM Workload Analysis.

Plan date: 2026-09-28.
Repository: `hbm-perf-model` (Python package `hbmperf`).
This file is the implementation authority for the repository.

## 1. Product definition

### 1.1 One-sentence claim

HBMPerfModel is a local-first Python tool that turns cited HBM and accelerator specifications into validated first-order predictions of LLM inference bandwidth, latency, and throughput, grounds those predictions in a timing-checked HBM pseudo-channel simulator, and synthesizes public benchmark and industry datasets into trend and architectural-intercept dashboards.

### 1.2 Primary user and triggering problem

Primary user: a memory-system or AI-infrastructure architect (for example, a new-grad HBM architecture engineer at a memory vendor or a performance architect at an accelerator or cloud company).

Triggering problem: questions such as "how much does moving from HBM3 to HBM3E change decode throughput for a 70B model", "at what batch size does decode stop being bandwidth-bound on this part", "what does KV-cache offload to host memory cost", or "in which year does single-device HBM capacity intercept a given model footprint" are usually answered with spreadsheets whose inputs are uncited and whose predictions are never checked against measurement.

### 1.3 Inputs and outputs

Inputs:

- Cited catalogs of accelerators, memory standards, DRAM timing presets, LLM architectures, memory tiers, and cloud instance types (repository-owned CSV files).
- Workload descriptions: model, precision, batch or concurrency, input and output sequence lengths, tensor-parallel degree.
- Public datasets: Epoch AI ML Hardware, SemiAnalysis InferenceX API snapshot, MLCommons MLPerf Inference results.
- Local measurements: C microbenchmarks and `llama-bench` runs on the development machine.

Outputs:

- Per-workload predictions (step time, TTFT, TPOT, tokens per second, bound classification, bandwidth utilization, capacity limits).
- DRAM simulator reports (achieved bandwidth, efficiency, row-buffer statistics, latency percentiles, loaded-latency curves).
- Validation reports with error statistics and `PASS`, `FAIL`, or `INSUFFICIENT_EVIDENCE` verdicts.
- Trend fits and intercept tables with bootstrap intervals.
- A static HTML dashboard and Markdown result tables.

### 1.4 Main workflow

1. `hbmperf catalog list accelerators` to inspect cited inputs.
2. `hbmperf llm predict --model llama-3.3-70b --device nvidia_h200_sxm --precision fp8 --tp 2 --batch 16 --isl 1024 --osl 1024` to get a prediction.
3. `hbmperf dram sim --preset hbm3_6400 --pattern kv_gather --kv-block-bytes 65536` to see how access pattern changes effective bandwidth.
4. `hbmperf hierarchy sweep ...` to evaluate KV-cache or weight offload trade-offs.
5. `hbmperf trends intercepts` to compute generational trends and intercept years.
6. `hbmperf validate {local,inferencex,mlperf}` to see how well the model matches measurement.
7. `hbmperf dashboard build` to produce `docs/dashboard/index.html`.

### 1.5 Owned technical center

1. An event-driven HBM pseudo-channel simulator with an independent command-log timing checker.
2. A first-order LLM inference model with exact byte and FLOP accounting (GQA, tied embeddings, gated MLP, MoE, tensor parallelism, KV capacity).
3. The link between them: simulator-derived access-pattern efficiency (for example KV-cache block size versus row-buffer locality) feeds the first-order model.
4. A validation harness with grouped calibration and holdout splits, clustered bootstrap statistics, and predeclared thresholds.
5. Trend and intercept analysis over cited hardware data.

### 1.6 Claims

Primary claim C1 (local measured, falsifiable): after calibrating two parameters (memory-efficiency factor and fixed per-step overhead) on one model family, the first-order decode model predicts held-out `llama-bench` decode step time (seconds per generated token) for different model families and context depths on the local Apple M2 with median absolute percentage error (MdAPE) at most 20 percent and a bootstrap 95 percent upper bound at most 30 percent, and it beats a calibrated file-size baseline (`t = a * file_tensor_bytes + b`, fitted on the same calibration examples) on the same evaluation examples.

Secondary claim C2 (industry, held-out hardware): calibrated on InferenceX Llama-3.3-70B FP8 results for H100, H200, and MI300X, the model predicts median TPOT for held-out B200, MI325X, and MI355X FP8 configurations with MdAPE at most 35 percent and a bootstrap 95 percent upper bound at most 50 percent, and it beats a calibrated bandwidth-scaled baseline on the same examples (for each evaluation row, the mean TPOT of calibration-hardware rows with the same framework family, TP, concurrency, ISL, and OSL, scaled by the ratio of calibration to evaluation peak bandwidth; rows without a match are excluded from the paired comparison and counted).
C2-FP4 is reported as a separate precision-extrapolation result (B200 NVFP4, MI355X MXFP4) with the same statistics and no threshold claim, because no calibration hardware ran FP4.

Sanity invariant I1 (industry, bound consistency, not a headline claim): the uncalibrated compute upper bound on Llama2-70B Offline output-token throughput computed from cited specs is at least the measured MLPerf Inference closed-division throughput for every mapped available-category row; the report's value is the distribution of measured-over-bound ratios by accelerator and round (achieved fraction of peak), not the pass itself.

Secondary claim C4 (simulator cross-check): on a matched HBM3 configuration and a fixed trace suite, the simulator's achieved bandwidth is within 15 percent and row-buffer hit rate within 0.05 absolute of Ramulator 2 for every pattern in the suite.

Secondary claim C5 (simulator invariants): the simulator passes every analytic golden test and the independent timing checker detects every seeded timing-violation fault while passing its negative controls.

All thresholds above are proposed decision criteria set before any evaluation data is seen.
They are chosen to be meaningfully tighter than the spec-peak baseline step-time error observed in the planning spike (roughly 40 to 70 percent for M2 decode, depending on session) while tolerating shared-machine noise, which the spike showed can shift decode throughput by up to 2x between sessions.
Revisit trigger: none after the evaluation tag; a failing claim is reported as failing.

Prohibited claims:

- No claim that the simulator reproduces any vendor's proprietary HBM controller or JEDEC-certified timing.
- No claim of production accuracy, optimality, or guaranteed throughput on any cloud instance.
- No claim that Apple M2 measurements represent NVIDIA, AMD, or HBM behavior.
- No claim that trend extrapolations are forecasts; they are labeled extrapolations with intervals.
- No claim that hierarchy or offload what-if results are validated; they are model-only analyses.

## 2. User workflows and demonstration path

Five-minute interview walkthrough:

1. Open `docs/dashboard/index.html`: trend panels (bandwidth, compute, capacity growth with doubling times), ridge-point trend, intercept table.
2. Show the validation panel: predicted versus measured scatter for local M2 (C1) and held-out InferenceX hardware (C2), with verdicts and intervals.
3. Run `hbmperf llm predict` for Llama-3.3-70B on H200 versus B200 and explain the bound classification and critical batch size.
4. Run `hbmperf dram sim` for streaming versus KV gather with 1-token and 64-token blocks and show the efficiency difference and loaded-latency curve.
5. Show `hbmperf hierarchy sweep` output for KV offload over NVLink-C2C versus PCIe Gen5 and the resulting TPOT penalty bounds.
6. Show the timing-checker fault-injection test and the Ramulator 2 cross-check table.

## 3. Hiring evidence and public-source provenance

Corpus accounting: collected 2026-09-28; 31 postings plus 2 company boards attempted; 24 postings fully retrieved from official applicant-tracking systems or career sites, all open on the collection date; sampling was purposive by keyword and is not representative; one memory vendor contributes 5 of the 24 postings.
Role categories: memory-vendor HBM or DRAM architecture (6), GPU or accelerator performance modeling (11), memory subsystem or ecosystem (4), system or data-center modeling (3).

Strict explicit-requirement counts across the 24 retrieved postings (not projected to the market):

| Requirement | Count |
| --- | --- |
| Performance modeling | 22/24 |
| Simulation or simulators | 19/24 |
| Python | 15/24 |
| C++ | 15/24 |
| Benchmarks | 12/24 |
| DRAM, DDR, or HBM | 10/24 |
| Memory bandwidth | 8/24 |
| Cost or perf-per-dollar modeling | 7/24 |
| LLM or transformer workloads | 6/24 |
| HBM specifically | 6/24 |
| Trend analysis | 5/24 |
| Visualization or dashboards | 4/24 |
| "CSP" | 3/24 |
| "Architectural intercept" | 2/24 |

The closest public postings (new-college-grad and intern roles at a memory vendor) explicitly ask for first-order HBM bandwidth and latency models, LLM pipeline analysis, trend dashboards using CSP benchmarks, and architectural intercept work; individual postings are not named in this public repository.

| Hiring signal | Project proof artifact |
| --- | --- |
| First-order HBM bandwidth and latency models | `hbmperf.llm`, `hbmperf.dram`, sensitivity sweeps, documented equations |
| Validate models against benchmarks | C1, C2, I1 reports with error statistics and verdicts |
| Simulators and memory subsystem behavior | Timing-checked pseudo-channel simulator, loaded-latency curves, Ramulator 2 cross-check |
| LLM pipeline, KV cache, capacity | Prefill and decode model, KV capacity limits, KV block-size locality study, offload trade-offs |
| Trend dashboards from CSP benchmarks and industry datasets | Static dashboard over Epoch AI, InferenceX, MLPerf, CSP instance catalog |
| Architectural intercept and sensitivity | Intercept tables with bootstrap intervals, parameter sensitivity sweeps |
| C or C++ plus Python | C host microbenchmarks, Python package, Ramulator 2 C++ build and harness |

The mapping is an inference from public postings, not a guarantee of any hiring outcome.
No private or enterprise evidence was used.

## 4. Portfolio gap and differentiation

The author's existing public repositories cover GPU kernels, TensorRT qualification, telemetry, data systems, and ML evaluation, but none covers DRAM or HBM modeling, memory-hierarchy analysis, or hardware trend analysis.

Prior art (dependencies or references, not the claimed contribution):

| Tool | Covers | Lacks relative to this project |
| --- | --- | --- |
| LLM-Viewer (MIT) | Per-layer roofline, web UI | DRAM timing, measured validation, memory tiers, trends |
| llm-analysis (Apache-2.0) | Analytical latency and memory | DRAM timing, offload (listed as TODO), held-out validation |
| GenZ (MIT) | Platform what-ifs, reported validation | Bank-level DRAM behavior, access-pattern efficiency |
| LLMCompass (BSD-3) | Tile-level performance and area | Memory modeled as capacity and bandwidth only |
| Vidur (MIT) | Profile-driven serving simulation | Hardware and DRAM model |
| Calculon (Apache-2.0) | Training with two memory tiers | Inference, DRAM timing |
| Ramulator 2 (MIT), DRAMsim3 (MIT) | Cycle-level DRAM simulation | LLM workload modeling, trends; used here only as a cross-check oracle |

Owned contribution: one cited, tested pipeline that connects HBM pseudo-channel timing behavior to LLM inference predictions, validates those predictions against local measurement and public industry benchmarks on held-out hardware, and turns cited hardware data into intercept analysis.
Name check: `hbm-perf-model` on GitHub matches only this repository; `hbmperf` is free on PyPI (not published by this plan).

## 5. Scope

### 5.1 Milestone 0 feasibility (already partly proven by the planning spike)

Observed during planning on the local M2 (8 GB):

- A C pointer-chase benchmark produced roughly 2.7 ns, 10 ns, 26 ns, and 124 ns for 32 KiB, 1 MiB, 16 MiB, and 256 MiB working sets.
- A multithreaded C read benchmark reached about 21, 41, 54, and 65 GB/s at 1, 2, 4, and 8 threads.
- Homebrew `llama.cpp` (formula version 0.5.0, MIT) ran `llama-bench` on Qwen2.5-0.5B-Instruct GGUF Q8_0 and F16 (Apache-2.0): decode 55.7 and 32.7 tokens per second with high run-to-run spread, supporting a bandwidth-bound decode.
- Ramulator 2 at commit `72427a1bba3771564c4fb0e494ba02242fd1eaa7` built natively after one Apple-clang fix (`config[name].template as<T>()` in `src/ramulator/base/param.h`) and ran HBM3 traces through its Python binding under Homebrew Python 3.14.
- Epoch AI CSV, InferenceX API, and MLPerf `summary_results.json` files downloaded and parsed.

These observations are planning evidence only; Milestone 0 must reproduce them through repository code.

### 5.2 Portfolio-ready core (Milestones 0 to 3)

Catalogs with provenance, the DRAM simulator with timing checker and golden tests, the first-order LLM model, local measurement harness, and the C1 evaluation with a README results section.

### 5.3 Extended V1 (Milestones 4 to 9)

InferenceX and MLPerf validation (C2, I1), memory hierarchy and cluster analysis, KV block-size locality study, trends and intercepts, dashboard, Ramulator 2 cross-check (C4), documentation and clean-checkout release audit.

### 5.4 Follow-on (not V1, never a V1 gate)

- Follow-on: energy-per-token and cost-per-token modules (require cited pJ/bit and price sources).
- Follow-on: MoE validation against InferenceX gpt-oss-120b and DeepSeek results.
- Follow-on: GPU measurement on Colab or university HPC.
- Follow-on: HBM4 and LPDDR5 simulator presets validated against an oracle.
- Follow-on: publishing to PyPI or enabling GitHub Pages.

### 5.5 Non-goals

- No cycle-accurate GPU core model, cache hierarchy model, or on-chip interconnect model.
- No serving scheduler simulation (continuous batching dynamics beyond steady-state concurrency).
- No web server; the dashboard is a static file.
- No proprietary data, JEDEC paywalled documents, or vendor NDA material.
- No training workloads.

## 6. System architecture

| Component | Module | Language | Inputs | Outputs | Failure modes |
| --- | --- | --- | --- | --- | --- |
| Provenance | `hbmperf/provenance.py` | Python | files, configs | SHA-256 manifests, run directories, run metadata | hash mismatch raises `ProvenanceError`; path escape refused |
| GGUF reader | `hbmperf/gguf.py` | Python | GGUF file header (untrusted) | tensor inventory with exact stored bytes | malformed or oversized header raises `GGUFError` |
| Catalog | `hbmperf/catalog.py` | Python | `data/catalog/*.csv` | typed dataclasses | schema violation raises `CatalogError` with file and line |
| DRAM simulator | `hbmperf/dram/` | Python | timing preset, traffic, controller config | `SimResult`, optional command log | timing checker violation raises `TimingViolation` in strict mode |
| LLM model | `hbmperf/llm.py` | Python | model config, device, precision, workload | `Prediction` | capacity overflow returns `feasible=False` with reason |
| Hierarchy and cluster | `hbmperf/hierarchy.py` | Python | tiers, placement, instance catalog | `TierResult`, cluster tables | infeasible placement returns reason |
| Trends | `hbmperf/trends.py` | Python | Epoch CSV, catalog | fits, intercepts with intervals | fewer than minimum points returns `INSUFFICIENT_EVIDENCE` |
| Validation | `hbmperf/validate/` | Python | measurements, snapshots | reports with verdicts | missing data or stale model hash yields `INSUFFICIENT_EVIDENCE` or `STALE` |
| Host benchmarks | `bench/host/*.c` | C | working-set sizes, threads | JSON lines | compile failure reported by CLI |
| llama.cpp harness | `hbmperf/measure.py` | Python | model manifest | measurement JSON | missing `llama-bench` or model file yields `BLOCKED` |
| Ramulator 2 harness | `tools/ramulator_crosscheck/` | Shell, Python, C++ build | pinned commit, patch, traces | cross-check CSV | build failure yields `BLOCKED` report |
| Dashboard | `hbmperf/dashboard.py` | Python, Plotly | result JSON and CSV | `docs/dashboard/index.html` | missing input panel shows explicit "not available" |
| CLI | `hbmperf/cli.py` | Python (argparse) | arguments | text, JSON | nonzero exit code on error |

Data flow: catalogs feed both the simulator presets and the LLM model; the simulator produces pattern efficiencies stored in `results/dram/efficiency.json`, which the LLM model optionally consumes via `--efficiency-source sim`; validation consumes measurements and snapshots; trends consume Epoch and catalog data; the dashboard consumes committed results only.

## 7. Data, schema, unit, and identity contracts

### 7.1 Units

Every numeric field name carries its unit suffix, and no field permits another unit:

- `_gb_per_s`: decimal gigabytes per second (1e9 bytes per second).
- `_gbit_per_s`: gigabits per second per pin.
- `_gb`: decimal gigabytes (1e9 bytes); `_gib`: binary gibibytes (2^30 bytes), used only when a source publishes GiB, converted on load.
- `_tflops`: 1e12 floating-point operations per second, dense (no structured sparsity).
- `_ck`: DRAM command-clock cycles; `_ps`: picoseconds; `_ns`: nanoseconds; `_s`: seconds.
- `_bits`: bits per value (may be fractional, for example 8.5 for GGUF Q8_0).
- `_w`: watts.

Simulator time is an integer count of command-clock cycles; conversion to nanoseconds uses the preset's `tck_ps`.
LLM model time is floating-point seconds.

### 7.2 Catalog schemas (`data/catalog/`)

All catalogs are UTF-8 CSV with a header, one row per entity, a stable lowercase snake-case `id`, a `source_url`, and a `notes` column.
`schema_version` is recorded in `data/catalog/SCHEMA.md` (starts at `1`).

- `accelerators.csv`: `id,vendor,name,year,memory_type,memory_gb,mem_bw_gb_per_s,hbm_stacks,bf16_dense_tflops,fp8_dense_tflops,fp4_dense_tflops,int8_dense_tops,scaleup_bw_gb_per_s,tdp_w,source_url,notes`.
- `memory_standards.csv`: `id,standard,year,interface_bits_per_stack,max_pin_rate_gbit_per_s,peak_bw_per_stack_gb_per_s,channels_per_stack,pseudo_channels_per_stack,max_stack_high,max_capacity_gb_per_stack,source_url,notes`.
- `dram_timing.csv`: `id,standard,data_rate_mbit_per_s,tck_ps,pc_width_bits,burst_length,burst_bytes,sids,bank_groups,banks_per_group,rows,row_bytes,pcs_per_stack,n_cl_ck,n_cwl_ck,n_rcd_rd_ck,n_rcd_wr_ck,n_rp_ck,n_ras_ck,n_wr_ck,n_rtp_ck,n_ccd_s_ck,n_ccd_l_ck,n_ccd_r_ck,n_rrd_s_ck,n_rrd_l_ck,n_faw_ck,n_wtr_s_ck,n_wtr_l_ck,n_rtw_ck,n_bl_ck,n_rfc_ck,n_refi_ck,source_url,source_ref,notes`.
- `models.csv`: `id,family,n_layers,d_model,n_heads,n_kv_heads,head_dim,d_ff,vocab,tied_embeddings,mlp_type,n_experts,top_k,n_shared_experts,d_ff_expert,published_params,source_url,notes`.
- `tiers.csv`: `id,name,link,peak_bw_gb_per_s,measured_bw_gb_per_s,latency_ns,capacity_gb,source_url,notes`.
- `csp_instances.csv`: `id,csp,instance,accelerator_id,accelerator_count,host_mem_gb,scaleout_gbit_per_s,source_url,notes`; instance accelerator memory is derived from `accelerator_count` and the accelerator row, and the provider's published figure and unit (GB or GiB) are preserved verbatim in `notes`.
- `accelerators.csv` `scaleup_bw_gb_per_s` is the vendor-published aggregate scale-up bandwidth per device (bidirectional where the vendor states it); the model uses half of it as the per-direction ring bandwidth, and the resulting error is absorbed by the calibrated `alpha` and documented in LIMITATIONS.
- `models.csv` also carries `qkv_bias` (true for Qwen2.5).
- The `apple_m2` row records the measured 8 GB configuration, with the published 24 GB maximum in `notes`.

Validation rules: required fields non-empty; numeric fields parse and are positive; blanks mean "not published" and load as `None`; `source_url` starts with `https://`; `accelerator_id` references must resolve; duplicate `id` is an error.
Timing presets from Ramulator 2 labeled "Ramulator Guesstimate" in source must carry `notes=guesstimate`; JEDEC-formula-derived values carry the formula reference.

### 7.3 Measurement and snapshot records

Every run writes `results/<kind>/<run_id>/` where `run_id = <UTC timestamp YYYYmmddTHHMMSSZ>-<8 hex of config hash>`; existing directories are never overwritten.
Each run directory contains `meta.json` with: `schema_version`, `hbmperf_version`, `git_sha`, `git_dirty`, `python_version`, `platform`, `cpu_brand`, `mem_bytes`, `tool_versions`, `seed`, `config`, `config_sha256`, `input_manifest` (path to SHA-256 map), `llm_model_sha256` (hash of `src/hbmperf/llm.py`), `claim_input_sha256` (for validation runs: hash of the modules the claim uses, `llm.py`, `gguf.py`, `stats.py`, `validate/__init__.py`, and the claim's `validate/*.py` module, plus the canonical JSON of the computational fields of the catalog rows used (excluding `notes` and `source_url`) and the SHA-256 of every committed measurement or snapshot file the claim reads), and `started_utc` and `finished_utc` in ISO-8601 UTC.

External snapshots are stored under `data/external/<source>/<YYYY-MM-DD>/` (ignored by Git) with a committed manifest `data/manifests/<source>-<YYYY-MM-DD>.json` containing URL, retrieval UTC time, byte size, SHA-256, and license URL.
Loading a snapshot verifies the SHA-256; a mismatch raises `ProvenanceError`.

Null semantics: missing published value is `None`; measurement failure is a record with `status` in `{ok, failed, invalid}` and a `reason`; invalid measurements are never averaged into results.

Deterministic serialization: JSON written with sorted keys, two-space indent, and a trailing newline; CSV rows sorted by `id` or by declared key columns.

## 8. Algorithms and models

### 8.1 HBM pseudo-channel simulator (`hbmperf/dram/`)

State per pseudo-channel: for each bank (bank group index, bank index) the open row or `None` and the earliest cycle for ACT, PRE, RD, and WR; per bank group the last RD and WR cycles; per pseudo-channel the last ACT cycle per bank group, a deque of the last four ACT cycles (tFAW), the data-bus busy-until cycle, the last write-data end cycle (tWTR), refresh due and refresh-busy-until cycles.

Address mapping: the simulator models one pseudo-channel, so traffic generators emit pseudo-channel-local byte addresses; an address is split, above the `burst_bytes` offset bits, into fields in a configurable most-significant-to-least-significant order.
Default `RoCoSiBaBg` (bank group in the lowest field) so consecutive bursts alternate bank groups and are limited by `tCCD_S`; `RoSiBaBgCo` (column lowest, consecutive bursts in one bank) is supported to show the `tCCD_L` penalty.

Controller: per pseudo-channel read queue and write queue, each with capacity `queue_depth` (default 32, matching the Ramulator 2 HBM3 controller's read and write buffers); each cycle the controller may issue one column command and one row command (HBM's separate column and row command buses): the column command is the oldest row-hit RD or WR in the active queue that is legal now; the row command comes from scanning requests oldest first (active queue, then the other queue) and issuing the first ACT (bank closed) or PRE (another row open) that is legal now, except that a bank whose open row still has a queued hit in the active queue is never precharged (open-row protection); ties broken by arrival order; open-page policy by default, closed-page (auto-precharge) optional; write drain with high and low watermarks (0.8 and 0.2) as in the Ramulator 2 HBM3 controller.

Refresh: all-bank refresh every `n_refi_ck`; the controller precharges all banks and blocks the pseudo-channel for `n_rfc_ck`.

Timing constraints enforced: tRCD, tRP, tRAS, tRC (= tRAS + tRP), tCL, tCWL, tBL, tCCD_S and tCCD_L, tRRD_S and tRRD_L, tFAW, tWR, tRTP, tWTR_S and tWTR_L, tRTW, tRFC, tREFI, and data-bus non-overlap.

Simplifications (documented): independent command buses per pseudo-channel; no command-bus bandwidth limit; no per-bank refresh; no ECC; no power-down.

Traffic generators (seeded `numpy.random.Generator(PCG64(seed))`):

- `stream`: sequential reads over a region.
- `random`: uniform random `burst_bytes`-aligned reads.
- `strided`: fixed stride.
- `mix`: read and write mix with a given write fraction.
- `kv_gather`: pages of `kv_block_tokens * kv_bytes_per_token_per_layer` contiguous bytes, pages placed randomly, read in page order.
- `llm_decode`: a byte-weighted mix of `stream` (weights) and `kv_gather` (KV cache) derived from an `hbmperf.llm` prediction.

Traffic modes: closed-loop (fixed outstanding requests, used for peak achievable bandwidth) and open-loop (Poisson or fixed-rate arrivals, used for loaded-latency curves).

Outputs: served bytes, elapsed cycles, achieved bandwidth in `_gb_per_s`, efficiency (achieved over peak), row hits, misses, and conflicts, read latency mean and p50, p95, and p99 in ns, mean queue occupancy.

Stack or device scaling: device bandwidth = pseudo-channel bandwidth times pseudo-channel count; this assumes uniform interleaving and independent pseudo-channels (documented ceiling).

Complexity: O(requests x queue_depth) scheduling work; memory O(queue_depth + banks).
Target: at least 20,000 requests per second of wall time on the M2 so a 200,000-request run finishes in under 10 seconds (performance target, not a claim).

Independent timing checker (`hbmperf/dram/checker.py`): replays a command log `(cycle, cmd, bg, bank, row)` and verifies every constraint pair from the preset without sharing code with the scheduler; any violation reports the constraint name, the two commands, and the observed and required gap.

### 8.2 First-order LLM inference model (`hbmperf/llm.py`)

Model parameters from `models.csv`.
Per-layer weights: attention Q, K, V, O projections `d_model*(n_heads*head_dim) + 2*d_model*(n_kv_heads*head_dim) + (n_heads*head_dim)*d_model`; gated MLP `3*d_model*d_ff` (dense) or MoE with `n_experts` experts of `3*d_model*d_ff_expert` plus router `d_model*n_experts` and shared experts; norms `2*d_model`.
Embeddings: `vocab*d_model`; LM head separate unless `tied_embeddings`.
Total parameter count is compared to `published_params` (test tolerance 2 percent).

Bytes read per decode step for batch `B` with per-sequence context lengths `L_i`:

- Weights: all layer weights plus LM head (the embedding table contributes only `B` rows when untied) times `weight_bits/8`; for MoE, expected unique experts touched per layer `E*(1-(1-k/E)^B)` times expert bytes.
- KV read: `sum_i L_i * kv_bytes_per_token` where `kv_bytes_per_token = 2*n_layers*n_kv_heads*head_dim*kv_bits/8`.
- KV write: `B * kv_bytes_per_token`.
- Activations: ignored in V1 and reported as a documented approximation.

FLOPs per decode step: `2 * active_params_excluding_embedding_lookup * B + sum_i 4 * n_layers * n_heads * head_dim * L_i`.
Prefill for `S` prompt tokens per sequence: `2 * layer_matmul_params * B * S + 2 * lm_head_params * B + 2 * n_layers * n_heads * head_dim * B * S^2` (causal; the LM head runs only on the last prompt token), bytes are weights plus KV write.
Precisions (bits per weight, KV bits, peak column): `bf16` and `f16` (16, 16, bf16), `q8_0` (8.5, 16, bf16), `fp8` (8, 8, fp8), `nvfp4` (4.5, 8, fp4), `mxfp4` (4.25, 8, fp4); `kv_bits` can be overridden per call.

Time per step: `t = max(F/(P*eta_c), Bytes/(W*eta_m)) + t_fixed + t_comm` where `P` is the precision-matched dense peak, `W` peak memory bandwidth, `eta_c` and `eta_m` efficiencies (default 1.0 for the spec-peak baseline).
Ablations: `sum` combining (`F/(P*eta_c) + Bytes/(W*eta_m)`), weights-only bytes, and no-overhead.
Tensor parallelism with degree `p`: weights and KV divided by `p`; two all-reduces per layer of `B*d_model*act_bits/8` bytes with ring time `2*(p-1)/p * msg/link_bw + 2*(p-1)*alpha` (alpha default 5 microseconds, a labeled assumption, calibrated in C2).
Capacity: `weights/p + KV(B, L)/p + reserve <= memory_gb * usable_fraction` (default 0.9); infeasible predictions return `feasible=False`.
Derived outputs: TPOT, TTFT, tokens per second per device and per sequence, arithmetic intensity, ridge point `P/W`, bound class (`memory`, `compute`), and critical batch `B*` where compute time equals memory time at a given context.

Calibration: `eta_m` and `t_fixed` are fitted by ordinary least squares of measured step time on predicted memory-time-at-unit-efficiency, which is a closed-form two-parameter linear fit; C2 additionally fits `eta_c` and `alpha` by bounded least squares (`scipy` is not used; a small deterministic grid search over bounded ranges followed by least squares on the remaining linear parameters).

### 8.3 Hierarchy and cluster analysis (`hbmperf/hierarchy.py`)

Tiered placement: fraction `f` of KV cache (or weights) resides in tier 2 with bandwidth `W2`; per-step memory time lower bound (perfect overlap) `max(B1/W1, B2/W2)` and upper bound (serialized) `B1/W1 + B2/W2`; the report shows both bounds and the capacity gained.
Sweeps: offload fraction, context length, and batch; outputs the maximum feasible batch and the TPOT penalty interval; tiers from `tiers.csv` (NVLink-C2C with the published peak and a measured fraction from arXiv 2408.11556, PCIe Gen5 x16, host DDR5).
Cluster view: for each CSP instance, aggregate HBM capacity and bandwidth, largest model and context that fits, predicted Llama-3.3-70B decode throughput at TP equal to the accelerator count.

### 8.4 Trends and intercepts (`hbmperf/trends.py`)

Filter Epoch AI rows to datacenter accelerators with non-null values for each metric.
Fit `log2(metric) = a + b*(year - 2016)` by ordinary least squares on release-date decimal years; doubling time `1/b`.
Uncertainty: nonparametric bootstrap over devices, 2000 resamples, seed `20260928`, percentile 95 percent intervals; fits require at least 8 devices, else `INSUFFICIENT_EVIDENCE`.
Intercepts:

1. Ridge-point trend (dense BF16 FLOP per byte of bandwidth) and its growth rate.
2. Critical decode batch `B*` for Llama-3.3-70B FP8 per device, plotted by year.
3. Capacity intercept: year at which the fitted capacity trend crosses the footprint of a named model plus KV for a named context and batch.
4. Bandwidth-requirement intercept: year at which the fitted bandwidth trend meets the bandwidth needed for a target TPOT for a named model and batch.

Intercept years inherit bootstrap intervals by recomputing the crossing inside each resample; a crossing outside the range 2010 to 2040 is reported as "no intercept within range".

### 8.5 Statistical decision rules

Common to C1 and C2:

- Error per example: absolute percentage error of predicted versus measured (decode step time for C1, median TPOT for C2).
- Estimand: median APE over examples, with examples grouped into clusters (C1: one model file; C2: hardware, framework, precision, and TP configuration).
- Coverage caveat: a percentile bootstrap over 6 to 16 clusters can under-cover; reports also give the leave-one-cluster-out range of MdAPE.
- Interval: cluster bootstrap, 2000 resamples, seed `20260928`, statistic recomputed inside each resample, percentile 95 percent interval.
- Paired comparison: in each resample, the difference in MdAPE between baseline and candidate on identical examples; candidate "beats" baseline when the lower bound of the difference is above zero.
- Minimum support: C1 at least 4 evaluation clusters and 24 evaluation examples; C2 at least 6 evaluation clusters and 60 examples; below either, the verdict is `INSUFFICIENT_EVIDENCE`.
- `PASS`: point MdAPE at or below the threshold, upper bound at or below the upper threshold, and candidate beats baseline; `FAIL` otherwise with support met.

I1: every mapped row must satisfy bound >= measured (any violation is a `FAIL` that indicates a catalog or model error and is investigated); the report gives the measured-over-bound ratio distribution per accelerator and round; with fewer than 30 mapped rows the report is `INSUFFICIENT_EVIDENCE`.
The bound uses output-token compute only (`P / (2 * active_params)` per device, with `P` the dense peak for the row's weight precision from `weight_data_types`, FP8 when absent or unparseable), which omits input-token compute and is therefore a looser, still valid upper bound.
C4: per-pattern tolerances as in section 1.6; any pattern outside tolerance yields `FAIL` for C4 with the pattern named.

## 9. Repository layout

```text
hbm-perf-model/
  BUILD_PLAN.md                  committed, this plan
  README.md                      committed
  LICENSE                        committed, MIT
  pyproject.toml                 committed
  uv.lock                        committed
  .gitignore                     committed
  src/hbmperf/
    __init__.py
    cli.py
    gguf.py
    provenance.py
    catalog.py
    llm.py
    hierarchy.py
    trends.py
    measure.py
    fetch.py
    dashboard.py
    stats.py
    dram/
      __init__.py
      timing.py                  presets from dram_timing.csv
      mapping.py
      traffic.py
      sim.py
      checker.py
    validate/
      __init__.py
      local.py                   C1
      inferencex.py              C2
      mlperf.py                  I1
  bench/host/
    latency.c
    bandwidth.c
    Makefile
  tools/ramulator_crosscheck/
    build.sh                     pinned commit, applies patch, builds into .cache/
    param_template.patch
    run.py
  data/
    catalog/*.csv, SOURCES.md, SCHEMA.md     committed
    manifests/*.json                         committed
    measurements/                            committed small JSON measurement records
    derived/*.csv                            committed small derived tables with attribution
    external/                                ignored raw downloads
  results/                                   committed small evidence (JSON, CSV, Markdown)
  docs/
    METHODOLOGY.md
    RESULTS.md
    LIMITATIONS.md
    dashboard/index.html                     generated, committed
  tests/
    fixtures/                                synthetic, repository-owned
    test_*.py
```

Ignored: `data/external/`, `.cache/`, `*.gguf`, `bench/host/build/`, `.venv/`.
Model weights live outside the repository in `~/.cache/hbmperf/models` (path configurable by `HBMPERF_MODEL_DIR`); only their manifests are committed.

## 10. Toolchain and dependencies

- Python 3.12 managed by `uv`; `requires-python = ">=3.11"` (uses `tomllib` availability and modern typing).
- Runtime dependencies: `numpy`, `plotly`.
- Development dependencies: `pytest`, `ruff`.
- Standard library for CSV, JSON, argparse, hashlib, subprocess, urllib.
- C compiler: system `clang` (`-O2`, pthreads).
- External tools, optional and probed: `llama-bench` from Homebrew `llama.cpp` (C1), Homebrew Python 3.14 plus CMake for the Ramulator 2 cross-check (C4).
- Version recording: `meta.json` stores tool versions; `llama-bench` build info from its JSON output.

## 11. Data acquisition, licensing, manifests, and splits

| Source | Use | License | Redistribution plan |
| --- | --- | --- | --- |
| Vendor datasheets and JEDEC press releases | Catalog values | Facts with citation | Values and URLs committed |
| Ramulator 2 and DRAMsim3 presets | Timing presets | MIT | Values with file and line reference committed |
| Epoch AI ML Hardware CSV `https://epoch.ai/data/ml_hardware.csv` | Trends | Creative Commons Attribution | Raw CSV in ignored snapshot; derived fits committed with attribution |
| InferenceX API `https://inferencex.semianalysis.com/api/v1/benchmarks?model=Llama-3.3-70B-Instruct-FP8` | C2 | Repository Apache-2.0; API has no explicit data license | Raw snapshot ignored; only derived residual tables and aggregate statistics committed with attribution |
| MLPerf Inference `summary_results.json` for v5.0, v5.1, v6.0 | I1 | Apache-2.0 (v5.0 repository lacks a license file; its rows are used only for I1 with that caveat noted) | Small filtered table committed with attribution |
| Official CSP documentation pages | Instance catalog | Facts with citation | Values and URLs committed |
| Hugging Face GGUF models (Apache-2.0 or MIT for both the base model and the quantizer repository) | C1 measurement | Per model | Never committed; manifest with SHA-256 committed |

C1 split (grouped by model family, fixed now):

- Calibration family: Qwen2.5 from `Qwen/Qwen2.5-0.5B-Instruct-GGUF` and `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, Q8_0 and F16 (4 files).
- Evaluation families: SmolLM2 360M and 1.7B Instruct from `bartowski/SmolLM2-360M-Instruct-GGUF` and `bartowski/SmolLM2-1.7B-Instruct-GGUF` (Q8_0 and F16), and TinyLlama 1.1B Chat from `andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF` (Q8_0 and F16, Apache-2.0); if a listed file is unavailable, another Apache-2.0 or MIT family with a public `config.json` is chosen in Milestone 0 before any measurement.
- Examples per file: context depths `{0, 256, 512, 1024, 2048, 4096}` restricted to `depth + 128 <= max_position_embeddings`, generation length 128, 5 repetitions; Qwen2.5 and SmolLM2 give 6 depths per file and TinyLlama gives 4, so evaluation has 6 files and 32 examples (8 above the support floor).
- Committed C1 measurement records also store, per file, the GGUF total tensor bytes, decode-read bytes, and guard results, so `hbmperf validate local` reproduces the verdict from a clean checkout without model files.
- Model files larger than 4.0e9 bytes are excluded; the largest planned file (Qwen2.5-1.5B F16, 3,560,416,288 bytes) fits the M2's roughly 5.7 GB Metal working set, and an out-of-memory failure marks that configuration `invalid`.
- Milestone 0 freezes `data/manifests/c1_models.json` with repository, filename, SHA-256, license, and split; evaluation files are not benchmarked until the evaluation tag exists.

C2 split (grouped by hardware, fixed now): calibration hardware `h100`, `h200`, `mi300x` (FP8 rows); evaluation hardware `b200`, `mi325x`, `mi355x`; single snapshot date recorded in the manifest; all ISL and OSL pairs and concurrencies included; single-node, non-disaggregated rows only.
Precision handling: `fp8` rows use FP8 weights and FP8 peak; `fp4` rows map to `nvfp4` on NVIDIA and `mxfp4` on AMD with the FP4 dense peak from the catalog, else the row is excluded with a logged reason; KV cache is assumed FP8 (a labeled assumption with a sensitivity slice at 16 bits).
Slices: ISL and OSL pair (8192-token inputs are expected to show prefill interference), concurrency at most 16 versus above 16, framework.

I1 set: llama2-70b (`-99` and `-99.9` deduplicated) Offline, closed division, available category, rounds v5.0, v5.1, and v6.0 (the rounds with a machine-readable summary file), rows whose accelerator maps to a catalog id; unmapped rows are listed and excluded; rows whose submitter is a cloud service provider are tagged so the dashboard can show CSP-submitted results separately.
"CSP benchmarks" in this project means exactly two things: MLPerf results submitted by cloud providers (tagged as above) and the official CSP instance catalog; InferenceX is an independent benchmark and is not labeled as CSP data.

## 12. Testing, fault injection, security, and reproducibility

### 12.1 Tests

- Unit: unit conversions, catalog parsing and rejection of bad rows (synthetic fixtures), address mapping round trip, parameter counts versus published values, KV bytes per token for known models.
- Golden (simulator): isolated read latency for row hit `tCL + tBL`, row miss `tRCD + tCL + tBL`, row conflict `tRP + tRCD + tCL + tBL` in cycles; single-bank random throughput equals one access per `tRC` within 1 percent (valid because `tRC >= tRCD + tRTP + tRP` for every preset); streaming efficiency without refresh at least 0.95 under the default bank-group-interleaved mapping with a 64-entry queue for every catalog preset with `sids = 2` (all Milestone 1 presets; the C4-only preset `hbm3_6400_4hi` has one stack ID and is excluded; the 32-entry and 1-stack-ID sensitivities are reported, not gated); single-bank streaming (every burst of consecutive rows of one bank under the column-lowest mapping) at most `tBL/tCCD_L + 0.02` with row-hit rate above 0.9; streaming efficiency with refresh within 0.02 of `1 - (tRFC + tRP)/tREFI` times the no-refresh efficiency; closed-page random reads (one ACT per read) never exceed `4 * burst_bytes / tFAW` bytes per cycle; every command log contains at most 4 ACTs in any `tFAW` window.
- Property: achieved bandwidth never exceeds peak; latency never below the row-hit minimum; determinism (same seed gives byte-identical result JSON); Little's law in closed-loop within 2 percent.
- Metamorphic: halving `tck_ps` with identical cycle counts doubles streaming bandwidth within 1 percent; doubling batch in the LLM model never decreases step time; doubling bandwidth never increases memory-bound step time.
- Integration: CLI commands on synthetic fixtures produce valid JSON; dashboard build on fixture results produces an HTML file containing every panel id.
- Real path: `hbmperf measure host`, `hbmperf measure llamacpp` on the frozen manifest, `hbmperf validate local`.

### 12.2 Fault injection with negative controls

- Timing checker faults: scheduler variants with one constraint disabled (`tFAW`, `tRCD`, `tRP`, `tCCD_L`, `tWTR`, `tRFC`) run through a seeded trace that exercises the constraint; the checker must flag the named constraint; the unmodified scheduler on the same trace must pass (negative control).
- Catalog faults: a fixture with a GiB value in a `_gb` column tagged as GiB in notes, a duplicate id, a dangling `accelerator_id`, and a non-HTTPS source each fail with the expected error code; a clean fixture passes.
- Provenance faults: a snapshot with one byte changed fails hash verification; the original passes.
- Validation faults: a synthetic measurement set with known ground truth yields the known MdAPE; a support-deficient set yields `INSUFFICIENT_EVIDENCE`; a result file whose `claim_input_sha256` differs from the recomputed value (after editing a used module, a used catalog row, or a committed measurement record) yields `STALE`, while adding an unrelated catalog row does not.

### 12.3 Security and privacy

- Trust boundaries: downloaded JSON, CSV, and GGUF files; `llama-bench` output.
- Downloads use HTTPS only, a size cap (300 MB per data file, 4.0e9 bytes per GGUF file), a timeout, and SHA-256 verification against the manifest after first retrieval.
- No archive extraction is performed.
- Subprocesses run with argument lists (no shell) and a timeout.
- Output paths are resolved and must stay under the repository `results/` or `data/` directories; no-clobber directory creation.
- No network services, no credentials, no secrets; no personal paths or hostnames in committed files (checked by a test that scans committed text files for `/Users/` and `/home/` patterns).
- Dependency audit: `uv pip list` recorded; no install scripts beyond `uv sync`.

### 12.4 Reproducibility

- `uv.lock` pins dependencies.
- All random operations take explicit seeds (default `20260928`).
- Every run records `meta.json` as in section 7.3.
- `make`-free rerun commands are documented in `docs/METHODOLOGY.md`.

## 13. Evaluation and benchmark protocol

### 13.1 Local measurement protocol (C1)

- Hardware: the local Apple M2 (8 GB) recorded in `meta.json`; results are M2-only evidence.
- Before measurement: no other benchmark or agent workload runs; record the 1-minute load average and the CPU percent used by processes other than llama-bench; if other processes use more than 200 percent CPU (two of eight cores), wait up to 60 seconds in 10-second steps, then mark the configuration `invalid` (amendment A1 in section 21).
- `llama-bench -m <file> -p 0 -n 128 -d <depth> -r 5 -o json` with default Metal offload and llama-bench's built-in warm-up run; all 5 entries of the JSON `samples_ns` array are used and the array must be present; step time is the median of `samples_ns / 128`.
- Session anchor: every measurement session begins and ends with the anchor configuration (Qwen2.5-0.5B Q8_0, depth 0); the evaluation session is `invalid` if its anchor median differs by more than 10 percent from the calibration-session anchor, in which case the whole evaluation session is re-run once and then reported as `INSUFFICIENT_EVIDENCE` if it is still out of tolerance.
- Session comparability (predeclared): calibration and evaluation sessions run on AC power with no agent, benchmark, or build workload, after closing user applications, with the process count and load average recorded; the evaluation session starts within 2 hours of the end of the calibration session (the fit and tag happen in between). An evaluation session that starts outside this window is `invalid` and consumes the single allowed re-run; the single re-run (whether triggered by the anchor or by the window) must also start within 2 hours of the end of the calibration session, otherwise the C1 verdict is `INSUFFICIENT_EVIDENCE`. Evaluation model files are downloaded and hash-verified, without benchmarking, before the calibration session so downloads do not consume the window.
- Drift-normalized secondary analysis (predeclared, non-headline): evaluation step times multiplied by the ratio of calibration-anchor to evaluation-anchor median step time, reported alongside the headline result whatever the anchor verdict.
- Committed measurement records keep only the model file basename and SHA-256, never absolute paths.
- A configuration is `invalid` if the coefficient of variation across repetitions exceeds 15 percent after one automatic retry.
- Order: randomized with seed `20260928` across files and depths, to spread drift.
- Prediction inputs: model config from `models.csv`, `weight_bits` from the quantization type (F16 16.0, Q8_0 8.5), `kv_bits` 16, device `apple_m2`, batch 1, context `depth + 64` (mean position during 128-token generation).
- Error metric: absolute percentage error of predicted versus measured decode step time.
- Baselines: the calibrated file-size baseline `t = a * file_tensor_bytes + b` (the comparison that decides "beats baseline") and the spec-peak model (`eta_m = 1`, `t_fixed = 0`, reported for context).
- Ablations reported: weights-only bytes, no fixed overhead.
- Correctness guards, both required before C1 runs: (a) `hbmperf.gguf` total stored tensor bytes equal the `model_size` reported by `llama-bench` exactly; (b) the model's predicted batch-1 decode weight bytes are within 1 percent of the GGUF decode-read bytes (all tensors except the token-embedding table when a separate `output.weight` exists), which handles files that duplicate tied embeddings.

### 13.2 Industry protocols (C2, I1)

Described in sections 8.5 and 11; slices reported for C2: concurrency at most 16 versus above 16, framework, precision, ISL and OSL pair.

### 13.3 Simulator performance

Wall-time throughput of the simulator is measured with 3 repetitions on a 200,000-request stream trace and reported; this is engineering telemetry, not a claim.

### 13.4 Final holdout discipline

- C1: calibration measurements are taken first; the LLM model code and calibration are then frozen by the Git tag `eval-c1-v1`; evaluation files (downloaded and hash-verified earlier without benchmarking) are measured once at that tag; the report records the tag SHA and module hash; `src/hbmperf/llm.py` is feature-complete at Milestone 2 (all precisions, knobs, prefill, TP, MoE, capacity) so later milestones do not modify it; C2 calibration lives in `validate/inferencex.py`; `hbmperf validate` recomputes `claim_input_sha256` and reports `STALE` when it differs from the recorded value, so adding unrelated catalog rows does not stale a claim but editing a used module or row does; a `STALE` C1 requires a new, untouched evaluation family.
Therefore every module in the C1 hash (`llm.py`, `gguf.py`, `stats.py`, `validate/local.py`, `validate/__init__.py`) and the C1 catalog rows are frozen at `eval-c1-v1`; these files stay byte-identical after `eval-c1-v1`, later claims add new modules only (`validate/__init__.py` holds only hashing and staleness helpers, never a claim registry; command dispatch lives in `cli.py`), and the hash covers only computational catalog fields (it excludes `notes` and `source_url`), so documentation edits to a row do not stale a claim.
- C2: calibration uses only calibration hardware rows; the evaluation report is generated once at tag `eval-c2-v1`; same staleness rule.
- Fallback claim tree for C1 and C2, evaluated from the same run: (a) full claim; (b) "calibrated model beats the claim's own predeclared calibrated baseline" (C1: file-size baseline; C2: bandwidth-scaled calibration baseline) if the threshold fails but the paired comparison passes; (c) "measured error characterization" with the observed MdAPE and interval if both fail.

## 14. CLI, reports, and visualization

```text
hbmperf catalog list {accelerators,memory,timing,models,tiers,instances}
hbmperf catalog validate
hbmperf llm predict --model ID --device ID --precision {bf16,f16,q8_0,fp8,nvfp4,mxfp4} [--tp N] [--batch N] [--isl N] [--osl N] [--efficiency-source {peak,sim,calibrated}] [--json]
hbmperf llm sweep --model ID --device ID --precision P --batch-range 1:256 [--json]
hbmperf dram sim --preset ID --pattern {stream,random,strided,mix,kv_gather,llm_decode} [--requests N] [--queue-depth N] [--rate-gb-per-s X] [--kv-block-bytes N] [--mapping M] [--page-policy {open,closed}] [--seed N] [--log PATH] [--json]
hbmperf dram curve --preset ID --pattern P   (loaded-latency curve)
hbmperf dram check --log PATH
hbmperf hierarchy sweep --model ID --device ID --tier ID [--offload kv|weights] [--json]
hbmperf cluster table [--model ID]
hbmperf trends fit | hbmperf trends intercepts [--json]
hbmperf fetch {epoch,inferencex,mlperf} [--date YYYY-MM-DD]
hbmperf measure host | hbmperf measure llamacpp --split {calibration,evaluation} [--probe]
hbmperf validate {local,inferencex,mlperf} [--json]
hbmperf dashboard build [--out docs/dashboard/index.html]
```

Exit codes: 0 success, 1 user or data error, 2 `BLOCKED` external prerequisite, 3 validation `FAIL` when `--strict` is given.

Dashboard: single static HTML file using Plotly with `include_plotlyjs="cdn"`; panels: accelerator bandwidth and compute trends, ridge-point trend, capacity and bandwidth intercepts, HBM standard per-stack bandwidth by year, roofline with model operating points, C1 and C2 predicted-versus-measured scatter with verdict badges, I1 achieved-fraction-of-bound chart, simulator efficiency by pattern and loaded-latency curves, KV block-size study, hierarchy offload bounds, CSP instance table; each panel lists its sources and snapshot date.

## 15. Milestones

Each milestone ends with a focused conventional commit pushed to `origin main` and a check that `git ls-remote origin main` matches the local `HEAD`.

### Milestone 0: foundation, catalogs, vertical slice, and feasibility gate

Objective: prove toolchain, data availability, catalog provenance, and a thin end-to-end prediction.
Deliverables:

- Commit `BUILD_PLAN.md` first (`docs: add reviewed build plan`).
- `pyproject.toml`, `uv.lock`, `.gitignore`, `LICENSE`, `README.md` stub, `src/hbmperf/{__init__,provenance,catalog,gguf,cli}.py`.
- `data/catalog/*.csv` with sources; `SOURCES.md`, `SCHEMA.md`.
- `llm.py` decode prediction and `hbmperf llm predict`; `gguf.py` header reader (needed by guard (a)).
- `hbmperf fetch` for Epoch, InferenceX, MLPerf with manifests.
- `bench/host/` C benchmarks and `hbmperf measure host`.
- `data/manifests/c1_models.json` frozen with availability and SHA-256 verified for calibration and evaluation files (evaluation files hashed by HTTP metadata or download without benchmarking).
- Probe results for `llama-bench` and Ramulator 2 toolchain recorded in `results/probe/`.

Tests: catalog schema tests with fault fixtures, unit tests, parameter-count tests for all catalog models, provenance hash tests.
Real-path verification: `uv run hbmperf catalog validate` passes; `uv run hbmperf llm predict --model qwen2.5-0.5b --device apple_m2 --precision q8_0` prints a memory-bound prediction; `uv run hbmperf measure host` writes a run directory.
Acceptance gate: all tests pass; `ruff check` clean; three fetches verified by hash; C1 manifest has 4 calibration files and evaluation files giving at least 4 files and at least 24 depth examples in total, all with Apache-2.0 or MIT base-model and quantizer licenses; parameter counts within 2 percent; GGUF guard (a) passes on the calibration files.
Kill or fallback: if fewer than 4 evaluation files are available, add another permitted family before any measurement; if none exist, C1 support rule yields `INSUFFICIENT_EVIDENCE` and the fallback tree applies.
Commit boundary: plan commit, then `feat: catalogs, provenance, and prediction slice`, then `feat: data fetchers and host benchmarks`.

### Milestone 1: HBM pseudo-channel simulator and timing checker

Objective: a correct, deterministic simulator with golden tests and fault injection.
Deliverables: `dram/timing.py`, `mapping.py`, `traffic.py`, `sim.py`, `checker.py`; `hbmperf dram sim|curve|check`; `results/dram/efficiency.json` for presets `hbm2_2000`, `hbm2e_3200`, `hbm3_6400`, and `hbm3e_9600` (`hbm2e_3200` and `hbm3e_9600` are derived presets: nanosecond timings of the base preset held constant, cycles recomputed as `ceil(n_base * tck_base / tck_new)`, burst-limited `nBL` and `tCCD_S` unchanged, and `tCCD_L`, `tRTW`, `tRFC`, `tREFI` recomputed with the base source's formulas; labeled `derived` in `notes`).
Tests: all golden, property, metamorphic tests in 12.1; timing checker fault matrix in 12.2.
Real-path verification: CLI runs of each pattern and a loaded-latency curve; the command log from a CLI run passes `hbmperf dram check`.
Acceptance gate: all tests pass, every seeded fault detected by the expected constraint name, negative controls pass, 200,000-request stream run under 10 seconds wall time on the M2 (if slower, record the observed rate and continue; this is not a claim gate).
Commit boundary: `feat(dram): pseudo-channel simulator and timing checker`.

### Milestone 2: complete first-order LLM model

Objective: prefill and decode, GQA, tied embeddings, MoE, TP communication, capacity limits, critical batch, efficiency sources including `sim`.
Deliverables: full `llm.py`, `llm sweep`, `--efficiency-source sim` reading `results/dram/efficiency.json`.
Tests: parameter counts, KV bytes per token for Llama-3.3-70B (327,680 bytes per token at 16 bits: `2*80*8*128*2`), metamorphic monotonicity, capacity boundary tests, TP reduces per-device bytes, MoE expected-expert formula at `B=1` equals `k` experts.
Real-path verification: CLI predictions for Llama-3.3-70B on H100, H200, B200, MI300X printed with bound class and `B*`.
Acceptance gate: all tests pass and the predictions are internally consistent (memory-bound at batch 1, compute-bound above `B*`).
Commit boundary: `feat(llm): complete first-order inference model`.

### Milestone 3: local measurement and C1 evaluation (portfolio-ready core)

Objective: measure calibration files, fit, freeze, measure evaluation files once, report.
Deliverables: `measure.py` llama-bench harness, `validate/local.py`, `stats.py` (cluster bootstrap and verdicts over pre-filtered `(prediction, baseline, measurement, cluster id)` rows with the threshold, upper threshold, and support minima passed in, so C2 reuses it unchanged; a test runs a C2-shaped synthetic set through it), `data/measurements/c1/*.json`, `results/validation/c1.json` and `docs/RESULTS.md` C1 section, README quickstart and results summary.
Order inside the milestone: (1) measure calibration split with session anchors, commit; (2) fit calibration, commit; (3) create tag `eval-c1-v1` and push it; (4) measure the already-downloaded evaluation split within the 2-hour window, commit measurements; (5) run `hbmperf validate local`, commit report.
Tests: stats unit tests with synthetic ground truth, `INSUFFICIENT_EVIDENCE` and `STALE` fault tests, byte-accounting guard versus `model_size`.
Real-path verification: `uv run hbmperf validate local` prints the verdict and writes the report.
Acceptance gate: evaluation completed with the verdict computed exactly as predeclared (any of `PASS`, `FAIL`, or `INSUFFICIENT_EVIDENCE` is a completed gate; the README states the verdict truthfully and applies the fallback claim tree); clean-checkout test (section 19) passes for the core commands.
Hardware state: local only.
Commit boundary: `feat(validate): local llama.cpp measurement and C1 evaluation` plus data commits listed above.

### Milestone 4: industry validation (C2, I1)

Deliverables: `validate/inferencex.py`, `validate/mlperf.py`, derived tables in `data/derived/`, reports in `results/validation/c2.json` and `i1.json`, tag `eval-c2-v1` before the C2 evaluation run.
Tests: parsers on synthetic fixtures mirroring verified field names (`hardware`, `framework`, `precision`, `decode_tp`, `conc`, `isl`, `osl`, `metrics.median_tpot`; MLPerf `Suite`, `Category`, `Availability`, `Model`, `Scenario`, `Accelerator`, `a#`, `Nodes`, `Performance_Result`, `Performance_Units`), accelerator-name mapping table tests, dedup test for `-99` and `-99.9`.
Acceptance gate: verdicts computed per section 8.5; unmapped rows listed; slices reported.
Commit boundary: `feat(validate): InferenceX and MLPerf validation`.

### Milestone 5: memory hierarchy, cluster, and KV locality analysis

Deliverables: `hierarchy.py`, `hbmperf hierarchy sweep`, `hbmperf cluster table`, KV block-size study (`kv_gather` efficiency for block sizes 1, 4, 16, 64, 256 tokens on `hbm3_6400`) written to `results/analysis/`.
Tests: bound ordering (lower <= upper), zero offload equals single-tier result, capacity gain monotonic in offload fraction.
Acceptance gate: tests pass; results labeled model-only.
Commit boundary: `feat: memory hierarchy, cluster, and KV locality analysis`.

### Milestone 6: trends and intercepts

Deliverables: `trends.py`, `hbmperf trends fit|intercepts`, `results/trends/*.json`, and `data/derived/epoch_accelerators.csv` (the filtered device-level rows used by the fits, redistributed with Creative Commons Attribution credit) so fits and intercepts reproduce from a clean checkout without a new download.
Tests: fit recovers a known synthetic doubling time within 2 percent; intercept of synthetic lines computed exactly; `INSUFFICIENT_EVIDENCE` with fewer than 8 points; bootstrap determinism.
Acceptance gate: tests pass; every fit reports its device count and interval.
Commit boundary: `feat(trends): growth fits and architectural intercepts`.

### Milestone 7: dashboard

Deliverables: `dashboard.py`, `docs/dashboard/index.html`.
Tests: fixture build contains every panel id and source list; missing-input panel shows "not available".
Real-path verification: open the generated file in a browser and capture a screenshot for the README.
Acceptance gate: all panels render with committed results.
Commit boundary: `feat(dashboard): static trend and validation dashboard`.

### Milestone 8: Ramulator 2 cross-check (C4)

Deliverables: `tools/ramulator_crosscheck/build.sh` (clones commit `72427a1bba3771564c4fb0e494ba02242fd1eaa7` into `.cache/`, applies the patch, builds), `run.py` (writes identical traces, runs both simulators with matched HBM3 organization, timing preset `HBM3_6400Mbps`, FR-FCFS, open page, all-bank refresh, and queue depth), `results/crosscheck/c4.csv` and report.
Matching procedure: C4 uses Ramulator 2 organization `HBM3_16Gb_4hi` (one stack ID, 16 banks per pseudo-channel, tRFC 260 ns) with timing `HBM3_6400Mbps`, and a matching catalog preset `hbm3_6400_4hi`; before comparing, the preset is aligned to the exact Ramulator 2 values resolved from `python/ramulator/dram/hbm3.py`, and every remaining modeling difference (for example `nPPD`, request size, frontend issue rate, and refresh manager behavior) is documented; traces are written as pre-decoded addresses that Ramulator 2's mapper and the hbmperf mapping string both decode to the same (stack ID, bank group, bank, row, column) tuple, verified by a decode-equivalence check over every trace address before any comparison (if no built-in Ramulator 2 mapper allows this, `PassThroughAddrMapper` with pre-decoded vectors is used); both sides use 32-entry read and write queues; documented differences also include Ramulator 2's open-row protection rule (it protects a row only while the request that opened it is unserved), its buffer sharing across the two pseudo-channels of a channel (C4 traces therefore target pseudo-channel 0 only), and its frontend issue rate; C4 compares only after this alignment and never retunes after seeing the comparison for the reported suite; the suite is `stream`, `random`, `strided` (stride 4 KiB), `mix` (30 percent writes), `kv_gather` (16-token blocks).
Blocked state: if the build fails, the report is `BLOCKED` with the compiler error, and C4 is not claimed.
Acceptance gate: report produced with a per-pattern verdict; `FAIL` is reported truthfully with analysis.
Commit boundary: `feat(tools): Ramulator 2 cross-check`.

### Milestone 9: documentation and release audit

Deliverables: `README.md` (purpose, quickstart, screenshots, results table with verdicts, limitations), `docs/METHODOLOGY.md`, `docs/RESULTS.md`, `docs/LIMITATIONS.md`, final dashboard.
Acceptance gate: section 19 passes.
Commit boundary: `docs: methodology, results, and limitations`.

## 16. Compute matrix

| Workload | Preferred resource | Minimum resource | Expected duration | Checkpoint strategy | Required evidence | Fallback |
| --- | --- | --- | --- | --- | --- | --- |
| Unit, golden, property tests | Local M2 | Any CPU | Under 2 minutes | Not needed | pytest output | None needed |
| DRAM simulator sweeps | Local M2 | Any CPU | Minutes | Per-run directories | `results/dram/` | Fewer requests with recorded counts |
| Host microbenchmarks | Local M2 | Local M2 | Under 5 minutes | Per-run directories | `results/host/` | None; local only |
| llama-bench C1 | Local M2 (Metal) | Local M2 | Under 2 hours total | One JSON per file and depth, resumable by skipping completed keys | `data/measurements/c1/` | `INSUFFICIENT_EVIDENCE` if files unavailable |
| InferenceX, MLPerf, Epoch processing | Local M2 | Any CPU | Minutes | Snapshot manifests | `results/validation/`, `results/trends/` | Cached snapshot |
| Ramulator 2 cross-check | Local M2 | Local M2 with CMake and Python 3.14 | Under 30 minutes | Build cache in `.cache/` | `results/crosscheck/` | `BLOCKED` report |

No accelerator rental, Colab, or HPC is required for V1; GPU measurements are a follow-on.
Probe: `hbmperf measure llamacpp --probe` records `llama-bench` version and backend and returns exit code 2 (`BLOCKED`) when absent.

## 17. Documentation and interview walkthrough

- README: one-paragraph claim, quickstart (`uv sync`, three commands), result table with verdicts and intervals, dashboard screenshot, limitations link.
- METHODOLOGY: equations from section 8, protocols from section 13, rerun commands.
- RESULTS: C1, C2, C2-FP4, C4, C5, and I1 with tables and verdicts.
- LIMITATIONS: simplifications, guesstimate timing sources, M2-only local evidence, third-party benchmark caveats, extrapolation caveats.
- Walkthrough: section 2.

## 18. Risks, kill gates, and limitations

| Risk | Mitigation | Kill or fallback |
| --- | --- | --- |
| Noisy M2 measurements | Load-average and CV invalidation rules, randomized order, 5 repetitions | Fallback claim tree |
| Evaluation GGUF files unavailable | Milestone 0 availability check before measurement | `INSUFFICIENT_EVIDENCE` |
| Ramulator 2 timing values are guesstimates | Label in catalog; C4 is a cross-model check, not ground truth | None needed |
| InferenceX data changes or disappears | Dated snapshot with hash | Use cached snapshot; if never fetched, `BLOCKED` |
| FP4 peak not published for a device | Exclude rows with logged reason | Reduced support may yield `INSUFFICIENT_EVIDENCE` |
| Python simulator too slow | Bounded request counts, simple data structures | Report measured rate |
| MLPerf rows with mixed or multi-node hardware | Map only single-type rows, total accelerators `a# * Nodes` | Exclude and list |
| Ambiguous vendor specs (sparsity, GiB vs GB, B200 7.7 versus 8 TB/s) | Dense-only rule, unit conversion on load, notes column | Sensitivity note in LIMITATIONS |

Explicit limitations: first-order model ignores kernel-level inefficiencies other than the fitted factors, activation traffic, and scheduler interference; the simulator abstracts the command bus; hierarchy results are unvalidated; trends are extrapolations.

## 19. Final release and clean-checkout acceptance

1. `git clone` into a temporary directory, `uv sync`, `uv run pytest -q` passes, `uv run ruff check .` passes.
2. `uv run hbmperf catalog validate`, `uv run hbmperf llm predict --model llama-3.3-70b --device nvidia_h200_sxm --precision fp8 --tp 2 --batch 16`, `uv run hbmperf dram sim --preset hbm3_6400 --pattern stream --requests 20000`, `uv run hbmperf trends intercepts` (with the fetched snapshot or the committed derived data), and `uv run hbmperf dashboard build --out <tmp>/index.html` all succeed.
3. Committed results reproduce: rerunning `hbmperf validate local` from committed measurements gives the identical verdict and statistics.
4. No placeholders, no personal paths, no raw restricted snapshots, no model weights in Git.
5. `git status` clean and `origin/main` equals local `HEAD`.
No release tag, GitHub release, PyPI publication, or GitHub Pages change is made without separate authorization.

## 20. Authoritative public sources

- JEDEC HBM3 (JESD238) and HBM4 (JESD270-4) announcements: `https://www.jedec.org/news/pressreleases`.
- Ramulator 2: `https://github.com/CMU-SAFARI/ramulator2` (MIT), `python/ramulator/dram/hbm2.py`, `python/ramulator/dram/hbm3.py`.
- DRAMsim3: `https://github.com/umd-memsys/DRAMsim3` (MIT), `configs/HBM2_8Gb_x128.ini`.
- Epoch AI ML Hardware: `https://epoch.ai/data/machine-learning-hardware`.
- InferenceX: `https://github.com/SemiAnalysisAI/InferenceX`, `https://inferencex.semianalysis.com/api/openapi.json`.
- MLPerf Inference results: `https://github.com/mlcommons/inference_results_v5.1`, `https://github.com/mlcommons/inference_results_v6.0`.
- NVIDIA H100, H200, B200, GH200 datasheets; AMD Instinct MI300X, MI325X, MI355X datasheets; Google Cloud TPU documentation; Intel Gaudi documentation (exact URLs in `data/catalog/accelerators.csv`).
- Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", arXiv 2402.13499.
- GH200 characterization, arXiv 2408.11556.
- Llama 3 herd of models, arXiv 2407.21783 (Llama 3 70B architecture).
- Qwen2.5, SmolLM2, TinyLlama model cards and `config.json` on Hugging Face.
- llama.cpp: `https://github.com/ggml-org/llama.cpp` (MIT).
- AWS EC2 accelerated instance documentation, Google Cloud accelerator-optimized machines, Microsoft Azure ND-series documentation (exact URLs in `data/catalog/csp_instances.csv`).

## 21. Plan amendments

Amendments made during implementation, each before any affected evaluation data was seen.

- A1 (2026-09-28, before any C1 measurement): the contention rule in section 13.1 originally invalidated a configuration when the 1-minute load average exceeded 2.0. On the measurement machine the idle desktop load average was already 2.4 to 2.75, and each llama-bench run raises the 1-minute average by itself, so the rule would have rejected measurements for reasons unrelated to contention. It was replaced by a direct measure of CPU used by other processes (more than 200 percent after a 60-second bounded wait marks the configuration `invalid`); the load average is still recorded.
- A2 (2026-09-29, after all evaluations): at the repository owner's request, the public repository history is published as a single squashed commit instead of pushed milestone commits. The milestone history, including the tags `eval-c1-v1` and `eval-c2-v1` that froze code and calibration before each evaluation, is kept in the owner's local repository; the committed reports record the tagged commit SHAs and `claim_input_sha256` values. Separately, the C4 harness uses Ramulator 2's `CacheLineInterleave` channel mapper instead of `PassThroughAddrMapper`-style pass-through because the pass-through channel mapper crashes at the pinned commit; with a single channel it passes addresses through unchanged, so the decode-equivalence check is unaffected.
