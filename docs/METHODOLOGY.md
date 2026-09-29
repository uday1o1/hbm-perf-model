# Methodology

This document describes what each component computes, how every claim is tested, and how to rerun it.
`BUILD_PLAN.md` is the full contract; this is the implementer-facing summary.

## Units and provenance

Every numeric catalog column carries its unit (`_gb_per_s`, `_tflops`, `_ck`, ...); see `data/catalog/SCHEMA.md`.
Compute peaks are dense: vendor "with sparsity" figures are halved.
Every run writes `meta.json` (git SHA and dirty flag, Python, platform, CPU, seed, config hash).
External datasets are stored as dated snapshots in `data/external/` (not committed) with committed SHA-256 manifests in `data/manifests/`; loading a snapshot re-verifies its hash.
Validation reports record `claim_input_sha256`, which covers the modules a claim uses, the computational fields of the catalog rows it reads, and the committed measurement or snapshot files; `hbmperf validate local --status` reports `STALE` if any of them change.

## HBM pseudo-channel simulator (`hbmperf.dram`)

One pseudo-channel is simulated in integer command-clock cycles and scaled by the pseudo-channel count for stack or device figures.

- Organization: stack IDs x bank groups x banks, rows of `row_bytes`, 32-byte bursts (HBM2: 64-bit pseudo-channel x BL4; HBM3: 32-bit x BL8).
- Constraints: tRCD (read and write), tRP, tRAS, tRC = tRAS + tRP, tCL, tCWL, tBL, tCCD_S, tCCD_L, tCCD_R (different stack ID), tRRD_S, tRRD_L, tFAW, tWR, tRTP, tWTR_S and tWTR_L, tRTW, tRFC, tREFI.
- Controller: separate 32-entry read and write queues, write drain between 80 and 20 percent watermarks, and each cycle at most one column command (oldest legal row hit) and one row command (oldest request's legal ACT or PRE). A bank whose open row still has a queued hit is never precharged (open-row protection). All-bank refresh every tREFI blocks the pseudo-channel for tRFC after a precharge-all.
- Time advances event to event: when nothing can issue, the clock jumps to the earliest cycle at which some command becomes legal.
- Address mapping: fields above the burst offset, most significant first. The default `RoCoSiBaBg` puts the bank group lowest so consecutive bursts alternate bank groups (limited by tCCD_S); `RoSiBaBgCo` keeps consecutive bursts in one bank (limited by tCCD_L).
- Traffic: seeded generators for stream, random, strided, read/write mix, paged KV gather, and an LLM decode mix (80 percent weight stream, 20 percent KV gather); closed-loop (queues kept full) or open-loop Poisson arrivals.

Correctness evidence (claim C5, `tests/test_dram.py`):

- Isolated read latency equals `tCL + tBL` (hit), `tRCD + tCL + tBL` (miss), and `tRP + tRCD + tCL + tBL` (conflict) exactly, for every preset.
- Single-bank row conflicts complete one access per tRC within 1 percent.
- Bank-group-interleaved streaming reaches at least 95 percent of peak with 64-entry queues; single-bank streaming stays at or below `tBL/tCCD_L + 0.02`.
- Refresh costs `1 - (tRFC + tRP)/tREFI` within 0.02.
- Closed-page random traffic never exceeds `4 * burst_bytes / tFAW`, and no log has more than four ACTs in a tFAW window.
- An independent checker (`hbmperf.dram.checker`) replays each command log from command history, sharing no scheduling code; for every pattern, page policy, and preset the unmodified scheduler produces zero violations, and scheduler variants with one constraint disabled are caught under that constraint's name.
- Little's law holds between time-integrated queue occupancy and per-request queueing delay within 2 percent; results are bit-for-bit deterministic for a seed; halving tCK doubles streaming bandwidth.

## First-order LLM inference model (`hbmperf.llm`)

Parameters per layer: attention `d*(h*hd) + 2*d*(kv*hd) + (h*hd)*d` (plus QKV bias where present), gated MLP `3*d*d_ff`, or MoE experts `3*d*d_ff_expert` plus a router.
Parameter counts match published counts for every catalog model (exactly for eight of nine).

For a decode step with batch `B` and mean cached context `L`:

- Weight bytes: all layer weights, the LM head (the whole vocabulary matrix even when tied), and `B` embedding rows when untied; MoE layers read the expected number of distinct experts `E * (1 - (1 - k/E)^B)`.
- KV bytes: `B * L * kv_bytes_per_token` read plus `B * kv_bytes_per_token` written, with `kv_bytes_per_token = 2 * layers * kv_heads * head_dim * kv_bits / 8`.
- FLOPs: `2 * layer_matmul_params * B + 2 * lm_head * B + 4 * layers * heads * head_dim * L * B`.
- Time: `max(FLOPs / (P * eta_c), bytes / (W * eta_m)) + t_fixed + t_comm`, where tensor parallelism divides bytes and FLOPs by `p` and adds two ring all-reduces per layer, `2(p-1)/p * msg / link_bw + 2(p-1) * alpha`.
- Prefill: `2 * layer_matmul_params * S * B + 2 * lm_head * B + 2 * layers * heads * head_dim * S^2 * B` (causal; the LM head runs on the last prompt token only).
- Capacity: weights and KV at the maximum context must fit in 90 percent of device memory.

Precisions: `bf16`/`f16` (16-bit weights and KV), `q8_0` (8.5 bits: 32 int8 values plus an fp16 scale), `fp8` (8-bit weights and KV), `nvfp4` (4.5 bits), `mxfp4` (4.25 bits).

## Claim C1: local decode measurement (`hbmperf.validate.local`)

- Hardware: the development Apple M2 (8 GB), llama.cpp `llama-bench` (Homebrew 0.5.0, Metal backend) with its built-in warm-up; `-p 0 -n 128 -d <depth> -r 5`; step time is the median of `samples_ns / 128`.
- Files: 10 Apache-2.0 GGUF files frozen in `data/manifests/c1_models.json` by Hugging Face LFS SHA-256 before any measurement. Calibration: Qwen2.5 0.5B and 1.5B (Q8_0, F16). Evaluation: SmolLM2 360M and 1.7B, TinyLlama 1.1B (Q8_0, F16). Depths 0 to 4096 within each model's context limit.
- Byte-accounting guards: `hbmperf.gguf` reads each file's tensor inventory; stored tensor bytes must equal llama-bench's `model_size`, and the model's batch-1 decode weight bytes must be within 1 percent of the GGUF decode-read bytes (every tensor except the token-embedding table when a separate output tensor exists). All ten files pass within 0.05 percent.
- Session control: a configuration waits (up to 60 s) until other processes use at most two cores, else it is invalid; a configuration with coefficient of variation above 15 percent is retried once, then invalid. Each session starts and ends with an anchor (Qwen2.5-0.5B Q8_0, depth 0); the evaluation session must start within two hours of the calibration session and its anchor must be within 10 percent of the calibration anchor, else the verdict is `INSUFFICIENT_EVIDENCE`. A drift-normalized MdAPE is reported regardless.
- Fit: `step_time = unit_memory_time / eta_m + t_fixed` by least squares on calibration examples only; the file-size baseline `step_time = a * file_bytes + b` is fitted on the same examples.
- Freeze and holdout: the fitted calibration and every C1 module are committed and tagged `eval-c1-v1` before any evaluation file is benchmarked; the evaluation split is measured once.
- Decision: absolute percentage error of step time; median over examples; paired cluster bootstrap over model files (2000 resamples, seed 20260928). `PASS` requires MdAPE at most 20 percent, upper 95 percent bound at most 30 percent, and the lower bound of (baseline MdAPE minus model MdAPE) above zero, with at least 4 files and 24 examples.

## Claim C2: held-out accelerators (`hbmperf.validate.inferencex`)

- Data: SemiAnalysis InferenceX public API snapshot for Llama-3.3-70B (single-node, non-disaggregated), median TPOT per (hardware, framework, precision, TP, concurrency, ISL, OSL).
- Prediction: decode step at batch = concurrency per replica, mean context `ISL + OSL/2`, FP8 KV cache (labeled assumption).
- Calibration: bounded grid search over `eta_m`, `eta_c`, `alpha`, and `t_fixed` minimizing mean squared log error on H100, H200, and MI300X FP8 rows only; committed and tagged `eval-c2-v1` before evaluation.
- Evaluation: B200, MI325X, MI355X FP8 rows, clusters = (hardware, framework, precision, TP); baseline = mean calibration-hardware TPOT for the same framework, TP, concurrency, ISL, and OSL, scaled by the peak bandwidth ratio. `PASS` needs MdAPE at most 35 percent, upper bound at most 50 percent, beating the baseline, and at least 6 clusters and 60 rows. FP4 rows (NVFP4 on B200, MXFP4 on MI355X) are reported separately with no threshold.

## Invariant I1: MLPerf compute bound (`hbmperf.validate.mlperf`)

For each closed-division, available datacenter Llama2-70B Offline row of MLPerf Inference v5.0 to v6.0 whose accelerator maps to the catalog, the bound `P / (2 * active_params)` output tokens per second per accelerator (P = dense peak at the row's reported weight precision, FP8 when unclear) must be at least the measured throughput per accelerator. Input-token compute is ignored, which makes the bound looser but still valid. The informative output is the achieved fraction of peak.

## Claim C4: Ramulator 2 cross-check (`tools/ramulator_crosscheck`)

Ramulator 2 (MIT) at commit `72427a1b` with a one-line Apple-clang fix, organization `HBM3_16Gb_4hi`, timing `HBM3_6400Mbps`, HBM34 controller, FR-FCFS, open page, all-bank refresh, RoBaRaCoCh mapping, versus hbmperf preset `hbm3_6400_4hi` with mapping `RoBaBgSiCo`.
Both simulators receive the same pseudo-channel-0 traces; every address is checked to decode to the same (stack ID, bank group, bank, row, column) tuple on both sides.
Per pattern, bandwidth must agree within 15 percent and row-hit rate within 0.05.

## Trends and intercepts (`hbmperf.trends`)

Epoch AI ML Hardware rows with a release date, filtered to devices with bandwidth or tensor BF16 data, are fitted as `log2(metric) = a + b * (year - 2016)`; doubling time is `1/b`.
Intervals: 2000-resample device bootstrap; intercept years (where the fitted line crosses a requirement derived from the LLM model) are recomputed inside each resample.
Intercepts are extrapolations of a log-linear fit, not forecasts.

## Memory hierarchy (`hbmperf.hierarchy`)

Offloading a fraction `f` of the KV cache (or weights) to a second tier with bandwidth `W2` gives a memory time between `max(B1/W1, B2/W2)` (perfect overlap) and `B1/W1 + B2/W2` (serialized); offload frees HBM, raising the largest feasible batch. These are model-only analyses.

## Rerun commands

```bash
uv sync
uv run pytest -q
uv run hbmperf catalog validate
uv run hbmperf fetch epoch && uv run hbmperf fetch inferencex && uv run hbmperf fetch mlperf
uv run hbmperf dram efficiency && uv run hbmperf dram studies
uv run hbmperf measure host
uv run hbmperf measure llamacpp --split evaluation --download-only
uv run hbmperf measure llamacpp --split calibration
uv run hbmperf validate local --calibrate
uv run hbmperf measure llamacpp --split evaluation
uv run hbmperf validate local
uv run hbmperf validate inferencex --calibrate && uv run hbmperf validate inferencex
uv run hbmperf validate mlperf
uv run hbmperf trends fit
uv run hbmperf hierarchy write
tools/ramulator_crosscheck/build.sh && uv run python tools/ramulator_crosscheck/run.py
uv run hbmperf dashboard build
```

Fresh measurements and fetches produce new data; the committed reports reproduce from committed data with `uv run hbmperf validate local` and `uv run hbmperf trends intercepts`.
