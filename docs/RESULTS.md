# Results

All numbers come from committed result files; `hbmperf dashboard build` renders them.
Thresholds and rules were fixed in `BUILD_PLAN.md` before any evaluation data was seen.

## Verdicts

| Claim | What it tests | Verdict | Key numbers |
| --- | --- | --- | --- |
| C1 | Calibrated decode step time on held-out model families, local Apple M2 | PASS | MdAPE 11.0% (95% CI 7.6-13.2%) vs 22.4% calibrated file-size baseline and 68.4% spec-peak model |
| C2 | Calibrated TPOT on held-out accelerators (B200, MI325X, MI355X), InferenceX | FAIL (fallback c) | MdAPE 19.1% (CI 15.8-21.9%) meets the 35% threshold but does not beat the bandwidth-scaled baseline (18.9%) |
| C2-FP4 | Precision extrapolation (NVFP4, MXFP4), reported only | n/a | MdAPE 19.7%; B200 NVFP4 14.5%, MI355X MXFP4 38.3% |
| I1 | MLPerf Llama2-70B Offline throughput below the dense compute bound | PASS | 98 rows, 0 violations; achieved fraction of peak median 0.22, max 0.33 |
| C4 | hbmperf versus Ramulator 2 on identical traces | PASS | bandwidth within 0.2-3.7%, row-hit rate within 0.001 on all five patterns |
| C5 | Simulator golden tests and timing-checker fault injection | PASS | 198 tests pass; every seeded constraint fault detected, all negative controls clean |

## C1: local decode on held-out model families

- Calibration (Qwen2.5 0.5B and 1.5B, Q8_0 and F16, 24 of 24 configurations valid): `eta_m` = 0.374 of the M2's 100 GB/s, `t_fixed` = 4.31 ms per token.
- Evaluation (SmolLM2 360M and 1.7B, TinyLlama 1.1B, Q8_0 and F16, depths 0 to 4096; 30 of 32 configurations valid, two exceeded the 15% CV limit): measured once, one minute after calibration ended, at tag `eval-c1-v1`.
- Session anchor drift: -1.2% (limit 10%).

| Predictor | MdAPE (step time) |
| --- | --- |
| Calibrated model (candidate) | 11.0% (95% CI 7.6-13.2%; leave-one-file-out 8.8-11.1%) |
| Ablation: calibrated, weights-only bytes | 13.8% |
| Calibrated file-size baseline | 22.4% (CI 13.5-29.0%) |
| Spec-peak model (no calibration) | 68.4% |
| Drift-normalized candidate (secondary) | 10.3% |

The paired difference (baseline minus candidate) is 11.5 points with 95% CI 0.8-21.2, so the candidate beats the baseline.
The file-size baseline cannot see context depth; the model's KV-cache term matters most for SmolLM2-1.7B, which uses full multi-head attention (196 KB of KV per token), where measured step time grows from 57 ms at depth 0 to 112 ms at depth 4096 (Q8_0).
The largest misses are SmolLM2-1.7B Q8_0 at depth 256 (97 ms measured versus 55 ms predicted) and depth 4096 (112 versus 75 ms): the model under-predicts the attention cost of long multi-head contexts on this backend.

Byte accounting: for all ten GGUF files the model's decode weight bytes match the file's tensor inventory within 0.05%.

## C2: held-out datacenter accelerators

Calibration on 237 H100, H200, and MI300X FP8 rows gave `eta_m` 0.50, `eta_c` 0.20, all-reduce hop latency 2 us, and `t_fixed` 2 ms (in-sample MdAPE 20.4%).
On 252 held-out FP8 rows in 16 clusters:

| Slice | Rows | MdAPE |
| --- | --- | --- |
| All held-out FP8 | 252 | 19.1% |
| B200 / MI325X / MI355X | 132 / 60 / 60 | 19.4% / 19.7% / 16.2% |
| ISL 1k, OSL 1k / 1k, 8k / 8k, 1k | 84 each | 17.0% / 16.7% / 26.1% |
| Concurrency at most 16 / above 16 | 144 / 108 | 16.8% / 22.4% |
| Spec-peak model (no calibration) | 252 | 44.8% |

The claim fails only on its baseline clause: scaling measured calibration-hardware TPOT by the ratio of peak HBM bandwidth predicts held-out TPOT as well as the model (18.9%).
That is itself evidence for the memory-bound premise: across these accelerator generations, decode TPOT tracks HBM bandwidth almost one-to-one.
The worst slice, 8k-token inputs, is where unmodeled prefill interference in continuous batching is largest.

## I1: MLPerf compute bound

98 closed-division, available datacenter Llama2-70B Offline rows (v5.0-v6.0) map to catalog accelerators; none exceeds the dense output-token compute bound.
Median achieved fraction by accelerator (latest round): H200 0.30, H100 0.27, MI325X 0.22, B200 0.20 (many rows use FP4 peaks), MI300X 0.18, GB200 0.17, MI355X 0.17.
13 rows come from cloud-provider submitters.
Unmapped rows (B300, GB300, MI350X, H100 NVL, H200 NVL, L40S, RTX PRO 6000, Arc Pro B60) are listed in `results/validation/i1.json`.

## C4: Ramulator 2 cross-check

| Pattern | hbmperf GB/s | Ramulator 2 GB/s | Difference | Row-hit rate (both) |
| --- | --- | --- | --- | --- |
| stream | 19.16 | 19.51 | -1.8% | 0.969 / 0.968 |
| random | 7.73 | 7.84 | -1.5% | 0.000 / 0.000 |
| strided 4 KiB | 2.67 | 2.68 | -0.2% | 0.000 / 0.000 |
| mix, 30% writes | 6.61 | 6.86 | -3.7% | 0.000 / 0.000 |
| KV gather, 16-token blocks | 19.16 | 19.51 | -1.8% | 0.969 / 0.968 |

Mean read latency is not compared: Ramulator 2 reports about twice hbmperf's value on every pattern, and the two tools measure latency from different points (Ramulator 2's frontend issue versus hbmperf's controller admission); the ratio is recorded in `results/crosscheck/c4.json`.

## Simulator studies

Efficiency is achieved over peak pseudo-channel bandwidth, closed-loop, 32-entry queues, with refresh (`results/dram/efficiency.json`):

| Preset | Stream | LLM decode mix | Random / 1 KB gather |
| --- | --- | --- | --- |
| HBM2 2.0 Gb/s | 0.88 | 0.86 | 0.44 |
| HBM2E 3.2 Gb/s (derived) | 0.84 | 0.59 | 0.25 |
| HBM3 6.4 Gb/s | 0.88 | 0.69 | 0.30 |
| HBM3E 9.6 Gb/s (derived) | 0.82 | 0.48 | 0.20 |

- Fine-grained access loses efficiency as the pin rate rises: row-activation limits (tFAW, tRRD, tRC) are fixed in nanoseconds while bursts get shorter, so extra pin bandwidth helps streaming but not small gathers.
- Paged KV blocks (`results/dram/kv_block_sweep.json`, 4 KB per token per layer): one-token blocks reach 0.73 on HBM3 and 0.58 on HBM3E, versus 0.88 and 0.82 for 16-token blocks; efficiency saturates at 16 tokens.
- Queue depth: at 32 entries HBM3E streaming reaches 0.92 without refresh; 64 entries recover 0.98, because tFAW spans more clock cycles at higher data rates.
- Loaded latency (HBM3 random reads): mean 68 ns at 10% load, 164 ns at 90%, 528 ns at saturation; p99 stays near 400 ns at low load because all-bank refresh blocks for 350 ns.
- Plausibility (not validation): simulated HBM3 streaming efficiency with refresh (0.88-0.90) is close to published H100 streaming-read measurements of 91-94% of peak.

## Hardware trends and intercepts

Epoch AI ML Hardware data, log-linear fits with 2000-resample device bootstrap (`results/trends/trends.json`):

| Metric | Devices | Doubling time (95% CI) |
| --- | --- | --- |
| Memory bandwidth | 136 | 2.77 years (2.40-3.20) |
| Dense BF16 compute | 65 | 2.40 years (2.03-2.91) |
| Memory capacity | 121 | 3.03 years (2.54-4.04) |
| Ridge point (FLOP per byte) | 63 | 9.45 years (6.69-17.19) |

Intercepts (extrapolations, not forecasts):

- Single-device bandwidth for Llama-3.3-70B FP8 batch-1 decode at 5 ms TPOT (about 14 TB/s): trend crosses in 2030.9 (95% CI 2029.3-2033.0).
- Single-device capacity for Llama-3.3-70B BF16 weights plus 32 sequences of 8k-token BF16 KV (about 227 GB): trend crosses in 2029.9 (95% CI 2027.8-2034.1).

## Memory hierarchy and clusters (model-only)

Llama-3.3-70B FP8, 8k input, 1k output (`results/analysis/hierarchy.json`):

- GH200 (HBM3E) with KV offload to Grace LPDDR5X over NVLink-C2C: 0% offload fits 39 sequences at 1527 tok/s; 10% offload fits 43 sequences at 1100-1690 tok/s (serialized to overlapped); beyond 25% offload throughput falls under either assumption. Small offload fractions pay off only if tier reads overlap HBM reads.
- H100 SXM (80 GB): the weights plus one 9k-token KV cache do not fit in 90% of HBM; KV offload over PCIe Gen5 makes batch 1 to 9 feasible at roughly 31-59 tok/s.
- CSP instances (TP across all GPUs, 1k/1k, batch capped at 256): predicted decode throughput ranges from about 13.6k tok/s (8x H100) to 25.3k tok/s (4x GB200); every configuration is memory-bound at this context.

## Local memory hierarchy (Apple M2, measured)

Pointer-chase latency: 2.0 ns up to 128 KiB, about 9.7 ns to 4 MiB, 12.7 ns at 8 MiB, 32 ns at 16 MiB, and about 117-122 ns from 32 MiB up (DRAM).
Streaming read bandwidth from CPU threads: 17.9, 35.4, 63.3, and 64.1 GB/s with 1, 2, 4, and 8 threads (Apple publishes 100 GB/s).
