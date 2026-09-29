# Limitations

What the evidence in this repository does and does not support.

## Scope of the evidence

- Local measurements come from one Apple M2 with 8 GB of LPDDR5 running llama.cpp on Metal. They validate the first-order model's byte accounting and calibration procedure on that machine; they are not evidence about HBM, NVIDIA, or AMD behavior.
- Datacenter validation (C2, I1) uses third-party published results (SemiAnalysis InferenceX, MLCommons MLPerf). Their software stacks, precisions, and batching are self-reported; the model does not see kernel choices, speculative decoding, or scheduler details.
- No GPU, HBM device, or cloud instance was measured by this project.

## Simulator simplifications

- One pseudo-channel is simulated and scaled by the pseudo-channel count; cross-pseudo-channel interference, the shared channel command bus, and controller buffer sharing are not modeled.
- At most one row and one column command issue per cycle; nPPD and finer command-bus occupancy classes are not modeled.
- All-bank refresh only (no per-bank refresh or refresh management); no power-down, ECC, or thermal throttling.
- Core timings of `hbm2_2000` and `hbm3_6400` are labeled "Ramulator Guesstimate" in their source; `hbm2e_3200` and `hbm3e_9600` are derived by holding nanosecond timings constant. Results are sensitivity studies of plausible HBM behavior, not vendor-accurate predictions. HBM3E has no separate JEDEC standard.
- The Ramulator 2 cross-check compares two models against each other; agreement is not ground truth.

## First-order model simplifications

- Activation traffic, softmax and normalization FLOPs, kernel launch gaps, and scheduler interference are not modeled explicitly; the fitted `eta_m`, `eta_c`, `alpha`, and `t_fixed` absorb them for the calibration hardware only.
- Continuous batching is represented by a steady-state concurrency; prefill interference with decode (which dominates at 8k-token inputs in C2) is not modeled.
- The KV cache is assumed FP8 for InferenceX rows; the true precision is not published per row.
- Vendor scale-up bandwidth figures are aggregate and sometimes bidirectional; the model uses half as a per-direction ring bandwidth and lets `alpha` absorb the difference.
- MoE traffic assumes independent uniform routing.
- FP4 predictions (C2-FP4) extrapolate a calibration obtained only on FP8 hardware.

## Datasets

- Vendor catalog values have documented ambiguities (for example B200 bandwidth 7.7 versus 8.0 TB/s, Gaudi 3 BF16 1678 versus 1835 TFLOPS); see `data/catalog/SOURCES.md`.
- InferenceX API data carries no explicit data license; only derived statistics and per-row predictions with attribution are committed, not the raw snapshot.
- The MLPerf v5.0 results repository has no license file; its rows are used only for the I1 invariant.
- Trend fits pool every Epoch AI device with the relevant field (including consumer GPUs and non-GPU accelerators), so doubling times describe that population, not any vendor roadmap. Intercept years are log-linear extrapolations with bootstrap intervals, not forecasts.

## Hierarchy and cluster analyses

Offload and CSP instance results are model-only: they are internally consistent with the validated first-order model but have not been compared with any measured offload or multi-GPU system.
