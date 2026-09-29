"""First-order LLM inference model: exact byte and FLOP accounting plus a roofline step time.

Equations are documented in docs/METHODOLOGY.md. Times are seconds, sizes bytes,
bandwidths bytes per second, compute FLOP per second.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from hbmperf import catalog

# name -> (weight_bits, kv_bits, act_bits, compute column prefix)
# GGUF Q8_0 stores 32 int8 values plus one fp16 scale per block: 34 bytes / 32 values = 8.5 bits.
# NVFP4 carries an 8-bit scale per 16 values (4 + 8/16 = 4.5 bits); MXFP4 an 8-bit scale per 32
# values (4 + 8/32 = 4.25 bits).
PRECISIONS: dict[str, tuple[float, float, float, str]] = {
    "bf16": (16.0, 16.0, 16.0, "bf16"),
    "f16": (16.0, 16.0, 16.0, "bf16"),
    "q8_0": (8.5, 16.0, 16.0, "bf16"),
    "fp8": (8.0, 8.0, 16.0, "fp8"),
    "nvfp4": (4.5, 8.0, 16.0, "fp4"),
    "mxfp4": (4.25, 8.0, 16.0, "fp4"),
}


@dataclass(frozen=True)
class Params:
    """Parameter counts by role (elements, not bytes)."""

    embed: int
    lm_head: int  # elements read by the LM head matmul (full vocab x d_model even when tied)
    attn_per_layer: int
    norm_per_layer: int
    mlp_dense_per_layer: int  # dense MLP, or shared experts for MoE
    expert: int  # one routed expert
    router_per_layer: int
    n_experts: int
    top_k: int
    n_layers: int
    final_norm: int
    tied: bool

    @property
    def total(self) -> int:
        per_layer = (
            self.attn_per_layer + self.norm_per_layer + self.mlp_dense_per_layer
            + self.router_per_layer + self.n_experts * self.expert
        )
        return self.embed + (0 if self.tied else self.lm_head) + self.n_layers * per_layer \
            + self.final_norm

    def layer_matmul_active(self) -> int:
        """Matmul parameters one token multiplies through in one layer."""
        return self.attn_per_layer + self.mlp_dense_per_layer + self.router_per_layer \
            + self.top_k * self.expert

    def expected_experts(self, tokens: int) -> float:
        """Expected distinct routed experts touched per layer by `tokens` tokens,
        assuming independent uniform top-k routing."""
        if self.n_experts == 0:
            return 0.0
        e, k = self.n_experts, self.top_k
        return e * (1.0 - (1.0 - k / e) ** tokens)


def params(model: dict) -> Params:
    d, h, kv, hd = model["d_model"], model["n_heads"], model["n_kv_heads"], model["head_dim"]
    q_out, kv_out = h * hd, kv * hd
    attn = d * q_out + 2 * d * kv_out + q_out * d
    if model["qkv_bias"]:
        attn += q_out + 2 * kv_out
    moe = model["mlp_type"] == "moe"
    n_exp = (model["n_experts"] or 0) if moe else 0
    d_ff_e = model["d_ff_expert"] or 0
    shared = (model["n_shared_experts"] or 0) * 3 * d * d_ff_e
    return Params(
        embed=model["vocab"] * d,
        lm_head=model["vocab"] * d,
        attn_per_layer=attn,
        norm_per_layer=2 * d,
        mlp_dense_per_layer=shared if moe else 3 * d * model["d_ff"],
        expert=3 * d * d_ff_e if moe else 0,
        router_per_layer=d * n_exp if moe else 0,
        n_experts=n_exp,
        top_k=(model["top_k"] or 0) if moe else 0,
        n_layers=model["n_layers"],
        final_norm=d,
        tied=model["tied_embeddings"],
    )


def kv_bytes_per_token(model: dict, kv_bits: float) -> float:
    return 2 * model["n_layers"] * model["n_kv_heads"] * model["head_dim"] * kv_bits / 8


@dataclass(frozen=True)
class Device:
    id: str
    mem_bw: float  # bytes/s
    mem_bytes: float
    peak_flops: float | None  # precision-matched dense FLOP/s, None if unpublished
    link_bw: float | None  # per-direction scale-up bytes/s


def device(dev_id: str, precision: str) -> Device:
    row = catalog.get("accelerators", dev_id)
    col = f"{PRECISIONS[precision][3]}_dense_tflops"
    tflops = row[col]
    link = row["scaleup_bw_gb_per_s"]
    return Device(
        id=dev_id,
        mem_bw=row["mem_bw_gb_per_s"] * 1e9,
        mem_bytes=row["memory_gb"] * 1e9,
        peak_flops=tflops * 1e12 if tflops else None,
        # catalog scale-up figures are aggregate bidirectional; ring all-reduce uses one direction
        link_bw=link * 1e9 / 2 if link else None,
    )


@dataclass(frozen=True)
class Knobs:
    """Calibration knobs. Defaults give the spec-peak baseline."""

    eta_m: float = 1.0  # achieved fraction of peak memory bandwidth
    eta_c: float = 1.0  # achieved fraction of peak compute
    t_fixed: float = 0.0  # fixed per-step overhead, seconds
    alpha: float = 5e-6  # per-hop all-reduce latency, seconds (labeled assumption)
    combine: str = "max"  # "max" (overlap) or "sum" (no overlap)
    weights_only: bool = False  # ablation: ignore KV traffic
    usable_mem_fraction: float = 0.9


DEFAULT_KNOBS = Knobs()


@dataclass
class Step:
    phase: str
    batch: int
    tokens_per_seq: int
    context: float
    flops: float  # per device
    bytes_weights: float  # per device
    bytes_kv: float  # per device
    t_mem: float
    t_compute: float | None
    t_comm: float
    t_fixed: float
    time: float
    bound: str

    @property
    def bytes_total(self) -> float:
        return self.bytes_weights + self.bytes_kv

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bytes_total"] = self.bytes_total
        return d


def _allreduce_time(msg_bytes: float, tp: int, dev: Device, k: Knobs) -> float:
    if tp == 1:
        return 0.0
    if dev.link_bw is None:
        raise ValueError(f"{dev.id} has no published scale-up bandwidth; cannot model tp={tp}")
    return 2 * (tp - 1) / tp * msg_bytes / dev.link_bw + 2 * (tp - 1) * k.alpha


def step(
    model: dict, dev: Device, precision: str, *, phase: str, batch: int, context: float,
    tokens_per_seq: int = 1, tp: int = 1, kv_bits: float | None = None,
    knobs: Knobs = DEFAULT_KNOBS,
) -> Step:
    """One forward step for `batch` sequences each processing `tokens_per_seq` new tokens.

    decode: tokens_per_seq = 1 and `context` is the mean number of cached tokens attended to.
    prefill: tokens_per_seq = S prompt tokens and `context` is ignored (causal S x S attention).
    """
    if batch < 1 or tp < 1 or tokens_per_seq < 1:
        raise ValueError("batch, tp, and tokens_per_seq must be >= 1")
    w_bits, default_kv_bits, act_bits, _ = PRECISIONS[precision]
    kv_bits = kv_bits or default_kv_bits
    p = params(model)
    L, h, hd = model["n_layers"], model["n_heads"], model["head_dim"]
    tokens = batch * tokens_per_seq

    # weight bytes read once per step, shared by all tokens in the step
    experts = p.expected_experts(tokens)
    layer_w = p.attn_per_layer + p.norm_per_layer + p.mlp_dense_per_layer \
        + p.router_per_layer + experts * p.expert
    embed_rows = 0 if p.tied else tokens * model["d_model"]
    w_elems = L * layer_w + p.lm_head + p.final_norm + embed_rows
    bytes_w = w_elems * w_bits / 8

    kv_tok = kv_bytes_per_token(model, kv_bits)
    if phase == "decode":
        bytes_kv = batch * context * kv_tok + tokens * kv_tok  # read cache + write new entry
        attn_flops = 4 * L * h * hd * context * batch
    elif phase == "prefill":
        bytes_kv = tokens * kv_tok
        attn_flops = 2 * L * h * hd * tokens_per_seq**2 * batch
    else:
        raise ValueError(f"unknown phase {phase!r}")
    if knobs.weights_only:
        bytes_kv = 0.0
    # the LM head runs once per sequence per step (on the last token during prefill)
    flops = 2 * L * p.layer_matmul_active() * tokens + 2 * p.lm_head * batch + attn_flops

    flops, bytes_w, bytes_kv = flops / tp, bytes_w / tp, bytes_kv / tp
    t_mem = (bytes_w + bytes_kv) / (dev.mem_bw * knobs.eta_m)
    t_comp = flops / (dev.peak_flops * knobs.eta_c) if dev.peak_flops else None
    msg = tokens * model["d_model"] * act_bits / 8
    t_comm = 2 * L * _allreduce_time(msg, tp, dev, knobs)
    tc = t_comp or 0.0
    core = max(t_mem, tc) if knobs.combine == "max" else t_mem + tc
    return Step(
        phase=phase, batch=batch, tokens_per_seq=tokens_per_seq, context=context, flops=flops,
        bytes_weights=bytes_w, bytes_kv=bytes_kv, t_mem=t_mem, t_compute=t_comp, t_comm=t_comm,
        t_fixed=knobs.t_fixed, time=core + t_comm + knobs.t_fixed,
        bound="compute" if tc > t_mem else "memory",
    )


def footprint_bytes(model: dict, precision: str, *, batch: int, max_context: int,
                    kv_bits: float | None = None) -> tuple[float, float]:
    """(weight bytes, KV bytes) resident for `batch` sequences of `max_context` tokens."""
    w_bits, default_kv_bits, _, _ = PRECISIONS[precision]
    weights = params(model).total * w_bits / 8
    kv = batch * max_context * kv_bytes_per_token(model, kv_bits or default_kv_bits)
    return weights, kv


def critical_batch(model: dict, dev: Device, precision: str, context: float,
                   kv_bits: float | None = None, tp: int = 1,
                   knobs: Knobs = DEFAULT_KNOBS) -> float:
    """Decode batch at which compute time equals memory time (math.inf if never)."""
    if dev.peak_flops is None:
        return math.inf
    s1 = step(model, dev, precision, phase="decode", batch=1, context=context, tp=tp,
              kv_bits=kv_bits, knobs=knobs)
    s2 = step(model, dev, precision, phase="decode", batch=2, context=context, tp=tp,
              kv_bits=kv_bits, knobs=knobs)
    # both times are affine in batch for dense models: t(B) = t(1) + (B - 1) * slope
    dm, dc = s2.t_mem - s1.t_mem, s2.t_compute - s1.t_compute
    if dc <= dm:
        return math.inf
    return 1 + (s1.t_mem - s1.t_compute) / (dc - dm)


@dataclass
class Prediction:
    model: str
    device: str
    precision: str
    tp: int
    batch: int
    isl: int
    osl: int
    feasible: bool
    reason: str | None
    weight_bytes_per_device: float
    kv_bytes_per_device: float
    capacity_bytes_per_device: float
    ttft_s: float
    tpot_s: float
    output_tok_per_s: float
    output_tok_per_s_per_device: float
    output_tok_per_s_per_seq: float
    mem_bw_utilization: float
    arithmetic_intensity: float
    ridge_point: float | None
    critical_batch: float
    bound: str
    decode: dict
    prefill: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


def predict(
    model_id: str, device_id: str, precision: str, *, batch: int = 1, isl: int = 1024,
    osl: int = 128, tp: int = 1, kv_bits: float | None = None, knobs: Knobs = DEFAULT_KNOBS,
) -> Prediction:
    model = catalog.get("models", model_id)
    dev = device(device_id, precision)
    ctx = isl + osl / 2  # mean cached context during generation
    dec = step(model, dev, precision, phase="decode", batch=batch, context=ctx, tp=tp,
               kv_bits=kv_bits, knobs=knobs)
    pre = step(model, dev, precision, phase="prefill", batch=1, context=0, tokens_per_seq=isl,
               tp=tp, kv_bits=kv_bits, knobs=knobs) if isl > 0 else None
    w, kv = footprint_bytes(model, precision, batch=batch, max_context=isl + osl, kv_bits=kv_bits)
    cap = dev.mem_bytes * knobs.usable_mem_fraction
    feasible = (w + kv) / tp <= cap
    reason = None if feasible else (
        f"needs {(w + kv) / tp / 1e9:.1f} GB per device, usable {cap / 1e9:.1f} GB"
    )
    return Prediction(
        model=model_id, device=device_id, precision=precision, tp=tp, batch=batch, isl=isl,
        osl=osl, feasible=feasible, reason=reason, weight_bytes_per_device=w / tp,
        kv_bytes_per_device=kv / tp, capacity_bytes_per_device=cap, ttft_s=pre.time if pre else 0.0,
        tpot_s=dec.time, output_tok_per_s=batch / dec.time,
        output_tok_per_s_per_device=batch / dec.time / tp,
        output_tok_per_s_per_seq=1 / dec.time,
        mem_bw_utilization=dec.bytes_total / dec.time / dev.mem_bw,
        arithmetic_intensity=dec.flops / dec.bytes_total,
        ridge_point=dev.peak_flops / dev.mem_bw if dev.peak_flops else None,
        critical_batch=critical_batch(model, dev, precision, ctx, kv_bits, tp, knobs),
        bound=dec.bound, decode=dec.to_dict(), prefill=pre.to_dict() if pre else None,
    )
