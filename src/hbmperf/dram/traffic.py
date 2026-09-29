"""Seeded traffic generators yielding (arrival_cycle or None, pc-local byte address, is_write)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from hbmperf.dram.timing import Preset
from hbmperf.provenance import DEFAULT_SEED

PATTERNS = ("stream", "random", "strided", "mix", "kv_gather", "llm_decode", "single_bank")


def addresses(p: Preset, pattern: str, n: int, *, seed: int = DEFAULT_SEED,
              stride: int = 4096, write_fraction: float = 0.3, block_bytes: int = 1024,
              weight_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Return (addresses, is_write) arrays for `n` burst-sized requests."""
    rng = np.random.Generator(np.random.PCG64(seed))
    bb = p.burst_bytes
    span = p.pc_bytes
    writes = np.zeros(n, dtype=bool)
    if pattern == "stream":
        addr = np.arange(n, dtype=np.int64) * bb
    elif pattern == "random":
        addr = rng.integers(0, span // bb, n, dtype=np.int64) * bb
    elif pattern == "strided":
        addr = np.arange(n, dtype=np.int64) * max(stride, bb)
    elif pattern == "mix":
        addr = rng.integers(0, span // bb, n, dtype=np.int64) * bb
        writes = rng.random(n) < write_fraction
    elif pattern == "kv_gather":
        addr = _kv_gather(rng, n, bb, block_bytes, span)
    elif pattern == "llm_decode":
        # weights stream sequentially from one region while KV blocks are gathered from another;
        # each request comes from the weight stream with probability `weight_fraction`
        from_weights = rng.random(n) < weight_fraction
        n_w = int(from_weights.sum())
        half = span // 2
        w = np.arange(n_w, dtype=np.int64) * bb % half
        kv = half + _kv_gather(rng, n - n_w, bb, block_bytes, half)
        addr = np.empty(n, dtype=np.int64)
        addr[from_weights] = w
        addr[~from_weights] = kv
    elif pattern == "single_bank":
        # every burst of consecutive rows of bank 0 under the column-lowest mapping
        from hbmperf.dram.mapping import Mapper

        m = Mapper(p, "RoSiBaBgCo")
        per_row = p.bursts_per_row
        addr = np.array([m.encode(0, 0, 0, (i // per_row) % p.rows, i % per_row)
                         for i in range(n)], dtype=np.int64)
    else:
        raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")
    return addr % span, writes


def _kv_gather(rng, n: int, bb: int, block_bytes: int, span: int) -> np.ndarray:
    block_bytes = max(bb, block_bytes // bb * bb)
    per_block = block_bytes // bb
    n_blocks = -(-n // per_block)
    bases = rng.integers(0, span // block_bytes, n_blocks, dtype=np.int64) * block_bytes
    offsets = np.arange(per_block, dtype=np.int64) * bb
    return (bases[:, None] + offsets[None, :]).reshape(-1)[:n]


def closed_loop(addr: np.ndarray, writes: np.ndarray) -> Iterator[tuple[None, int, bool]]:
    yield from ((None, a, w) for a, w in zip(addr.tolist(), writes.tolist(), strict=True))


def open_loop(p: Preset, addr: np.ndarray, writes: np.ndarray, rate_gb_per_s: float,
              seed: int = DEFAULT_SEED) -> Iterator[tuple[int, int, bool]]:
    """Poisson arrivals at the offered load `rate_gb_per_s` (bytes = burst_bytes per request)."""
    rng = np.random.Generator(np.random.PCG64(seed + 1))
    mean_gap_ck = p.burst_bytes / (rate_gb_per_s * 1e9) / (p.tck_ps * 1e-12)
    arrivals = np.floor(np.cumsum(rng.exponential(mean_gap_ck, len(addr)))).astype(np.int64)
    yield from zip(arrivals.tolist(), addr.tolist(), writes.tolist(), strict=True)
