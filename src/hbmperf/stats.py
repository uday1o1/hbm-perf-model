"""Cluster-bootstrap error statistics and predeclared verdict rules (BUILD_PLAN section 8.5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from hbmperf.provenance import DEFAULT_SEED

N_RESAMPLES = 2000


def ape(pred, meas) -> np.ndarray:
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    return np.abs(pred - meas) / meas


@dataclass
class Comparison:
    n_clusters: int
    n_examples: int
    mdape: float
    mdape_ci: tuple[float, float]
    baseline_mdape: float
    baseline_mdape_ci: tuple[float, float]
    diff: float  # baseline minus candidate MdAPE (positive favors candidate)
    diff_ci: tuple[float, float]
    loco_range: tuple[float, float]  # leave-one-cluster-out MdAPE range
    verdict: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def compare(clusters: list[str], pred, baseline, meas, *, threshold: float, upper: float,
            min_clusters: int, min_examples: int, seed: int = DEFAULT_SEED,
            n_resamples: int = N_RESAMPLES) -> Comparison:
    """Paired cluster bootstrap of candidate versus baseline median APE.

    Clusters are resampled with replacement; each resample recomputes both medians and their
    difference on the identical resampled examples.
    """
    clusters = np.asarray(clusters)
    e_c, e_b = ape(pred, meas), ape(baseline, meas)
    ids = sorted(set(clusters.tolist()))
    groups = [np.flatnonzero(clusters == c) for c in ids]
    rng = np.random.Generator(np.random.PCG64(seed))
    stats = np.empty((n_resamples, 3))
    for i in range(n_resamples):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        mc, mb = np.median(e_c[idx]), np.median(e_b[idx])
        stats[i] = (mc, mb, mb - mc)
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    loco = [float(np.median(np.delete(e_c, g))) for g in groups] if len(groups) > 1 else [0.0]
    point_c, point_b = float(np.median(e_c)), float(np.median(e_b))
    reasons = []
    if len(ids) < min_clusters or len(e_c) < min_examples:
        verdict = "INSUFFICIENT_EVIDENCE"
        reasons.append(f"support {len(ids)} clusters / {len(e_c)} examples below "
                       f"{min_clusters} / {min_examples}")
    else:
        if point_c > threshold:
            reasons.append(f"MdAPE {point_c:.3f} > {threshold}")
        if hi[0] > upper:
            reasons.append(f"upper bound {hi[0]:.3f} > {upper}")
        if lo[2] <= 0:
            reasons.append(f"baseline difference lower bound {lo[2]:.3f} <= 0")
        verdict = "FAIL" if reasons else "PASS"
    return Comparison(
        n_clusters=len(ids), n_examples=int(len(e_c)), mdape=point_c,
        mdape_ci=(float(lo[0]), float(hi[0])), baseline_mdape=point_b,
        baseline_mdape_ci=(float(lo[1]), float(hi[1])), diff=point_b - point_c,
        diff_ci=(float(lo[2]), float(hi[2])), loco_range=(min(loco), max(loco)),
        verdict=verdict, reasons=reasons,
    )


def fallback_claim(c: Comparison) -> str:
    """Ordered fallback claim tree (BUILD_PLAN section 13.4)."""
    if c.verdict == "PASS":
        return "a: full claim holds"
    if c.verdict == "INSUFFICIENT_EVIDENCE":
        return "none: insufficient evidence"
    if c.diff_ci[0] > 0:
        return "b: calibrated model beats its predeclared calibrated baseline"
    return "c: measured error characterization only"


def fit_linear(x, y) -> tuple[float, float]:
    """Ordinary least squares y = a * x + b; returns (a, b)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)
