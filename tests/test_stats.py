import numpy as np
import pytest

from hbmperf import stats


def _data(n_clusters=6, per=5, cand_err=0.05, base_err=0.5, seed=0):
    rng = np.random.default_rng(seed)
    clusters, meas, cand, base = [], [], [], []
    for c in range(n_clusters):
        for _ in range(per):
            m = rng.uniform(1, 2)
            clusters.append(f"c{c}")
            meas.append(m)
            cand.append(m * (1 + cand_err * rng.choice([-1, 1])))
            base.append(m * (1 + base_err))
    return clusters, cand, base, meas


RULES = dict(threshold=0.2, upper=0.3, min_clusters=4, min_examples=24)


def test_known_error_recovered_and_passes():
    c = stats.compare(*_data(), **RULES)
    assert c.mdape == pytest.approx(0.05)
    assert c.baseline_mdape == pytest.approx(0.5)
    assert c.verdict == "PASS" and stats.fallback_claim(c).startswith("a")


def test_large_error_fails_with_fallback_b():
    c = stats.compare(*_data(cand_err=0.35), **RULES)
    assert c.verdict == "FAIL" and stats.fallback_claim(c).startswith("b")


def test_worse_than_baseline_falls_to_c():
    c = stats.compare(*_data(cand_err=0.6, base_err=0.1), **RULES)
    assert stats.fallback_claim(c).startswith("c")


def test_insufficient_support():
    c = stats.compare(*_data(n_clusters=3, per=10), **RULES)
    assert c.verdict == "INSUFFICIENT_EVIDENCE"
    c = stats.compare(*_data(n_clusters=6, per=3), **RULES)
    assert c.verdict == "INSUFFICIENT_EVIDENCE"


def test_bootstrap_deterministic():
    a = stats.compare(*_data(cand_err=0.1), **RULES).to_dict()
    assert a == stats.compare(*_data(cand_err=0.1), **RULES).to_dict()


def test_fit_linear():
    assert stats.fit_linear([1, 2, 3], [3, 5, 7]) == pytest.approx((2, 1))


def test_c2_shaped_rows_reuse_compare():
    # 16 hardware-configuration clusters with 4 concurrencies each, C2 rules
    rng = np.random.default_rng(3)
    clusters = [f"hw{i}" for i in range(16) for _ in range(4)]
    meas = rng.uniform(0.005, 0.05, len(clusters))
    pred = meas * (1 + 0.2 * rng.standard_normal(len(clusters)))
    base = meas * 1.7
    c = stats.compare(clusters, pred, base, meas, threshold=0.35, upper=0.5, min_clusters=6,
                      min_examples=60)
    assert c.n_clusters == 16 and c.n_examples == 64 and c.verdict == "PASS"
