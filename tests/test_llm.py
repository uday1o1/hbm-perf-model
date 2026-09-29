import math

import pytest

from hbmperf import catalog, llm


@pytest.mark.parametrize("model_id", sorted(catalog.load("models")))
def test_parameter_count_matches_published(model_id):
    m = catalog.get("models", model_id)
    assert llm.params(m).total == pytest.approx(m["published_params"], rel=0.02)


def test_kv_bytes_per_token_llama33_70b():
    assert llm.kv_bytes_per_token(catalog.get("models", "llama-3.3-70b"), 16) == 327_680


def test_moe_expected_experts():
    p = llm.params(catalog.get("models", "mixtral-8x7b"))
    assert p.expected_experts(1) == pytest.approx(2)
    assert p.expected_experts(1000) == pytest.approx(8)
    assert p.expected_experts(4) < p.expected_experts(8)


def _step(batch=1, ctx=1024, tp=1, dev="nvidia_h100_sxm", model="llama-3.3-70b", prec="fp8", **k):
    return llm.step(catalog.get("models", model), llm.device(dev, prec), prec, phase="decode",
                    batch=batch, context=ctx, tp=tp, knobs=llm.Knobs(**k))


def test_decode_batch1_memory_bound_and_weights_dominate():
    s = _step()
    assert s.bound == "memory"
    assert s.bytes_weights == pytest.approx(70.55e9 - 128256 * 8192 + 8192, rel=0.01)


def test_monotonic_in_batch_and_bandwidth():
    times = [_step(batch=b).time for b in (1, 2, 4, 8, 64, 256)]
    assert times == sorted(times)
    slow = _step(dev="nvidia_h100_sxm").t_mem
    fast = _step(dev="nvidia_h200_sxm").t_mem
    assert fast < slow


def test_tp_divides_bytes_and_adds_comm():
    s1, s2 = _step(tp=1), _step(tp=2)
    assert s2.bytes_total == pytest.approx(s1.bytes_total / 2)
    assert s2.t_comm > 0 and s1.t_comm == 0


def test_sum_combine_not_faster_than_max():
    assert _step(batch=64, combine="sum").time >= _step(batch=64).time


def test_weights_only_ablation_drops_kv():
    assert _step(weights_only=True).bytes_kv == 0


def test_critical_batch_short_context():
    m = catalog.get("models", "llama-3.3-70b")
    dev = llm.device("nvidia_h100_sxm", "fp8")
    b = llm.critical_batch(m, dev, "fp8", context=1)
    # at negligible context B* ~ ridge point * bytes per param / 2 flops per param
    assert b == pytest.approx(1979e12 / 3350e9 * 1.0 / 2, rel=0.05)
    assert _step(batch=int(b * 1.2), ctx=1).bound == "compute"
    assert _step(batch=int(b * 0.8), ctx=1).bound == "memory"


def test_capacity_infeasible_on_small_device():
    r = llm.predict("llama-3.3-70b", "nvidia_l4", "fp8", batch=1)
    assert not r.feasible and "GB" in r.reason
    assert llm.predict("llama-3.3-70b", "nvidia_h200_sxm", "fp8", batch=1).feasible


def test_prefill_lm_head_once_per_sequence():
    m = catalog.get("models", "qwen2.5-0.5b")
    dev = llm.device("nvidia_h100_sxm", "bf16")
    p = llm.params(m)
    s = llm.step(m, dev, "bf16", phase="prefill", batch=1, context=0, tokens_per_seq=100)
    expected = 2 * m["n_layers"] * p.layer_matmul_active() * 100 + 2 * p.lm_head + \
        2 * m["n_layers"] * m["n_heads"] * m["head_dim"] * 100**2
    assert s.flops == pytest.approx(expected)


def test_no_peak_means_memory_only():
    r = llm.predict("qwen2.5-0.5b", "apple_m2", "q8_0", isl=0, osl=128)
    assert r.bound == "memory" and math.isinf(r.critical_batch) and r.ttft_s == 0


def test_fp4_bits():
    assert llm.PRECISIONS["nvfp4"][0] == 4.5 and llm.PRECISIONS["mxfp4"][0] == 4.25
