import math

import pytest

from hbmperf import dashboard, hierarchy, trends
from hbmperf.validate import mlperf


def _synthetic_devices(doubling=2.5, n=20):
    return [{"name": f"d{i}", "year": 2012 + i * 0.6, "mem_bw": 1e11 * 2 ** ((i * 0.6) / doubling),
             "bf16_flops": 1e13 * 2 ** ((i * 0.6) / 2.0), "mem_bytes": 1e10 * 2 ** (i * 0.2)}
            for i in range(n)]


def test_fit_recovers_known_doubling_time():
    f = trends.fit_metric(_synthetic_devices(2.5), "mem_bw")
    assert f["doubling_years"] == pytest.approx(2.5, rel=0.02)


def test_intercept_exact_on_synthetic_line():
    devs = _synthetic_devices(2.0)
    target = devs[0]["mem_bw"] * 2**5  # five doublings after 2012 -> 2022
    f = trends.fit_metric(devs, "mem_bw", target)
    assert f["intercept_year"] == pytest.approx(2022.0, abs=0.01)
    assert f["intercept_year_ci"][0] <= 2022.0 <= f["intercept_year_ci"][1]


def test_insufficient_devices_and_out_of_range():
    assert trends.fit_metric(_synthetic_devices(n=5), "mem_bw")["verdict"] == \
        "INSUFFICIENT_EVIDENCE"
    f = trends.fit_metric(_synthetic_devices(), "mem_bw", 1e40)
    assert f["intercept_year"] is None and f["intercept_note"] == "no intercept within range"


def test_bootstrap_deterministic():
    devs = _synthetic_devices()
    assert trends.fit_metric(devs, "mem_bw", 1e13) == trends.fit_metric(devs, "mem_bw", 1e13)


def test_offload_bounds_ordered_and_zero_offload_matches_single_tier():
    rows = hierarchy.offload_sweep("llama-3.3-70b", "nvidia_h100_sxm", "fp8",
                                   "host_ddr5_pcie_gen5_x16", batch=8)
    assert rows[0]["tpot_s_lower"] == pytest.approx(rows[0]["tpot_s_upper"])
    for r in rows:
        assert r["tpot_s_lower"] <= r["tpot_s_upper"]
    caps = [r["max_feasible_batch"] for r in hierarchy.offload_sweep(
        "llama-3.3-70b", "nvidia_h100_sxm", "fp8", "host_ddr5_pcie_gen5_x16")]
    assert caps == sorted(caps)


def test_max_feasible_batch_zero_when_weights_do_not_fit():
    from hbmperf import catalog, llm

    m = catalog.get("models", "llama-3.3-70b")
    assert hierarchy.max_feasible_batch(m, llm.device("nvidia_l4", "fp8"), "fp8",
                                        max_context=2048) == 0
    assert math.isinf(hierarchy.max_feasible_batch(
        m, llm.device("nvidia_h200_sxm", "fp8"), "fp8", max_context=2048,
        kv_fraction_in_hbm=0.0))


def test_cluster_table_rows():
    rows = hierarchy.cluster_table()
    assert len(rows) == 13 and all(r["feasible"] for r in rows)


@pytest.mark.parametrize(("name", "acc_id"), [
    ("NVIDIA H200-SXM-141GB", "nvidia_h200_sxm"), ("NVIDIA B200-SXM-180GB", "nvidia_b200"),
    ("AMD Instinct MI325X 256GB HBM3E", "amd_mi325x"),
    ("NVIDIA GH200 Grace Hopper Superchip 144GB", "nvidia_gh200_hbm3e"),
    ("NVIDIA GB200", "nvidia_gb200_per_gpu"), ("NVIDIA H200-NVL-141GB", None),
    ("NVIDIA L40S", None)])
def test_mlperf_accelerator_mapping(name, acc_id):
    assert mlperf.map_accelerator(name) == acc_id


def test_mlperf_precision_parsing():
    assert mlperf.peak_precision("fp4, fp8") == "nvfp4"
    assert mlperf.peak_precision("fp16") == "bf16"
    assert mlperf.peak_precision("Dummy") == "fp8"


def test_dashboard_builds_every_panel(tmp_path):
    out = dashboard.build(tmp_path / "index.html")
    text = out.read_text()
    for pid in dashboard.PANEL_IDS:
        assert f"id='{pid}'" in text, pid
    assert "Sources" in text


def test_dashboard_missing_input_says_not_available(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    text = dashboard.build(tmp_path / "index.html").read_text()
    assert "not available" in text
