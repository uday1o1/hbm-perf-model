import math

import numpy as np
import pytest

from hbmperf.dram import checker, sim, timing, traffic
from hbmperf.dram.mapping import MAPPINGS, Mapper

PRESETS = timing.all_ids()


def run(p, pattern, n=6000, **cfg):
    mapping = "RoSiBaBgCo" if pattern == "single_bank" else cfg.pop("mapping", "RoCoSiBaBg")
    a, w = traffic.addresses(p, pattern, n)
    return sim.simulate(p, traffic.closed_loop(a, w), sim.Config(mapping=mapping, **cfg))


@pytest.mark.parametrize("pid", PRESETS)
def test_isolated_latencies(pid):
    p = timing.load(pid)
    m = Mapper(p)
    reqs = [(0, m.encode(0, 0, 0, 5, 0), False),       # bank closed: miss
            (2000, m.encode(0, 0, 0, 5, 1), False),    # same row: hit
            (4000, m.encode(0, 0, 0, 7, 0), False)]    # other row open: conflict
    r = sim.simulate(p, iter(reqs), sim.Config(refresh=False))
    hit = p.n_cl_ck + p.n_bl_ck
    assert r.read_latencies_ck == [p.n_rcd_rd_ck + hit, hit, p.n_rp_ck + p.n_rcd_rd_ck + hit]
    assert (r.row_misses, r.row_hits, r.row_conflicts) == (1, 1, 1)


@pytest.mark.parametrize("pid", PRESETS)
def test_single_bank_conflicts_one_access_per_trc(pid):
    p = timing.load(pid)
    assert p.n_rc_ck >= p.n_rcd_rd_ck + p.n_rtp_ck + p.n_rp_ck
    m = Mapper(p)
    n = 400
    reqs = ((None, m.encode(0, 0, 0, i % p.rows, 0), False) for i in range(n))
    r = sim.simulate(p, reqs, sim.Config(refresh=False))
    # after the first ACT, one ACT per tRC; the last read completes tRCD + tCL + tBL later
    expected = (n - 1) * p.n_rc_ck + p.n_rcd_rd_ck + p.n_cl_ck + p.n_bl_ck
    assert r.cycles == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("pid", [p for p in PRESETS if timing.load(p).sids == 2])
def test_streaming_bank_group_interleaved_near_peak(pid):
    # a 64-entry queue hides the burst of activations at each row boundary for every preset;
    # at 32 entries HBM3E drops below 0.95 because tFAW spans more cycles (see RESULTS)
    r = run(timing.load(pid), "stream", 20000, refresh=False, queue_depth=64)
    assert r.efficiency >= 0.95


@pytest.mark.parametrize("pid", PRESETS)
def test_single_bank_streaming_limited_by_tccd_l(pid):
    p = timing.load(pid)
    r = run(p, "single_bank", 4000, refresh=False)
    assert r.efficiency <= p.n_bl_ck / p.n_ccd_l_ck + 0.02
    assert r.row_hit_rate > 0.9


@pytest.mark.parametrize("pid", PRESETS)
def test_refresh_overhead(pid):
    p = timing.load(pid)
    n = 60000
    base = run(p, "stream", n, refresh=False).efficiency
    with_ref = run(p, "stream", n, refresh=True).efficiency
    expected = base * (1 - (p.n_rfc_ck + p.n_rp_ck) / p.n_refi_ck)
    assert with_ref == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("pid", PRESETS)
def test_closed_page_random_obeys_faw_bound(pid):
    p = timing.load(pid)
    r = run(p, "random", 6000, page_policy="closed", log_commands=True)
    bytes_per_cycle = r.bytes / r.cycles
    assert bytes_per_cycle <= 4 * p.burst_bytes / p.n_faw_ck * 1.001
    assert checker.check_log(p, r.command_log)["max_acts_in_faw_window"] <= 4


@pytest.mark.parametrize("pid", PRESETS)
@pytest.mark.parametrize("pattern", traffic.PATTERNS)
@pytest.mark.parametrize("policy", ["open", "closed"])
def test_properties_and_checker_clean(pid, pattern, policy):
    p = timing.load(pid)
    r = run(p, pattern, 3000, page_policy=policy, log_commands=True)
    assert 0 < r.efficiency <= 1.0 + 1e-9
    assert r.requests == 3000
    assert min(r.read_latencies_ck or [p.n_cl_ck + p.n_bl_ck]) >= p.n_cl_ck + p.n_bl_ck
    report = checker.check_log(p, r.command_log)
    assert report["violations"] == 0, report["first"]


FAULTS = [({"faw"}, "random", "faw"), ({"rcd"}, "random", "rcd"), ({"rp"}, "random", "rp"),
          ({"ccd_l"}, "single_bank", "ccd_l"), ({"wtr"}, "mix", "wtr"), ({"rfc"}, "random", "rfc"),
          ({"ras"}, "random", "ras"), ({"rrd_s"}, "random", "rrd_s"), ({"rtw"}, "mix", "rtw"),
          ({"wr"}, "mix", "wr"), ({"ccd_s", "bl"}, "stream", "bl")]


@pytest.mark.parametrize(("disabled", "pattern", "expected"), FAULTS)
def test_checker_detects_seeded_faults(disabled, pattern, expected):
    p = timing.load("hbm3_6400")
    faulty = run(p, pattern, 8000, disabled=frozenset(disabled), log_commands=True)
    assert expected in checker.check_log(p, faulty.command_log)["by_constraint"]
    control = run(p, pattern, 8000, log_commands=True)  # negative control, same trace
    assert checker.check_log(p, control.command_log)["violations"] == 0


def test_determinism():
    p = timing.load("hbm3_6400")
    a = run(p, "llm_decode", 5000).to_dict()
    b = run(p, "llm_decode", 5000).to_dict()
    assert a == b


def test_littles_law_queue():
    p = timing.load("hbm3_6400")
    r = run(p, "mix", 20000)
    throughput = r.requests / r.cycles
    assert r.mean_queue_occupancy == pytest.approx(throughput * r.mean_queue_delay_ck, rel=0.02)


def test_halving_tck_doubles_streaming_bandwidth():
    from dataclasses import replace

    p = timing.load("hbm3_6400")
    fast = replace(p, tck_ps=p.tck_ps // 5 * 5 // 2)
    a = run(p, "stream", 10000, refresh=False).achieved_gb_per_s
    b = run(fast, "stream", 10000, refresh=False).achieved_gb_per_s
    assert b / a == pytest.approx(p.tck_ps / fast.tck_ps, rel=0.01)


@pytest.mark.parametrize("pid", PRESETS)
@pytest.mark.parametrize("mapping", list(MAPPINGS))
def test_mapping_round_trip(pid, mapping):
    p = timing.load(pid)
    m = Mapper(p, mapping)
    rng = np.random.default_rng(0)
    for _ in range(200):
        sid, bg, ba = (int(rng.integers(x)) for x in (p.sids, p.bank_groups, p.banks_per_group))
        row, co = int(rng.integers(p.rows)), int(rng.integers(p.bursts_per_row))
        assert m.decode(m.encode(sid, bg, ba, row, co)) == (sid, bg, ba, row, co)


def test_open_loop_latency_rises_with_load():
    p = timing.load("hbm3_6400")
    a, w = traffic.addresses(p, "random", 4000)
    lat = []
    for frac in (0.1, 0.25):
        rate = frac * p.peak_gb_per_s
        r = sim.simulate(p, traffic.open_loop(p, a, w, rate))
        assert r.achieved_gb_per_s == pytest.approx(rate, rel=0.1)
        lat.append(r.read_latency_ns_mean)
    assert lat[1] > lat[0]
    assert not math.isnan(lat[0])
