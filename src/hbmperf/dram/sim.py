"""Event-driven HBM pseudo-channel simulator with an FR-FCFS controller.

Time is an integer count of command-clock cycles. The controller may issue one row command
(ACT, PRE, REF) and one column command (RD, WR) per cycle, modeling HBM's separate row and
column command buses. When nothing can issue, time jumps to the earliest cycle at which some
command could become legal, so idle stretches cost nothing.

Simplifications (documented in docs/LIMITATIONS.md): one pseudo-channel simulated and scaled by
the pseudo-channel count; no command-bus bandwidth limit beyond one row and one column command
per cycle; all-bank refresh only; no power-down or ECC.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np

from hbmperf.dram.mapping import DEFAULT_MAPPING, Mapper
from hbmperf.dram.timing import Preset

NEVER = 1 << 62
CONSTRAINTS = {"rcd", "rp", "ras", "rtp", "wr", "ccd_s", "ccd_l", "ccd_r", "rrd_s", "rrd_l",
               "faw", "wtr", "rtw", "bl", "rfc"}


@dataclass
class Config:
    queue_depth: int = 32  # per read queue and per write queue
    mapping: str = DEFAULT_MAPPING
    page_policy: str = "open"  # "open" or "closed" (auto-precharge after each access)
    refresh: bool = True
    write_high: float = 0.8
    write_low: float = 0.2
    log_commands: bool = False
    disabled: frozenset[str] = frozenset()  # fault injection: constraints to ignore


@dataclass
class _Req:
    id: int
    arrival: int
    write: bool
    sid: int
    bg: int
    bank: int  # flat bank index within the pseudo-channel
    gbg: int  # flat bank-group index (sid, bg)
    row: int
    state: str = "hit"  # row-buffer outcome: hit, miss (bank closed), conflict (other row open)


@dataclass
class SimResult:
    preset: str
    requests: int
    reads: int
    writes: int
    cycles: int
    bytes: int
    achieved_gb_per_s: float
    peak_gb_per_s: float
    efficiency: float
    row_hits: int
    row_misses: int
    row_conflicts: int
    row_hit_rate: float
    read_latency_ns_mean: float
    read_latency_ns_p50: float
    read_latency_ns_p95: float
    read_latency_ns_p99: float
    mean_queue_occupancy: float
    mean_queue_delay_ck: float
    refreshes: int
    config: dict
    command_log: list = field(default_factory=list, repr=False)
    read_latencies_ck: list = field(default_factory=list, repr=False)

    def to_dict(self, with_log: bool = False) -> dict:
        d = asdict(self)
        d.pop("read_latencies_ck")
        if not with_log:
            d.pop("command_log")
        return d


def simulate(
    preset: Preset,
    traffic: Iterable[tuple[int | None, int, bool]],
    config: Config | None = None,
) -> SimResult:
    """Run `traffic`, an iterable of (arrival_cycle or None, byte_address, is_write).

    arrival None means closed-loop: the request enters as soon as its queue has space.
    Open-loop requests wait in an unbounded input FIFO until their queue has space; latency is
    measured from their arrival cycle.
    """
    cfg = config or Config()
    bad = set(cfg.disabled) - CONSTRAINTS
    if bad:
        raise ValueError(f"unknown constraints {sorted(bad)}")
    on = {c: c not in cfg.disabled for c in CONSTRAINTS}
    p = preset
    mapper = Mapper(p, cfg.mapping)
    n_banks, n_gbg = p.n_banks, p.sids * p.bank_groups
    per_sid = p.bank_groups * p.banks_per_group
    closed_page = cfg.page_policy == "closed"
    if cfg.page_policy not in ("open", "closed"):
        raise ValueError("page_policy must be 'open' or 'closed'")

    # timing constants (a disabled constraint contributes zero)
    def c(name, value):
        return value if on[name] else 0

    rcd_rd, rcd_wr = c("rcd", p.n_rcd_rd_ck), c("rcd", p.n_rcd_wr_ck)
    rp, ras, rtp = c("rp", p.n_rp_ck), c("ras", p.n_ras_ck), c("rtp", p.n_rtp_ck)
    wr_rec = c("wr", p.n_cwl_ck + p.n_bl_ck + p.n_wr_ck)
    rc = max(ras + rp, 0) if (on["ras"] or on["rp"]) else 0
    ccd_s, ccd_l = c("ccd_s", p.n_ccd_s_ck), c("ccd_l", p.n_ccd_l_ck)
    ccd_r = c("ccd_r", p.n_ccd_r_ck)
    rrd_s, rrd_l, faw = c("rrd_s", p.n_rrd_s_ck), c("rrd_l", p.n_rrd_l_ck), p.n_faw_ck
    wtr_s = c("wtr", p.n_cwl_ck + p.n_bl_ck + p.n_wtr_s_ck)
    wtr_l = c("wtr", p.n_cwl_ck + p.n_bl_ck + p.n_wtr_l_ck)
    rtw, bl, rfc = c("rtw", p.n_rtw_ck), c("bl", p.n_bl_ck), c("rfc", p.n_rfc_ck)
    rd_done, wr_done = p.n_cl_ck + p.n_bl_ck, p.n_cwl_ck + p.n_bl_ck

    # earliest legal cycle per command and scope
    act_b = [0] * n_banks
    pre_b = [0] * n_banks
    rd_b = [0] * n_banks
    wr_b = [0] * n_banks
    act_g = [0] * n_gbg
    rd_g = [0] * n_gbg
    wr_g = [0] * n_gbg
    rd_s = [0] * p.sids
    wr_s = [0] * p.sids
    act_pc = rd_pc = wr_pc = 0
    faw_q: deque[int] = deque(maxlen=4)
    open_row: list[int | None] = [None] * n_banks
    ref_due = p.n_refi_ck if cfg.refresh else NEVER
    ref_pending = False

    log: list[tuple] = []
    rq: list[_Req] = []
    wq: list[_Req] = []
    pending: deque = deque()  # open-loop requests not yet admitted
    source = iter(traffic)
    exhausted = False
    next_id = 0
    write_mode = False

    lat: list[int] = []
    delays_sum = 0
    hits = misses = conflicts = 0
    served = n_reads = n_writes = refreshes = 0
    first_arrival: int | None = None
    last_done = 0
    occ_area = 0
    now = 0
    q_cap = cfg.queue_depth

    def fetch():
        nonlocal exhausted
        if exhausted:
            return None
        try:
            return next(source)
        except StopIteration:
            exhausted = True
            return None

    while True:
        # --- admission ---
        while True:
            if not pending:
                item = fetch()
                if item is None:
                    break
                pending.append(item)
            arrival, addr, is_w = pending[0]
            q = wq if is_w else rq
            if len(q) >= q_cap or (arrival is not None and arrival > now):
                break
            pending.popleft()
            sid, bg, ba, row, _ = mapper.decode(int(addr))
            gbg = sid * p.bank_groups + bg
            bank = sid * per_sid + bg * p.banks_per_group + ba
            arr = now if arrival is None else int(arrival)
            if first_arrival is None:
                first_arrival = arr
            q.append(_Req(next_id, arr, bool(is_w), sid, bg, bank, gbg, row))
            next_id += 1

        if not rq and not wq and not pending and exhausted:
            break

        # --- write-drain mode ---
        if len(wq) >= cfg.write_high * q_cap or (not rq and wq):
            write_mode = True
        elif write_mode and (len(wq) <= cfg.write_low * q_cap) and rq:
            write_mode = False
        active = wq if write_mode else rq
        other = rq if write_mode else wq

        if now >= ref_due:
            ref_pending = True

        next_t = NEVER
        issued_row = False

        # --- refresh: precharge all, then REF ---
        if ref_pending:
            open_banks = [b for b in range(n_banks) if open_row[b] is not None]
            if open_banks:
                b = min(open_banks, key=lambda b: pre_b[b])
                t = pre_b[b]
                if t <= now:
                    open_row[b] = None
                    act_b[b] = max(act_b[b], now + rp)
                    issued_row = True
                    if cfg.log_commands:
                        log.append((now, "PRE", b, -1))
                else:
                    next_t = min(next_t, t)
            else:
                t = max(act_b)  # includes PRE->ACT (tRP) and ACT->ACT (tRC); same as ->REF
                if t <= now:
                    ref_pending = False
                    ref_due += p.n_refi_ck
                    refreshes += 1
                    end = now + rfc
                    for b in range(n_banks):
                        act_b[b] = max(act_b[b], end)
                    issued_row = True
                    if cfg.log_commands:
                        log.append((now, "REF", -1, -1))
                else:
                    next_t = min(next_t, t)

        # --- column command: oldest ready row hit (FR) ---
        col_req = None
        for r in active:
            b = r.bank
            if open_row[b] != r.row:
                continue
            if r.write:
                t = max(wr_b[b], wr_g[r.gbg], wr_s[r.sid], wr_pc)
            else:
                t = max(rd_b[b], rd_g[r.gbg], rd_s[r.sid], rd_pc)
            if t <= now:
                col_req = r
                break
            next_t = min(next_t, t)
        if col_req is not None:
            r, b = col_req, col_req.bank
            active.remove(r)
            delays_sum += now - r.arrival
            if r.write:
                n_writes += 1
                pre_b[b] = max(pre_b[b], now + wr_rec)
                wr_g[r.gbg] = max(wr_g[r.gbg], now + ccd_l)
                rd_g[r.gbg] = max(rd_g[r.gbg], now + wtr_l)
                for s in range(p.sids):
                    wr_s[s] = max(wr_s[s], now + ccd_s)
                wr_pc = max(wr_pc, now + bl)
                rd_pc = max(rd_pc, now + wtr_s)
                done = now + wr_done
            else:
                n_reads += 1
                pre_b[b] = max(pre_b[b], now + rtp)
                rd_g[r.gbg] = max(rd_g[r.gbg], now + ccd_l)
                for s in range(p.sids):
                    rd_s[s] = max(rd_s[s], now + (ccd_s if s == r.sid else ccd_r))
                rd_pc = max(rd_pc, now + bl)
                wr_pc = max(wr_pc, now + rtw)
                done = now + rd_done
                lat.append(done - r.arrival)
            served += p.burst_bytes
            last_done = max(last_done, done)
            if cfg.log_commands:
                log.append((now, "WR" if r.write else "RD", b, r.row))
            if r.state == "conflict":
                conflicts += 1
            elif r.state == "miss":
                misses += 1
            else:
                hits += 1
            if closed_page:
                # auto-precharge: the bank closes as soon as PRE is legal
                open_row[b] = None
                act_b[b] = max(act_b[b], pre_b[b] + rp)
                if cfg.log_commands:
                    log.append((pre_b[b], "PRE", b, -1))

        # --- row command: oldest request needing ACT or PRE (FCFS) ---
        if not issued_row and not ref_pending:
            hit_banks = {r.bank for r in active if open_row[r.bank] == r.row}
            for r in (*active, *other):
                b = r.bank
                cur = open_row[b]
                if cur == r.row:
                    continue
                if cur is None:
                    t = max(act_b[b], act_g[r.gbg], act_pc)
                    if on["faw"] and len(faw_q) == 4:
                        t = max(t, faw_q[0] + faw)
                    if t <= now:
                        open_row[b] = r.row
                        rd_b[b] = max(rd_b[b], now + rcd_rd)
                        wr_b[b] = max(wr_b[b], now + rcd_wr)
                        pre_b[b] = max(pre_b[b], now + ras)
                        act_b[b] = max(act_b[b], now + rc)
                        act_g[r.gbg] = max(act_g[r.gbg], now + rrd_l)
                        act_pc = max(act_pc, now + rrd_s)
                        faw_q.append(now)
                        if r.state == "hit":
                            r.state = "miss"
                        issued_row = True
                        if cfg.log_commands:
                            log.append((now, "ACT", b, r.row))
                        break
                    next_t = min(next_t, t)
                else:
                    if b in hit_banks:
                        continue  # protect an open row that still has pending hits
                    t = pre_b[b]
                    if t <= now:
                        open_row[b] = None
                        act_b[b] = max(act_b[b], now + rp)
                        r.state = "conflict"
                        issued_row = True
                        if cfg.log_commands:
                            log.append((now, "PRE", b, -1))
                        break
                    next_t = min(next_t, t)

        # --- advance time ---
        progressed = issued_row or col_req is not None
        if pending and pending[0][0] is not None and pending[0][0] > now:
            next_t = min(next_t, pending[0][0])  # next open-loop arrival
        if cfg.refresh and not ref_pending:
            next_t = min(next_t, ref_due)
        step_to = now + 1 if progressed else max(next_t, now + 1)
        if step_to >= NEVER:
            raise RuntimeError(f"simulator deadlock at cycle {now}: {len(rq)} reads, "
                               f"{len(wq)} writes, {len(pending)} pending")
        occ_area += (len(rq) + len(wq)) * (step_to - now)
        now = step_to

    start = first_arrival or 0
    span = max(last_done - start, 1)
    lat_arr = np.asarray(lat, dtype=float) if lat else np.zeros(1)
    total = n_reads + n_writes
    achieved = served / (span * p.tck_ps) * 1e3
    return SimResult(
        preset=p.id,
        requests=total,
        reads=n_reads,
        writes=n_writes,
        cycles=span,
        bytes=served,
        achieved_gb_per_s=achieved,
        peak_gb_per_s=p.peak_gb_per_s,
        efficiency=achieved / p.peak_gb_per_s,
        row_hits=hits,
        row_misses=misses,
        row_conflicts=conflicts,
        row_hit_rate=hits / total if total else 0.0,
        read_latency_ns_mean=p.ns(float(lat_arr.mean())),
        read_latency_ns_p50=p.ns(float(np.percentile(lat_arr, 50))),
        read_latency_ns_p95=p.ns(float(np.percentile(lat_arr, 95))),
        read_latency_ns_p99=p.ns(float(np.percentile(lat_arr, 99))),
        mean_queue_occupancy=occ_area / span,
        mean_queue_delay_ck=delays_sum / total if total else 0.0,
        refreshes=refreshes,
        config={**asdict(cfg), "disabled": sorted(cfg.disabled)},
        command_log=log,
        read_latencies_ck=lat,
    )
