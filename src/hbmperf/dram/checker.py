"""Independent command-log timing checker.

Replays (cycle, command, flat_bank, row) tuples and verifies every timing constraint from the
preset using command history, sharing no scheduling code with sim.py.
"""

from __future__ import annotations

from collections import defaultdict

from hbmperf.dram.timing import Preset

NEG = -(1 << 60)


def check_log(p: Preset, log: list[tuple]) -> dict:
    per_sid = p.bank_groups * p.banks_per_group
    rc = p.n_ras_ck + p.n_rp_ck
    wr_to_rd_s = p.n_cwl_ck + p.n_bl_ck + p.n_wtr_s_ck
    wr_to_rd_l = p.n_cwl_ck + p.n_bl_ck + p.n_wtr_l_ck
    wr_to_pre = p.n_cwl_ck + p.n_bl_ck + p.n_wr_ck

    last = defaultdict(lambda: NEG)  # (kind, scope) -> cycle
    open_row: dict[int, int] = {}
    acts: list[int] = []
    windows: list[tuple[int, int]] = []
    violations: list[dict] = []
    max_acts_in_faw = 0

    def need(name, t, prev, required, cmd):
        if t - prev < required:
            violations.append({"constraint": name, "cycle": t, "command": cmd,
                               "previous_cycle": prev, "gap": t - prev, "required": required})

    def state(name, t, cmd, msg):
        violations.append({"constraint": name, "cycle": t, "command": cmd, "detail": msg})

    for t, cmd, bank, row in sorted(log, key=lambda e: e[0]):
        sid = bank // per_sid if bank >= 0 else -1
        g = bank // p.banks_per_group if bank >= 0 else -1
        if cmd == "ACT":
            if bank in open_row:
                state("act_open_bank", t, cmd, f"bank {bank} already open")
            need("rp", t, last["pre", bank], p.n_rp_ck, cmd)
            need("rc", t, last["act", bank], rc, cmd)
            need("rrd_l", t, last["act_g", g], p.n_rrd_l_ck, cmd)
            need("rrd_s", t, last["act_pc", 0], p.n_rrd_s_ck, cmd)
            need("rfc", t, last["ref", 0], p.n_rfc_ck, cmd)
            if len(acts) >= 4:
                need("faw", t, acts[-4], p.n_faw_ck, cmd)
            acts.append(t)
            max_acts_in_faw = max(max_acts_in_faw,
                                  sum(1 for a in acts[-8:] if t - a < p.n_faw_ck))
            open_row[bank] = row
            last["act", bank] = last["act_g", g] = last["act_pc", 0] = t
        elif cmd == "PRE":
            if bank not in open_row:
                state("pre_closed_bank", t, cmd, f"bank {bank} not open")
            need("ras", t, last["act", bank], p.n_ras_ck, cmd)
            need("rtp", t, last["rd", bank], p.n_rtp_ck, cmd)
            need("wr", t, last["wr", bank], wr_to_pre, cmd)
            open_row.pop(bank, None)
            last["pre", bank] = last["pre_pc", 0] = t
        elif cmd in ("RD", "WR"):
            if open_row.get(bank) != row:
                state("cas_row", t, cmd, f"bank {bank} row {open_row.get(bank)} != {row}")
            if cmd == "RD":
                need("rcd", t, last["act", bank], p.n_rcd_rd_ck, cmd)
                need("ccd_l", t, last["rd_g", g], p.n_ccd_l_ck, cmd)
                for s in range(p.sids):
                    need("ccd_s" if s == sid else "ccd_r", t, last["rd_s", s],
                         p.n_ccd_s_ck if s == sid else p.n_ccd_r_ck, cmd)
                need("bl", t, last["rd_pc", 0], p.n_bl_ck, cmd)
                need("wtr", t, last["wr_g", g], wr_to_rd_l, cmd)
                need("wtr", t, last["wr_pc", 0], wr_to_rd_s, cmd)
                start = t + p.n_cl_ck
                last["rd", bank] = last["rd_g", g] = last["rd_s", sid] = last["rd_pc", 0] = t
            else:
                need("rcd", t, last["act", bank], p.n_rcd_wr_ck, cmd)
                need("ccd_l", t, last["wr_g", g], p.n_ccd_l_ck, cmd)
                need("ccd_s", t, last["wr_pc", 0], p.n_ccd_s_ck, cmd)
                need("bl", t, last["wr_pc", 0], p.n_bl_ck, cmd)
                need("rtw", t, last["rd_pc", 0], p.n_rtw_ck, cmd)
                start = t + p.n_cwl_ck
                last["wr", bank] = last["wr_g", g] = last["wr_pc", 0] = t
            end = start + p.n_bl_ck
            for s0, e0 in windows[-8:]:
                if start < e0 and s0 < end:
                    state("data_bus", t, cmd, f"window [{start},{end}) overlaps [{s0},{e0})")
            windows.append((start, end))
        elif cmd == "REF":
            if open_row:
                state("ref_open_bank", t, cmd, f"{len(open_row)} banks open")
            need("rp", t, max((last["pre", b] for b in range(p.n_banks)), default=NEG),
                 p.n_rp_ck, cmd)
            need("rc", t, max((last["act", b] for b in range(p.n_banks)), default=NEG),
                 rc, cmd)
            need("rfc", t, last["ref", 0], p.n_rfc_ck, cmd)
            last["ref", 0] = t
        else:
            state("unknown_command", t, cmd, cmd)

    by_name: dict[str, int] = defaultdict(int)
    for v in violations:
        by_name[v["constraint"]] += 1
    return {"commands": len(log), "violations": len(violations), "by_constraint": dict(by_name),
            "max_acts_in_faw_window": max_acts_in_faw, "first": violations[:5]}
