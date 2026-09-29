"""HBM timing presets from data/catalog/dram_timing.csv (timings in command-clock cycles)."""

from __future__ import annotations

from dataclasses import dataclass, fields

from hbmperf import catalog


@dataclass(frozen=True)
class Preset:
    id: str
    standard: str
    data_rate_mbit_per_s: int
    tck_ps: int
    pc_width_bits: int
    burst_length: int
    burst_bytes: int
    sids: int
    bank_groups: int
    banks_per_group: int
    rows: int
    row_bytes: int
    pcs_per_stack: int
    n_cl_ck: int
    n_cwl_ck: int
    n_rcd_rd_ck: int
    n_rcd_wr_ck: int
    n_rp_ck: int
    n_ras_ck: int
    n_wr_ck: int
    n_rtp_ck: int
    n_ccd_s_ck: int
    n_ccd_l_ck: int
    n_ccd_r_ck: int
    n_rrd_s_ck: int
    n_rrd_l_ck: int
    n_faw_ck: int
    n_wtr_s_ck: int
    n_wtr_l_ck: int
    n_rtw_ck: int
    n_bl_ck: int
    n_rfc_ck: int
    n_refi_ck: int

    @property
    def n_rc_ck(self) -> int:
        return self.n_ras_ck + self.n_rp_ck

    @property
    def n_banks(self) -> int:
        return self.sids * self.bank_groups * self.banks_per_group

    @property
    def bursts_per_row(self) -> int:
        return self.row_bytes // self.burst_bytes

    @property
    def pc_bytes(self) -> int:
        return self.n_banks * self.rows * self.row_bytes

    @property
    def peak_gb_per_s(self) -> float:
        """Peak pseudo-channel bandwidth: one burst every nBL cycles."""
        return self.burst_bytes / (self.n_bl_ck * self.tck_ps) * 1e3  # bytes/ps -> GB/s

    @property
    def stack_peak_gb_per_s(self) -> float:
        return self.peak_gb_per_s * self.pcs_per_stack

    def ns(self, cycles: float) -> float:
        return cycles * self.tck_ps / 1000

    def check(self) -> None:
        """Sanity checks the simulator relies on."""
        for name in ("sids", "bank_groups", "banks_per_group", "rows", "bursts_per_row"):
            v = getattr(self, name)
            if v & (v - 1):
                raise ValueError(f"{self.id}: {name}={v} must be a power of two")
        if self.row_bytes % self.burst_bytes:
            raise ValueError(f"{self.id}: row_bytes must be a multiple of burst_bytes")
        if self.burst_bytes * 8 != self.pc_width_bits * self.burst_length:
            raise ValueError(f"{self.id}: burst_bytes != pc_width_bits * burst_length / 8")


def load(preset_id: str) -> Preset:
    row = catalog.get("dram_timing", preset_id)
    p = Preset(**{f.name: row[f.name] for f in fields(Preset)})
    p.check()
    return p


def all_ids() -> list[str]:
    return sorted(catalog.load("dram_timing"))
