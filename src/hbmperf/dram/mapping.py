"""Pseudo-channel-local address mapping.

A byte address is split, above the burst offset bits, into fields listed most-significant first,
for example "ro,co,si,ba,bg" (bank group lowest, the default). Field widths come from the preset;
the row field takes the remaining high bits modulo the row count.
"""

from __future__ import annotations

from hbmperf.dram.timing import Preset

MAPPINGS = {
    "RoCoSiBaBg": "ro,co,si,ba,bg",  # default: consecutive bursts alternate bank groups
    "RoSiBaBgCo": "ro,si,ba,bg,co",  # column lowest: consecutive bursts stay in one bank
    "RoBaBgSiCo": "ro,ba,bg,si,co",
}
DEFAULT_MAPPING = "RoCoSiBaBg"


def _widths(p: Preset) -> dict[str, int]:
    return {
        "co": p.bursts_per_row.bit_length() - 1,
        "bg": p.bank_groups.bit_length() - 1,
        "ba": p.banks_per_group.bit_length() - 1,
        "si": p.sids.bit_length() - 1,
    }


class Mapper:
    def __init__(self, preset: Preset, mapping: str = DEFAULT_MAPPING):
        order = MAPPINGS.get(mapping, mapping).split(",")
        if sorted(order) != ["ba", "bg", "co", "ro", "si"] or order[0] != "ro":
            raise ValueError(f"mapping must list ro,co,si,ba,bg with ro first: {mapping!r}")
        self.p = preset
        self.order = order
        self.widths = _widths(preset)
        self.offset_bits = preset.burst_bytes.bit_length() - 1

    def decode(self, addr: int) -> tuple[int, int, int, int, int]:
        """Return (sid, bank_group, bank, row, column_burst)."""
        x = addr >> self.offset_bits
        out = {}
        for field in reversed(self.order[1:]):
            w = self.widths[field]
            out[field] = x & ((1 << w) - 1)
            x >>= w
        return out["si"], out["bg"], out["ba"], x % self.p.rows, out["co"]

    def encode(self, sid: int, bg: int, ba: int, row: int, co: int) -> int:
        vals = {"si": sid, "bg": bg, "ba": ba, "co": co}
        x = row
        for field in self.order[1:]:
            x = (x << self.widths[field]) | vals[field]
        return x << self.offset_bits
