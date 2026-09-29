"""Load and validate the cited CSV catalogs in data/catalog/.

Every numeric column name carries its unit suffix (see data/catalog/SCHEMA.md).
Blank cells mean "not published" and load as None.
"""

from __future__ import annotations

import csv
from functools import cache
from pathlib import Path

from hbmperf.provenance import ROOT

CATALOG_DIR = ROOT / "data" / "catalog"

S, I, F, B = str, int, float, bool  # column types

# name -> (columns with types, required columns)
SCHEMAS: dict[str, tuple[dict[str, type], set[str]]] = {
    "accelerators": (
        dict(id=S, vendor=S, name=S, year=I, memory_type=S, memory_gb=F, mem_bw_gb_per_s=F,
             hbm_stacks=I, bf16_dense_tflops=F, fp8_dense_tflops=F, fp4_dense_tflops=F,
             int8_dense_tops=F, scaleup_bw_gb_per_s=F, tdp_w=F, source_url=S, notes=S),
        {"id", "vendor", "name", "year", "memory_type", "memory_gb", "mem_bw_gb_per_s",
         "source_url"},
    ),
    "memory_standards": (
        dict(id=S, standard=S, year=I, interface_bits_per_stack=I, max_pin_rate_gbit_per_s=F,
             peak_bw_per_stack_gb_per_s=F, channels_per_stack=I, pseudo_channels_per_stack=I,
             max_stack_high=I, max_capacity_gb_per_stack=F, source_url=S, notes=S),
        {"id", "standard", "year", "interface_bits_per_stack", "max_pin_rate_gbit_per_s",
         "source_url"},
    ),
    "dram_timing": (
        dict(id=S, standard=S, data_rate_mbit_per_s=I, tck_ps=I, pc_width_bits=I,
             burst_length=I, burst_bytes=I, sids=I, bank_groups=I, banks_per_group=I, rows=I,
             row_bytes=I, pcs_per_stack=I, n_cl_ck=I, n_cwl_ck=I, n_rcd_rd_ck=I, n_rcd_wr_ck=I,
             n_rp_ck=I, n_ras_ck=I, n_wr_ck=I, n_rtp_ck=I, n_ccd_s_ck=I, n_ccd_l_ck=I,
             n_ccd_r_ck=I, n_rrd_s_ck=I, n_rrd_l_ck=I, n_faw_ck=I, n_wtr_s_ck=I, n_wtr_l_ck=I,
             n_rtw_ck=I, n_bl_ck=I, n_rfc_ck=I, n_refi_ck=I, source_url=S, source_ref=S,
             notes=S),
        None,  # every column except notes is required
    ),
    "models": (
        dict(id=S, family=S, n_layers=I, d_model=I, n_heads=I, n_kv_heads=I, head_dim=I,
             d_ff=I, vocab=I, tied_embeddings=B, mlp_type=S, n_experts=I, top_k=I,
             n_shared_experts=I, d_ff_expert=I, qkv_bias=B, published_params=I, source_url=S,
             notes=S),
        {"id", "family", "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "vocab",
         "tied_embeddings", "mlp_type", "qkv_bias", "published_params", "source_url"},
    ),
    "tiers": (
        dict(id=S, name=S, link=S, peak_bw_gb_per_s=F, measured_bw_gb_per_s=F, latency_ns=F,
             capacity_gb=F, source_url=S, notes=S),
        {"id", "name", "link", "peak_bw_gb_per_s", "source_url"},
    ),
    "csp_instances": (
        dict(id=S, csp=S, instance=S, accelerator_id=S, accelerator_count=I, host_mem_gb=F,
             scaleout_gbit_per_s=F, source_url=S, notes=S),
        {"id", "csp", "instance", "accelerator_id", "accelerator_count", "source_url"},
    ),
}


class CatalogError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _parse(value: str, typ: type, where: str):
    if typ is S:
        return value
    if typ is B:
        if value.lower() not in ("true", "false"):
            raise CatalogError("bad_bool", f"{where}: expected true/false, got {value!r}")
        return value.lower() == "true"
    try:
        num = typ(value)
    except ValueError:
        raise CatalogError("bad_number", f"{where}: cannot parse {value!r}") from None
    if num <= 0:
        raise CatalogError("non_positive", f"{where}: expected positive value, got {value!r}")
    return num


def load_csv(name: str, path: Path | None = None) -> dict[str, dict]:
    """Load one catalog as {id: row}; raise CatalogError on any schema violation."""
    columns, required = SCHEMAS[name]
    if required is None:
        required = set(columns) - {"notes"}
    path = path or CATALOG_DIR / f"{name}.csv"
    rows: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(columns):
            raise CatalogError(
                "bad_header", f"{path.name}: expected {list(columns)}, got {reader.fieldnames}"
            )
        for lineno, raw in enumerate(reader, start=2):
            where = f"{path.name}:{lineno}"
            row = {}
            for col, typ in columns.items():
                value = (raw[col] or "").strip()
                if not value:
                    if col in required:
                        raise CatalogError("missing", f"{where}: required column {col} is blank")
                    row[col] = None
                    continue
                row[col] = _parse(value, typ, f"{where} {col}")
            if not all(u.startswith("https://") for u in row["source_url"].split(" ; ")):
                raise CatalogError("bad_source", f"{where}: source_url must be https")
            if row["id"] in rows:
                raise CatalogError("duplicate_id", f"{where}: duplicate id {row['id']}")
            rows[row["id"]] = row
    return rows


@cache
def load(name: str) -> dict[str, dict]:
    return load_csv(name)


def validate_all(catalog_dir: Path | None = None) -> dict[str, int]:
    """Load every catalog and check cross references; return row counts."""
    d = catalog_dir or CATALOG_DIR
    loaded = {name: load_csv(name, d / f"{name}.csv") for name in SCHEMAS}
    for row in loaded["csp_instances"].values():
        if row["accelerator_id"] not in loaded["accelerators"]:
            raise CatalogError(
                "dangling_ref", f"csp_instances {row['id']}: unknown {row['accelerator_id']}"
            )
    return {name: len(rows) for name, rows in loaded.items()}


def get(name: str, id_: str) -> dict:
    rows = load(name)
    if id_ not in rows:
        raise CatalogError("unknown_id", f"{name}: unknown id {id_!r}; known: {sorted(rows)}")
    return rows[id_]
