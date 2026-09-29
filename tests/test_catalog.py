import shutil
from pathlib import Path

import pytest

from hbmperf import catalog


def test_all_catalogs_valid():
    counts = catalog.validate_all()
    assert counts["accelerators"] >= 20 and counts["models"] >= 9


def _copy(tmp_path: Path) -> Path:
    d = tmp_path / "catalog"
    shutil.copytree(catalog.CATALOG_DIR, d)
    return d


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


@pytest.mark.parametrize(
    ("file", "old", "new", "code"),
    [
        ("accelerators.csv", "nvidia_h100_sxm,NVIDIA,H100 SXM,2022,HBM3,80,",
         "nvidia_h100_sxm,NVIDIA,H100 SXM,2022,HBM3,80 GiB,", "bad_number"),
        ("accelerators.csv", "nvidia_h200_sxm,", "nvidia_h100_sxm,", "duplicate_id"),
        ("csp_instances.csv", ",nvidia_b200,", ",nvidia_b999,", "dangling_ref"),
        ("tiers.csv", "https://pcisig.com", "http://pcisig.com", "bad_source"),
        ("models.csv", "llama-3.1-8b,llama3,32,", "llama-3.1-8b,llama3,,", "missing"),
        ("models.csv", ",false,gated,", ",no,gated,", "bad_bool"),
        ("dram_timing.csv", "hbm3_6400,HBM3,6400,625,", "hbm3_6400,HBM3,6400,-625,",
         "non_positive"),
    ],
)
def test_catalog_faults_rejected(tmp_path, file, old, new, code):
    d = _copy(tmp_path)
    _edit(d / file, old, new)
    with pytest.raises(catalog.CatalogError) as e:
        catalog.validate_all(d)
    assert e.value.code == code


def test_clean_copy_passes(tmp_path):
    assert catalog.validate_all(_copy(tmp_path)) == catalog.validate_all()


def test_units_in_numeric_column_names():
    allowed = ("_gb", "_gb_per_s", "_gbit_per_s", "_tflops", "_tops", "_ck", "_ps", "_w",
               "_bits", "_bytes", "_ns", "_mbit_per_s")
    exempt = {"year", "hbm_stacks", "sids", "bank_groups", "banks_per_group", "rows",
              "burst_length", "pcs_per_stack", "accelerator_count", "n_layers", "d_model",
              "n_heads", "n_kv_heads", "head_dim", "d_ff", "vocab", "n_experts", "top_k",
              "n_shared_experts", "d_ff_expert", "published_params", "interface_bits_per_stack",
              "channels_per_stack", "pseudo_channels_per_stack", "max_stack_high"}
    for columns, _ in catalog.SCHEMAS.values():
        for col, typ in columns.items():
            if typ in (int, float) and col not in exempt:
                assert any(u + "_" in col + "_" for u in allowed), col
