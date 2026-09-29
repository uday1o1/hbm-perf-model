import re
import struct
import subprocess

import pytest

from hbmperf import gguf, provenance
from hbmperf.provenance import ROOT


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, meta=None):
    """Write a GGUF v3 header (no tensor data) with (name, dims, type_id) tensors."""
    meta = meta or {"general.architecture": "llama"}
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += _s(k) + struct.pack("<I", 8) + _s(v)
    for name, dims, type_id in tensors:
        out += _s(name) + struct.pack("<I", len(dims))
        out += b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", type_id, 0)
    path.write_bytes(out)


def test_gguf_inventory_and_decode_bytes(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("token_embd.weight", [64, 100], 8), ("output.weight", [64, 100], 8),
                   ("blk.0.attn_q.weight", [64, 64], 1), ("output_norm.weight", [64], 0)])
    h = gguf.read_header(p)
    sizes = {t["name"]: t["n_bytes"] for t in h["tensors"]}
    assert sizes["token_embd.weight"] == 6400 // 32 * 34
    assert sizes["blk.0.attn_q.weight"] == 64 * 64 * 2
    d = gguf.decode_read_bytes(h)
    assert d["separate_output"] and d["decode_read_bytes"] == d["file_tensor_bytes"] - 6800


def test_gguf_tied_embeddings_read_whole_table(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("token_embd.weight", [64, 100], 1), ("output_norm.weight", [64], 0)])
    d = gguf.decode_read_bytes(gguf.read_header(p))
    assert not d["separate_output"] and d["decode_read_bytes"] == d["file_tensor_bytes"]


@pytest.mark.parametrize("payload", [b"NOPE" + b"\0" * 20, b"GGUF" + struct.pack("<I", 9),
                                     b"GGUF" + struct.pack("<IQQ", 3, 10**9, 0), b"GGUF"])
def test_gguf_rejects_malformed(tmp_path, payload):
    p = tmp_path / "bad.gguf"
    p.write_bytes(payload)
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(p)


def test_hash_mismatch_detected(tmp_path):
    f = tmp_path / "snap.csv"
    f.write_text("a,b\n1,2\n")
    good = provenance.sha256_file(f)
    provenance.verify_manifest_entry(f, good)
    f.write_text("a,b\n1,3\n")
    with pytest.raises(provenance.ProvenanceError):
        provenance.verify_manifest_entry(f, good)


def test_run_dir_no_clobber_and_confined(tmp_path):
    d = provenance.new_run_dir("x", {"k": 1}, base=tmp_path)
    assert d.parent.parent == tmp_path.resolve()
    with pytest.raises(provenance.ProvenanceError):
        provenance.new_run_dir("../../escape", {"k": 2}, base=tmp_path / "a")


def test_deterministic_json():
    assert provenance.dumps({"b": 1, "a": [1.5]}) == '{\n  "a": [\n    1.5\n  ],\n  "b": 1\n}\n'


def test_no_personal_paths_in_tracked_or_staged_text():
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    pattern = re.compile(r"/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/")
    offenders = []
    for name in files:
        path = ROOT / name
        if path.suffix in {".py", ".md", ".json", ".csv", ".c", ".toml", ".html", ".txt", ".sh"}:
            if pattern.search(path.read_text(errors="ignore")):
                offenders.append(name)
    assert not offenders
