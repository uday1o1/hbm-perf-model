"""Minimal GGUF header reader: tensor names, shapes, types, and byte sizes.

Only the metadata section is parsed; tensor data is never read.
Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path

# ggml type id -> (name, block size in elements, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}

# metadata value type id -> struct format (None for variable-size types)
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
           12: "d"}
_STRING, _ARRAY = 8, 9
MAX_HEADER_ITEMS = 1_000_000  # defensive bound for untrusted files


class GGUFError(ValueError):
    pass


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise GGUFError("unexpected end of file")
        return struct.unpack("<" + fmt, data)[0]

    def string(self) -> str:
        n = self.read("Q")
        if n > 1 << 20:
            raise GGUFError(f"string length {n} too large")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALAR:
            return self.read(_SCALAR[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, n = self.read("I"), self.read("Q")
            if n > 10_000_000:
                raise GGUFError(f"array length {n} too large")
            items = [self.value(etype) for _ in range(n)]
            return items if n <= 64 else f"<array of {n}>"
        raise GGUFError(f"unknown metadata type {vtype}")


def read_header(path: Path) -> dict:
    """Return {"metadata": {...}, "tensors": [{name, shape, type, n_elements, n_bytes}]}."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise GGUFError(f"{path} is not a GGUF file")
        version = r.read("I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.read("Q"), r.read("Q")
        if n_tensors > MAX_HEADER_ITEMS or n_kv > MAX_HEADER_ITEMS:
            raise GGUFError("implausible tensor or metadata count")
        metadata = {}
        for _ in range(n_kv):
            key = r.string()
            metadata[key] = r.value(r.read("I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.read("Q") for _ in range(r.read("I"))]
            type_id = r.read("I")
            r.read("Q")  # data offset
            if type_id not in GGML_TYPES:
                raise GGUFError(f"tensor {name}: unsupported ggml type {type_id}")
            tname, block, block_bytes = GGML_TYPES[type_id]
            n = 1
            for d in dims:
                n *= d
            tensors.append({"name": name, "shape": dims, "type": tname, "n_elements": n,
                            "n_bytes": n // block * block_bytes})
    return {"version": version, "metadata": metadata, "tensors": tensors}


def decode_read_bytes(header: dict) -> dict:
    """Bytes a batch-1 decode step reads, by role, from the tensor inventory.

    The token embedding table is only gathered (one row) unless the file has no separate
    output tensor, in which case the LM head reads the whole (tied) table.
    """
    tensors = {t["name"]: t for t in header["tensors"]}
    total = sum(t["n_bytes"] for t in tensors.values())
    emb = tensors.get("token_embd.weight")
    has_output = "output.weight" in tensors
    emb_bytes = emb["n_bytes"] if emb else 0
    read = total - emb_bytes if has_output else total
    return {"file_tensor_bytes": total, "decode_read_bytes": read,
            "token_embd_bytes": emb_bytes, "separate_output": has_output}
