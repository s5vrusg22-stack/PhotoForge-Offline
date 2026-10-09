#!/usr/bin/env python3
"""Independently validate archived PixelSmile source and converted tensors."""
import argparse
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
import numpy as np

EXPECTED_ORIGINAL_SHA256 = ""  # Set from verified source metadata when available.
EXPECTED_SOURCE_BYTES = 849543736
EXPECTED_TENSORS = 1440

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    for key in ("original", "bundle", "report", "output"):
        ap.add_argument("--" + key, required=True, type=Path)
    args = ap.parse_args()
    from importlib.machinery import SourceFileLoader
    # Do not import the converter: compare shape and dtype independently.
    original_hash = sha256(args.original)
    assert args.original.stat().st_size == EXPECTED_SOURCE_BYTES, "Unexpected source file size"
    with args.original.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        assert 0 < n < 100_000_000
        header = json.loads(f.read(n))
    specs = {k: v for k, v in header.items() if k != "__metadata__"}
    assert len(specs) == EXPECTED_TENSORS
    report = json.loads(args.report.read_text())
    bundle_hash = sha256(args.bundle)
    assert bundle_hash == report["sha256"], "Converted bundle SHA256 mismatch"
    checked = 0
    with zipfile.ZipFile(args.bundle) as archive:
        assert archive.testzip() is None, "ZIP CRC failure"
        items = json.loads(archive.read("manifest.json"))["tensors"]
        assert len(items) == EXPECTED_TENSORS == report["tensor_count"]
        assert len({item["name"] for item in items}) == len(items), "Duplicate tensor names"
        assert {item["name"] for item in items} == set(specs), "Tensor name mismatch"
        assert len({item["path"] for item in items}) == len(items), "Duplicate tensor paths"
        for item in items:
            spec = specs[item["name"]]
            assert item["shape"] == spec["shape"], "Shape mismatch: " + item["name"]
            expected_dtype = "float16" if spec["dtype"] in ("BF16", "F16", "F32", "F64") else str(np.dtype({
                "I8": "i1", "U8": "u1", "I16": "<i2", "U16": "<u2",
                "I32": "<i4", "U32": "<u4", "I64": "<i8", "U64": "<u8", "BOOL": "?"
            }[spec["dtype"]]))
            assert item["dtype"] == expected_dtype, "Manifest dtype mismatch: " + item["name"]
            with archive.open(item["path"]) as f:
                arr = np.load(f, allow_pickle=False)
            assert list(arr.shape) == spec["shape"], "Stored shape mismatch: " + item["name"]
            assert str(arr.dtype) == expected_dtype, "Stored dtype mismatch: " + item["name"]
            assert arr.size == math.prod(spec["shape"]), "Element count mismatch"
            checked += 1
    result = {"result": "PASS", "original_sha256": original_hash,
              "original_bytes": args.original.stat().st_size,
              "converted_sha256": bundle_hash, "verified_tensors": checked,
              "shape_dtype": "PASS", "source_hash_note": "SHA256 measured from restored original; source authenticity requires an external trusted digest"}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
