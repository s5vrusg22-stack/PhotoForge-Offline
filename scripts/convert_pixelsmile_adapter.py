#!/usr/bin/env python3
"""Convert PixelSmile safetensors tensors to an FP16 NPZ archive.

Tensor-format conversion only. This is NOT an Android inference model.
Reads one tensor at a time and verifies the stored values against the input.
"""
import argparse
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path

import numpy as np

DTYPES = {
    "F16": "<f2", "F32": "<f4", "F64": "<f8",
    "I8": "i1", "U8": "u1", "I16": "<i2", "U16": "<u2",
    "I32": "<i4", "U32": "<u4", "I64": "<i8", "U64": "<u8",
    "BOOL": "?",
}


def convert(raw, spec):
    dtype = spec["dtype"]
    shape = spec["shape"]
    if dtype == "BF16":
        if len(raw) % 2:
            raise ValueError("Invalid BF16 byte length")
        words = np.frombuffer(raw, dtype="<u2")
        result = (words.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    else:
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        result = np.frombuffer(raw, dtype=DTYPES[dtype])
        if result.dtype.kind == "f":
            result = result.astype(np.float16)
    if result.size != math.prod(shape):
        raise ValueError(f"Tensor size mismatch: {shape}")
    return result.reshape(shape)


def digest_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("models/expression/adapter/PixelSmile-preview.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("models/expression/pixelsmile-lora-fp16.npz"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = 0
    try:
        with args.input.open("rb") as source:
            header_bytes = source.read(8)
            if len(header_bytes) != 8:
                raise ValueError("Missing safetensors header")
            header_size = struct.unpack("<Q", header_bytes)[0]
            if header_size > args.input.stat().st_size - 8 or header_size > 100 * 1024 * 1024:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(source.read(header_size))
            data_start = 8 + header_size
            data_size = args.input.stat().st_size - data_start
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for key, spec in header.items():
                    if key == "__metadata__":
                        continue
                    start, end = spec["data_offsets"]
                    if not (0 <= start <= end <= data_size):
                        raise ValueError(f"Invalid offsets: {key}")
                    source.seek(data_start + start)
                    raw = source.read(end - start)
                    if len(raw) != end - start:
                        raise ValueError(f"Truncated tensor: {key}")
                    array = convert(raw, spec)
                    payload_io = io.BytesIO()
                    np.save(payload_io, array, allow_pickle=False)
                    payload = payload_io.getvalue()
                    entry = f"tensors/{len(manifest):06d}.npy"
                    archive.writestr(entry, payload)
                    manifest.append({"name": key, "path": entry, "shape": list(array.shape), "dtype": str(array.dtype)})
                    total += array.nbytes
                    print(f"Converted and verified tensor {len(manifest)}: {key}", flush=True)
                archive.writestr("manifest.json", json.dumps({"format": "pixelsmile-lora-npz-v1", "tensors": manifest}))
        with zipfile.ZipFile(args.output) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC check failed")
            recorded = json.loads(archive.read("manifest.json"))["tensors"]
            if not recorded or len(recorded) != len(manifest):
                raise ValueError("Tensor manifest mismatch")
            with args.input.open("rb") as source:
                source.seek(8)
                header_size = data_start - 8
                source.seek(8 + header_size)
                for item in recorded:
                    spec = header[item["name"]]
                    start, end = spec["data_offsets"]
                    source.seek(8 + header_size + start)
                    expected = convert(source.read(end - start), spec)
                    with archive.open(item["path"]) as stored:
                        recovered = np.load(stored, allow_pickle=False)
                    if recovered.shape != expected.shape or recovered.dtype != expected.dtype or not np.array_equal(recovered.tobytes(), expected.tobytes()):
                        raise ValueError(f"Archive tensor mismatch: {item['name']}")
                    del expected, recovered
        report = {
            "output": str(args.output),
            "tensor_count": len(manifest),
            "tensor_payload_bytes": total,
            "sha256": digest_file(args.output),
            "format_conversion": "complete",
            "android_inference": "NOT_IMPLEMENTED",
        }
        (args.output.parent / "adapter-conversion-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    except Exception:
        args.output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
