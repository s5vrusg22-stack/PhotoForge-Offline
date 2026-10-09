#!/usr/bin/env python3
"""Convert PixelSmile LoRA safetensors to a portable float16 tensor bundle.

This is a REAL tensor-format conversion, NOT an executable mobile diffusion model.
The bundle includes all tensor names/shapes and individual numpy .npy payloads.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from safetensors import safe_open
import mmap
import struct
import numpy as np
import io

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("models/expression/adapter/PixelSmile-preview.safetensors"))
    ap.add_argument("--output", type=Path, default=Path("models/expression/pixelsmile-lora-fp16.npz"))
    args = ap.parse_args()
    assert args.input.is_file(), f"Missing weights: {args.input}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    manifest = []
    # Parse the safetensors header directly: NumPy safetensors cannot read BF16.
    with args.input.open("rb") as source_file:
        with mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            header_size = struct.unpack_from("<Q", mm, 0)[0]
            header = json.loads(mm[8:8 + header_size])
            data_start = 8 + header_size
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for key, spec in header.items():
                    if key == "__metadata__":
                        continue
                    start, end = spec["data_offsets"]
                    raw = memoryview(mm)[data_start + start:data_start + end]
                    dtype = spec["dtype"]
                    if dtype == "BF16":
                        # BF16 stores the upper 16 bits of IEEE float32.
                        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                        array = bits.view(np.float32).astype(np.float16)
                    else:
                        dtypes = {"F16":"<f2","F32":"<f4","F64":"<f8",
                                  "I8":"i1","U8":"u1","I16":"<i2","U16":"<u2",
                                  "I32":"<i4","U32":"<u4","I64":"<i8","U64":"<u8","BOOL":"?"}
                        if dtype not in dtypes:
                            raise ValueError(f"Unsupported safetensors dtype {dtype} for {key}")
                        array = np.frombuffer(raw, dtype=dtypes[dtype])
                        if array.dtype.kind == "f":
                            array = array.astype(np.float16)
                    array = array.reshape(spec["shape"])
                    del raw
                    buf = io.BytesIO()
                    np.save(buf, array, allow_pickle=False)
                    payload = buf.getvalue()
                    archive.writestr(f"tensors/{count:06d}.npy", payload)
                    manifest.append({"name": key, "path": f"tensors/{count:06d}.npy", "shape": list(array.shape), "dtype": str(array.dtype)})
                    total += array.nbytes
                    count += 1
            archive.writestr("manifest.json", json.dumps({"format": "pixelsmile-lora-npz-v1", "tensors": manifest}, ensure_ascii=False))
    with zipfile.ZipFile(args.output) as archive:
        assert len([n for n in archive.namelist() if n.startswith("tensors/")]) == count
        assert len(json.loads(archive.read("manifest.json"))["tensors"]) == count
    sha = hashlib.sha256()
    with args.output.open("rb") as bundle:
        for chunk in iter(lambda: bundle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    report = {"output": str(args.output), "tensor_count": count, "tensor_payload_bytes": total, "sha256": digest, "format_conversion": "complete", "android_inference": "NOT_IMPLEMENTED"}
    (args.output.parent / "adapter-conversion-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
