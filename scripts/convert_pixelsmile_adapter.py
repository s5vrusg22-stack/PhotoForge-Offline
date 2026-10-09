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
    with safe_open(str(args.input), framework="np") as source:
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for key in source.keys():
                array = source.get_tensor(key)
                if array.dtype.kind == "f":
                    array = array.astype(np.float16)
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
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {"output": str(args.output), "tensor_count": count, "tensor_payload_bytes": total, "sha256": digest, "format_conversion": "complete", "android_inference": "NOT_IMPLEMENTED"}
    (args.output.parent / "adapter-conversion-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
