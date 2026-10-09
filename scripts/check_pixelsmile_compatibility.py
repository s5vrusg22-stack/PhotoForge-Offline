#!/usr/bin/env python3
"""Check PixelSmile LoRA tensor pairing and Qwen transformer module compatibility.

Metadata-only: does not claim successful loading into a live Qwen model.
"""
import argparse
import json
import struct
from pathlib import Path

SUFFIXES = (
    (".lora_A.weight", ".lora_B.weight", "peft"),
    (".lora.down.weight", ".lora.up.weight", "diffusers"),
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.adapter.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < 100_000_000:
            raise ValueError("Invalid safetensors header")
        header = json.loads(f.read(n))
    tensors = {k:v for k,v in header.items() if k != "__metadata__"}
    config = json.loads(args.base_config.read_text())
    if config.get("_class_name") != "QwenImageTransformer2DModel":
        raise ValueError("Unexpected transformer class")
    groups = {}
    errors = []
    for key, spec in tensors.items():
        found = False
        for down, up, convention in SUFFIXES:
            for suffix, side in ((down, "down"), (up, "up")):
                if key.endswith(suffix):
                    module = key[:-len(suffix)]
                    if not module.startswith("transformer."):
                        errors.append("Non-transformer module: " + key)
                    group = groups.setdefault(module, {"convention":convention})
                    if group["convention"] != convention or side in group:
                        errors.append("Conflicting/duplicate LoRA keys: " + key)
                    group[side] = (key, spec["shape"])
                    found = True
                    break
            if found: break
        if not found:
            errors.append("Unknown tensor naming convention: " + key)
    ranks = {}
    for module, group in groups.items():
        if "down" not in group or "up" not in group:
            errors.append("Missing LoRA pair: " + module)
            continue
        a = group["down"][1]; b = group["up"][1]
        if len(a) != 2 or len(b) != 2 or a[0] != b[1] or min(*a,*b) <= 0:
            errors.append(f"Incompatible pair dimensions: {module}: {a} / {b}")
        ranks[str(a[0])] = ranks.get(str(a[0]), 0) + 1
    nblocks = config.get("num_layers")
    block_ids = sorted({int(m.split(".")[2]) for m in groups if m.startswith("transformer.transformer_blocks.") and m.split(".")[2].isdigit()})
    if isinstance(nblocks, int) and any(i >= nblocks for i in block_ids):
        errors.append(f"LoRA block index exceeds Qwen config num_layers={nblocks}")
    result = {
        "status": "FAIL" if errors else "PASS_METADATA_ONLY",
        "base_class": config.get("_class_name"),
        "base_num_layers": nblocks,
        "adapter_tensors": len(tensors),
        "paired_modules": len(groups),
        "rank_distribution": ranks,
        "block_indices": block_ids,
        "conventions": sorted({g["convention"] for g in groups.values()}),
        "errors": errors[:100],
        "limitations": "No live module name/weight-shape lookup; no LoRA injection; no image generation; no Android inference."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)[:12000])
    if errors: raise SystemExit(1)

if __name__ == "__main__":
    main()
