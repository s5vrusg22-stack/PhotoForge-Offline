#!/usr/bin/env python3
"""Export four genuine FLUX.2 Klein 4B projected timestep rows for Android.

pip install torch safetensors huggingface_hub
python scripts/export_flux_timestep_projection.py --sigmas 1 0.75 0.5 0.25 0 --output flux_temb_4x3072.bin
The checkpoint download is large; use --checkpoint to reuse an existing file.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

REPO = "black-forest-labs/FLUX.2-klein-4B"
FILENAME = "flux-2-klein-4b.safetensors"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("flux_temb_4x3072.bin"))
    parser.add_argument("--sigmas", nargs=5, type=float, required=True)
    args = parser.parse_args()
    sigmas = args.sigmas
    if not all(math.isfinite(s) and 0 <= s <= 1 for s in sigmas):
        parser.error("sigmas must be finite in [0,1]")
    if not all(a > b for a, b in zip(sigmas, sigmas[1:])) or sigmas[-1] != 0:
        parser.error("five strictly descending sigmas ending at zero required")
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download(repo_id=REPO, filename=FILENAME))
    else:
        checkpoint = args.checkpoint
    with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
        names = ("time_in.in_layer.weight", "time_in.out_layer.weight")
        if not all(name in f.keys() for name in names):
            raise RuntimeError("Official time_in weights missing from checkpoint")
        w1, w2 = (f.get_tensor(name) for name in names)
    if tuple(w1.shape) != (3072, 256) or tuple(w2.shape) != (3072, 3072):
        raise RuntimeError(f"Unexpected time_in weight shapes: {w1.shape}, {w2.shape}")
    # Official timestep_embedding(t, 256): t*1000, exp(-log(10000)*i/128),
    # cos frequencies then sin frequencies. MLPEmbedder: out(SiLU(in(x))).
    # Keep the checkpoint's own dtype for matmul to match the reference model.
    t = torch.tensor(sigmas[:4], dtype=torch.float32) * 1000
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
    phases = t[:, None] * frequencies[None, :]
    fourier = torch.cat((torch.cos(phases), torch.sin(phases)), dim=1)
    with torch.no_grad():
        projected = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(fourier.to(w1.dtype), w1)), w2
        ).float().contiguous()
    if tuple(projected.shape) != (4, 3072) or not torch.isfinite(projected).all():
        raise RuntimeError("Invalid projected timestep tensor")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    data = projected.numpy().astype("<f4", copy=False).tobytes(order="C")
    if len(data) != 49152:
        raise RuntimeError("Unexpected binary tensor size")
    output.write_bytes(data)
    metadata = {
        "source_repo": REPO, "checkpoint": FILENAME, "sigmas": sigmas,
        "sha256": hashlib.sha256(data).hexdigest(), "shape": [4, 3072],
        "dtype": "float32", "byte_order": "little"
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
