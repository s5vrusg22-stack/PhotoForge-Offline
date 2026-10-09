#!/usr/bin/env python3
"""Download published FLUX.2 Klein LiteRT graphs with resumable transfers.
Usage: python scripts/download_flux_litert.py --mode generation --output /path/to/models
Requires: pip install huggingface_hub
"""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "litert-community/FLUX.2-klein-4B-LiteRT"
ENCODER = [f"ke_enc{i}.tflite" for i in range(3)]
VAE = ["kv_vae.tflite"]
def required(mode):
    prefix = "kce" if mode == "editing" else "kc"
    graphs = [f"{prefix}_prep.tflite"]
    graphs += [f"{prefix}_double{i}.tflite" for i in range(2)]
    graphs += [f"{prefix}_single{i}.tflite" for i in range(4)]
    graphs += [f"{prefix}_final.tflite"]
    return ENCODER + graphs + VAE + (["kv_vae_enc.tflite"] if mode == "editing" else [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generation", "editing"], default="generation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.model_info(MODEL_ID, revision=args.revision, files_metadata=True)
    sha = info.sha
    entries = {item.rfilename: item for item in info.siblings}
    manifest = {"repo": MODEL_ID, "revision": sha, "mode": args.mode, "graphs": []}
    for name in required(args.mode):
        if name not in entries:
            raise RuntimeError(f"Missing graph in upstream repository: {name}")
        file = Path(hf_hub_download(repo_id=MODEL_ID, filename=name, revision=sha, local_dir=str(out)))
        digest = hashlib.sha256()
        with file.open("rb") as src:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if file.stat().st_size == 0:
            raise RuntimeError(f"Empty model file: {name}")
        manifest["graphs"].append({"file": name, "size": file.stat().st_size, "sha256": digest.hexdigest()})
        print(f"Verified {name}: {file.stat().st_size} bytes")
    (out / "flux_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Model graph acquisition complete. GPU inference NOT verified.")
if __name__ == "__main__":
    main()
