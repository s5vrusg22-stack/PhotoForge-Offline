#!/usr/bin/env python3
"""Prepare licensed PixelSmile research weights for an OFF-DEVICE compatibility study.

This does not convert Qwen-Image-Edit to Android/ONNX. It downloads weights into
a local directory for inspection. Large files; run on a workstation with storage.
"""
import argparse
import json
from pathlib import Path

BASE = "Qwen/Qwen-Image-Edit-2511"
ADAPTER = "PixelSmile/PixelSmile"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models/expression"))
    parser.add_argument("--download", action="store_true",
                        help="Explicitly download multi-GB model files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "expression-edit",
        "base_model": BASE,
        "adapter": ADAPTER,
        "status": "RESEARCH_ONLY_NOT_ANDROID_COMPATIBLE",
        "reference_image_support": "unverified",
        "required_before_release": [
            "license review", "quantization and mobile runtime conversion",
            "model tokenizer/text encoder support", "end-to-end Android inference",
            "identity preservation evaluation", "RAM/latency benchmark"
        ]
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.download:
        print("Wrote model manifest only. Use --download to obtain weights.")
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=BASE, local_dir=str(args.out / "base"))
    snapshot_download(repo_id=ADAPTER, local_dir=str(args.out / "adapter"))
    print("Downloaded research weights. No Android conversion performed.")


if __name__ == "__main__":
    main()
