#!/usr/bin/env python3
"""Qwen Image Edit 2511 + PixelSmile end-to-end inference runner.

Requires a pre-provisioned full local Diffusers model directory and a GPU.
Does NOT download base weights. No fake images are produced on failure.
"""
import argparse
import json
import os
import time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",type=Path,required=True,help="Existing complete local Diffusers model directory")
    ap.add_argument("--adapter",type=Path,required=True)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--prompt",default="Edit only the person's expression to a gentle, natural smile. Preserve the person's identity, hairstyle, clothing, framing, lighting and background.")
    ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--report",type=Path,default=Path("inference-report.json"))
    args=ap.parse_args()
    report={"status":"FAIL","model_dir":str(args.model_dir),"output":str(args.output),
            "base_download_permitted":False,"image_generated":False}
    try:
        for path in (args.model_dir/"model_index.json",args.adapter,args.input):
            if not path.exists():raise FileNotFoundError(f"Required local file missing: {path}")
        if args.steps<1 or args.steps>100:raise ValueError("steps must be 1..100")
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        import torch
        from PIL import Image
        from diffusers import QwenImageEditPlusPipeline
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for this full-weight inference workflow; no CPU fallback")
        image=Image.open(args.input).convert("RGB")
        t=time.monotonic()
        pipe=QwenImageEditPlusPipeline.from_pretrained(
            str(args.model_dir),torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.to("cuda")
        # Qwen Image Edit Plus is a LoRA-capable Diffusers pipeline; preserve the
        # original adapter file and use Diffusers' own loader.
        pipe.load_lora_weights(str(args.adapter.parent),weight_name=args.adapter.name,
                               adapter_name="pixelsmile")
        pipe.set_adapters(["pixelsmile"],adapter_weights=[1.0])
        generator=torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.inference_mode():
            result=pipe(image=[image],prompt=args.prompt,num_inference_steps=args.steps,
                        generator=generator)
        if not result.images:raise RuntimeError("Pipeline returned no images")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result.images[0].save(args.output)
        report.update(status="PASS",image_generated=True,seconds=round(time.monotonic()-t,2),
                      width=result.images[0].width,height=result.images[0].height)
    except Exception as e:
        report.update(error=f"{type(e).__name__}: {e}")
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))
    if report["status"]!="PASS":raise SystemExit(1)

if __name__=="__main__":main()
