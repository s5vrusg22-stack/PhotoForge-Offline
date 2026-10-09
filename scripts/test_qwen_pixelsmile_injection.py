#!/usr/bin/env python3
"""Qwen module shape matching and real LoRA delta injection on isolated test layers.
No base weight download, no full pipeline execution, no image inference.
"""
import argparse, json, struct
from pathlib import Path
import torch
from diffusers import QwenImageTransformer2DModel
from safetensors import safe_open

PAIRS=((".lora_A.weight",".lora_B.weight"),(".lora.down.weight",".lora.up.weight"))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    config=json.loads(args.config.read_text())
    if config.get("_class_name")!="QwenImageTransformer2DModel":raise RuntimeError("Unexpected Qwen transformer")
    # Instantiate model structure without allocating actual Qwen weight storage.
    with torch.device("meta"):
        model=QwenImageTransformer2DModel.from_config(config)
    modules=dict(model.named_modules())
    with safe_open(str(args.adapter),framework="pt",device="cpu") as sf:
        keys=set(sf.keys())
        matched=[]; errors=[]; seen=set(); first=None
        for k in sorted(keys):
            for down,up in PAIRS:
                if k.endswith(down):
                    prefix=k[:-len(down)]
                    other=prefix+up
                    if other not in keys:
                        errors.append("Missing pair "+other);break
                    # Diffusers prefixes LoRA keys with transformer; model.named_modules does not.
                    module_name=prefix.removeprefix("transformer.")
                    target=modules.get(module_name)
                    if target is None:
                        errors.append("Missing Qwen module "+module_name);break
                    if not isinstance(target,torch.nn.Linear):
                        errors.append("Non-linear target "+module_name);break
                    a=sf.get_tensor(k);b=sf.get_tensor(other)
                    expected=(target.out_features,target.in_features)
                    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=expected[1] or b.shape[0]!=expected[0] or a.shape[0]!=b.shape[1]:
                        errors.append(f"Shape mismatch {module_name}: A={list(a.shape)} B={list(b.shape)} target={expected}")
                    else:
                        matched.append({"module":module_name,"rank":a.shape[0],"input":expected[1],"output":expected[0]})
                        if first is None: first=(module_name,a,b)
                    seen.add(k);seen.add(other)
                    break
        extra=keys-seen
        if extra: errors.append(f"Unmatched tensors: {len(extra)}, sample: {sorted(extra)[:5]}")
        injection={"status":"SKIPPED"}
        if first and not errors:
            name,a,b=first
            # Real CPU layer receives the saved LoRA A/B weights; test forward output changes.
            layer=torch.nn.Linear(a.shape[1],b.shape[0],bias=False,dtype=torch.float32)
            with torch.no_grad():layer.weight.zero_()
            x=torch.ones((1,a.shape[1]),dtype=torch.float32)
            with torch.no_grad():
                baseline=layer(x)
                # Standard LoRA: W_eff = W + scale * (B @ A), tested without full model weights.
                delta=b.float() @ a.float()
                layer.weight.add_(delta)
                after=layer(x)
            changed=not torch.equal(baseline,after)
            injection={"status":"PASS" if changed else "FAIL","module":name,"delta_nonzero":int(torch.count_nonzero(delta)),"forward_changed":changed}
            if not changed: errors.append("LoRA injection had no observable effect")
    result={"status":"PASS" if not errors else "FAIL","base_model":"Qwen/Qwen-Image-Edit-2511",
            "real_module_shapes_checked":len(matched),"adapter_tensors":len(keys),
            "injection_smoke_test":injection,"errors":errors[:60],
            "limits":"Qwen architecture instantiated on meta device; no base model weights loaded. Injection tests ONE isolated CPU Linear layer, not complete Qwen pipeline."}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)[:12000])
    if errors:raise SystemExit(1)
if __name__=="__main__":main()
