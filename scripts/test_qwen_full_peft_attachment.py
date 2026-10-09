#!/usr/bin/env python3
"""Attempt real PEFT LoRA attachment to the complete Qwen transformer architecture.
Qwen base weights are NEVER downloaded. Report failures honestly.
"""
import argparse, json, traceback
from pathlib import Path
import torch
from diffusers import QwenImageTransformer2DModel
from safetensors import safe_open
from peft import LoraConfig, inject_adapter_in_model

PAIRS=((".lora_A.weight",".lora_B.weight"),(".lora.down.weight",".lora.up.weight"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result={"status":"FAIL","stage":"init","base_weights_downloaded":False,"full_image_inference":False}
    try:
        cfg=json.loads(a.config.read_text())
        with torch.device("meta"):
            model=QwenImageTransformer2DModel.from_config(cfg)
        modules=dict(model.named_modules())
        with safe_open(str(a.adapter),framework="pt",device="cpu") as f:
            keys=set(f.keys())
            groups={}
            for key in keys:
                for low,high in PAIRS:
                    for suffix,side in ((low,"A"),(high,"B")):
                        if key.endswith(suffix):
                            name=key[:-len(suffix)].removeprefix("transformer.")
                            groups.setdefault(name,{})[side]=key
                            break
                    else:continue
                    break
            if len(groups)*2!=len(keys):raise RuntimeError("Unmatched or duplicate LoRA keys")
            rank_pattern={}
            for name,g in groups.items():
                if set(g)!={"A","B"}:raise RuntimeError("Missing pair "+name)
                if name not in modules or not isinstance(modules[name],torch.nn.Linear):
                    raise RuntimeError("Missing linear module "+name)
                ashape=f.get_slice(g["A"]).get_shape()
                bshape=f.get_slice(g["B"]).get_shape()
                if len(ashape)!=2 or len(bshape)!=2 or ashape[0]!=bshape[1]:
                    raise RuntimeError("Invalid LoRA shapes "+name)
                rank_pattern[name]=ashape[0]
            result.update(stage="inject_peft",target_modules=len(groups),tensors=len(keys))
            # Install LoRA layers on meta model: no base-weight allocation.
            peft_cfg=LoraConfig(r=max(rank_pattern.values()),lora_alpha=max(rank_pattern.values()),
                target_modules=list(groups),rank_pattern=rank_pattern,bias="none")
            inject_adapter_in_model(peft_cfg,model,adapter_name="pixelsmile")
            # Load actual adapter weights one module at a time to keep peak memory bounded.
            count=0
            with torch.no_grad():
                for name,g in groups.items():
                    module=dict(model.named_modules())[name]
                    for side,attr in (("A","lora_A"),("B","lora_B")):
                        src=f.get_tensor(g[side])
                        dest=getattr(module,attr)["pixelsmile"].weight
                        if tuple(src.shape)!=tuple(dest.shape):
                            raise RuntimeError(f"PEFT weight mismatch {name} {side}")
                        # Assign materialized CPU weights to adapter, not to meta base.
                        getattr(module,attr)["pixelsmile"].weight=torch.nn.Parameter(src.float(),requires_grad=False)
                        count+=1
            result.update(status="PASS",stage="complete",weights_attached=count,
                          note="Real PEFT adapters attached to every Qwen module on meta base. No full Qwen base weights, forward pass or image generation.")
    except Exception as e:
        result.update(error=str(e),traceback=traceback.format_exc(limit=5))
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)[:12000])
    if result["status"]!="PASS":raise SystemExit(1)

if __name__=="__main__":main()
