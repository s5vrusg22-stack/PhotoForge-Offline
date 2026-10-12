#!/usr/bin/env python3
"""Export exact 32-channel FLUX.2 Klein VAE latent normalization statistics."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from safetensors import safe_open

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("vae_stats_f32le.bin")); a=p.parse_args()
    with safe_open(str(a.checkpoint),framework="pt",device="cpu") as f:
        keys=set(f.keys())
        pairs=[
            ("bn.running_mean","bn.running_var"),
            ("vae.bn.running_mean","vae.bn.running_var"),
            ("latent_bn.running_mean","latent_bn.running_var"),
        ]
        pair=next((x for x in pairs if x[0] in keys and x[1] in keys),None)
        if pair is None:
            candidates=sorted(k for k in keys if "running_mean" in k or "running_var" in k)
            raise RuntimeError("Verified VAE BN keys not found. Candidates: "+repr(candidates[:50]))
        mean=f.get_tensor(pair[0]).float().cpu().numpy()
        var=f.get_tensor(pair[1]).float().cpu().numpy()
    if mean.shape!=(32,) or var.shape!=(32,) or not np.isfinite(mean).all() or not np.isfinite(var).all() or not (var>0).all():
        raise RuntimeError(f"Invalid VAE statistics shapes/values: {mean.shape}, {var.shape}")
    std=np.sqrt(var).astype("<f4"); mean=mean.astype("<f4")
    data=mean.tobytes()+std.tobytes()
    a.output.write_bytes(data)
    meta={"mean_key":pair[0],"variance_key":pair[1],"shape":[32],"sha256":hashlib.sha256(data).hexdigest(),"dtype":"float32","byte_order":"little"}
    a.output.with_suffix(a.output.suffix+".json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
