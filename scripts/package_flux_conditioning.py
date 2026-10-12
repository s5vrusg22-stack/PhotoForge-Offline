#!/usr/bin/env python3
"""Package already-exported FLUX conditioning tensors with strict SHA-256 provenance.
This does not invent rotary/mask values: all five files must come from a verified exporter.
"""
import argparse, hashlib, json, math, struct
from pathlib import Path

NAMES=("encoder_mask_f32le.bin","encoder_rotary_cos_f32le.bin","encoder_rotary_sin_f32le.bin",
       "dit_rotary_cos_f32le.bin","dit_rotary_sin_f32le.bin")
def main():
    p=argparse.ArgumentParser(); p.add_argument("directory",type=Path); a=p.parse_args()
    hashes={}
    for name in NAMES:
        f=a.directory/name
        data=f.read_bytes()
        if not data or len(data)%4: raise SystemExit(f"invalid float32 file: {name}")
        for (v,) in struct.iter_unpack("<f",data):
            if not math.isfinite(v): raise SystemExit(f"nonfinite value in {name}")
        hashes[name]=hashlib.sha256(data).hexdigest()
    manifest={"schema":1,"dtype":"float32","byte_order":"little","sha256":hashes}
    (a.directory/"conditioning_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
