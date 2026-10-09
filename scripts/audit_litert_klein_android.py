#!/usr/bin/env python3
"""No-download audit for Android FLUX.2 Klein LiteRT edit package.
All outputs are metadata only: never claim on-device execution.
"""
import json,sys,urllib.request
from pathlib import Path
REPO="litert-community/FLUX.2-klein-4B-LiteRT"
LIMIT=10*1024**3
def fetch():
    request=urllib.request.Request("https://huggingface.co/api/models/"+REPO+"?blobs=true",
        headers={"User-Agent":"PhotoForge-LiteRT-Metadata-Audit/1.0"})
    with urllib.request.urlopen(request,timeout=45) as response:return json.load(response)
def inspect(info):
    files={}
    for item in info.get("siblings",[]):
        name=item.get("rfilename","")
        size=item.get("size")
        if size is None and isinstance(item.get("lfs"),dict):size=item["lfs"].get("size")
        files[name]=size
    # 3 text encoder + 8 generation DiT + 8 editing DiT + 2 VAE = 21 graphs.
    graph_names=([f"ke_enc{i}.tflite" for i in range(3)]
        +["kc_prep.tflite"]+[f"kc_double{i}.tflite" for i in range(2)]
        +[f"kc_single{i}.tflite" for i in range(4)]+["kc_final.tflite"]
        +["kce_prep.tflite"]+[f"kce_double{i}.tflite" for i in range(2)]
        +[f"kce_single{i}.tflite" for i in range(4)]+["kce_final.tflite"]
        +["kv_vae.tflite","kv_vae_enc.tflite"])
    found={name:next((p for p in files if p==name or p.endswith("/"+name)),None) for name in graph_names}
    missing=[name for name,p in found.items() if p is None]
    unknown=[name for name,p in found.items() if p is not None and not isinstance(files[p],int)]
    graph_bytes=sum(files[p] for p in found.values() if p and isinstance(files[p],int))
    tokenizer={p:size for p,size in files.items() if p.startswith("tokenizer/")}
    embed=[p for p in tokenizer if p.endswith("qwen_embed_fp16.bin")]
    tokenizer_missing=not tokenizer or not embed
    tokenizer_unknown=[p for p,size in tokenizer.items() if not isinstance(size,int)]
    tokenizer_bytes=sum(v for v in tokenizer.values() if isinstance(v,int))
    total=graph_bytes+tokenizer_bytes
    complete=(not missing and not unknown and not tokenizer_missing and not tokenizer_unknown)
    # The manifest may contain additional required sidecars. A model cannot be
    # approved until all host-side assets and a real Android run are validated.
    return {"repository":REPO,"license":info.get("cardData",{}).get("license") if isinstance(info.get("cardData"),dict) else None,
        "required_graphs":len(graph_names),"found_graphs":len(graph_names)-len(missing),
        "missing_graphs":missing,"unknown_graph_sizes":unknown,
        "tokenizer_files":tokenizer,"tokenizer_missing":tokenizer_missing,
        "tokenizer_unknown_sizes":tokenizer_unknown,
        "known_graph_bytes":graph_bytes,"known_tokenizer_bytes":tokenizer_bytes,
        "known_minimum_bytes":total,"known_minimum_gib":round(total/1024**3,3),
        "known_minimum_within_10gib":total<=LIMIT,
        "graph_and_tokenizer_manifest_complete":complete,
        "all_runtime_sidecars_verified":False,
        "android_gpu_full_pipeline_tested":False,"galaxy_s25_ultra_tested":False,
        "peak_ram_measured":False,"real_photo_edit_generated":False,
        "approved_for_app":False,
        "note":"21 graph filenames and tokenizer checked. Minimum size is NOT the full app footprint. Host code, runtime assets, license and real phone inference remain gates."}
def main():
    out=Path("reports/litert-klein-android-metadata.json");out.parent.mkdir(exist_ok=True,parents=True)
    try:result=inspect(fetch());result["status"]="METADATA_CHECKED"
    except Exception as exc:result={"status":"METADATA_ERROR","error":str(exc),"approved_for_app":False}
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False,indent=2)[:15000])
    if result["status"]=="METADATA_ERROR":sys.exit(1)
if __name__=="__main__":main()
