#!/usr/bin/env python3
"""Metadata-only offline image-edit model screening. Never downloads weights."""
import json, os, sys, urllib.request
from pathlib import Path

REPOS = [
    "black-forest-labs/FLUX.2-klein-4B",
    "tonera/FLUX.2-klein-4B-fp8-diffusers",
    "MXKA/FLUX.2-klein-4B-GGUF",
]
LIMIT = 10 * 1024**3
def fetch(repo):
    url = "https://huggingface.co/api/models/" + repo + "?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent":"PhotoForge-Metadata-Audit/1.0"})
    with urllib.request.urlopen(req, timeout=35) as response:
        return json.load(response)
def inspect(repo):
    entry = {"repo":repo,"status":"UNVERIFIED","files":[],"total_known_weight_bytes":0,
             "total_known_bytes":0,"unknown_size_files":[],"model_editing_confirmed":False,
             "android_runtime_confirmed":False,"peak_ram_confirmed":False}
    try:
        info=fetch(repo)
        entry["license"]=info.get("cardData",{}).get("license") if isinstance(info.get("cardData"),dict) else None
        entry["pipeline_tag"]=info.get("pipeline_tag")
        siblings=info.get("siblings",[])
        for f in siblings:
            name=f.get("rfilename","")
            size=f.get("size")
            if size is None and isinstance(f.get("lfs"),dict):size=f["lfs"].get("size")
            if name.endswith((".safetensors",".gguf",".bin",".pt",".pth",".onnx",".tflite")):
                entry["files"].append({"path":name,"bytes":size})
                if size is None:entry["unknown_size_files"].append(name)
                else:entry["total_known_weight_bytes"]+=size
            if isinstance(size,int):entry["total_known_bytes"]+=size
        entry["weight_size_within_10gib"]=(not entry["unknown_size_files"] and bool(entry["files"]) and entry["total_known_weight_bytes"]<=LIMIT)
        entry["status"]="METADATA_OK"
        entry["note"]="No weights downloaded. Weight sizes exclude unreported and runtime assets. Android and real image editing untested."
    except Exception as e:
        entry["status"]="METADATA_ERROR";entry["error"]=str(e)
    return entry
def main():
    report={"policy":"offline-first","max_complete_package_bytes":LIMIT,
            "weights_downloaded":False,"image_generated":False,
            "android_tested":False,"candidate_approved":False,
            "candidates":[inspect(repo) for repo in REPOS]}
    out=Path("reports/small-model-metadata.json");out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for c in report["candidates"]:
        print(c["repo"],c["status"],"known_weight_GiB",round(c["total_known_weight_bytes"]/1024**3,2),
              "unknown_files",len(c["unknown_size_files"]),"within_limit",c.get("weight_size_within_10gib"))
        if "error" in c:print("ERROR:",c["error"])
    print("NO MODEL APPROVED: full package, Android backend, peak RAM and real edit remain unverified")
    if all(c["status"]=="METADATA_ERROR" for c in report["candidates"]):sys.exit(1)
if __name__=="__main__":main()
