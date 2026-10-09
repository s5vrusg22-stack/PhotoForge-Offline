#!/usr/bin/env python3
"""Prepare a trusted, pinned model manifest; download on a real networked machine.

No large model files are bundled into the Git repository. Uses huggingface_hub
snapshot_download with pinned commit and hashes every downloaded file.
"""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
REPO="litert-community/FLUX.2-klein-4B-LiteRT"

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def prepare(destination,revision=None,download=False):
    try:
        from huggingface_hub import HfApi,snapshot_download
    except ImportError as e:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub") from e
    api=HfApi()
    info=api.model_info(REPO,revision=revision,files_metadata=True)
    commit=info.sha
    siblings=info.siblings or []
    # Full model: keep BOTH kc_* (generation) and kce_* (editing).
    selected=[]
    for item in siblings:
        name=item.rfilename
        if name.endswith(".tflite") or name.startswith("tokenizer/"):
            if not name or name.startswith("/") or ".." in name.split("/") or "\\" in name:
                raise ValueError("Unsafe upstream file path")
            size=getattr(item,"size",None)
            if size is None and getattr(item,"lfs",None):
                size=getattr(item.lfs,"size",None) if not isinstance(item.lfs,dict) else item.lfs.get("size")
            selected.append({"path":name,"expected_bytes":size})
    graph_count=sum(e["path"].endswith(".tflite") for e in selected)
    if graph_count!=21:raise RuntimeError(f"Expected 21 graphs, found {graph_count}; upstream changed")
    if not any(e["path"].endswith("qwen_embed_fp16.bin") for e in selected):
        raise RuntimeError("Tokenizer embedding missing")
    destination=Path(destination)
    report={"repo":REPO,"revision":commit,"graph_count":graph_count,
            "files":selected,"downloaded":False,"verified":False}
    destination.mkdir(parents=True,exist_ok=True)
    report_path=destination/"model-download-plan.json"
    report_path.write_text(json.dumps(report,indent=2))
    if not download:return report
    required=sum(e["expected_bytes"] for e in selected if type(e["expected_bytes"]) is int)
    if any(type(e["expected_bytes"]) is not int or e["expected_bytes"]<=0 for e in selected):
        raise RuntimeError("Cannot safely preflight storage: missing file sizes")
    free=shutil.disk_usage(destination).free
    if free<required+required//7:
        raise RuntimeError(f"Insufficient disk: need {required+required//7}, available {free}")
    local=Path(snapshot_download(repo_id=REPO,revision=commit,local_dir=str(destination),
        allow_patterns=[e["path"] for e in selected],max_workers=2))
    manifest={"repository":REPO,"revision":commit,"assets":[]}
    for entry in selected:
        path=local/entry["path"]
        if not path.is_file() or path.stat().st_size!=entry["expected_bytes"]:
            raise RuntimeError("Missing or truncated file: "+entry["path"])
        manifest["assets"].append({"path":entry["path"],"size":entry["expected_bytes"],
                                   "sha256":sha256_file(path)})
    tmp=destination/"trusted-assets.json.tmp"
    tmp.write_text(json.dumps(manifest,indent=2))
    os.replace(tmp,destination/"trusted-assets.json")
    report.update({"downloaded":True,"verified":True,"trusted_manifest":"trusted-assets.json"})
    report_path.write_text(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--destination",required=True)
    p.add_argument("--revision",default=None)
    p.add_argument("--download",action="store_true",help="Explicitly download full weights")
    a=p.parse_args()
    try:
        result=prepare(a.destination,a.revision,a.download)
        print(json.dumps(result,indent=2));return 0
    except Exception as e:
        print("MODEL PREPARATION FAILED:",str(e),file=sys.stderr);return 2
if __name__=="__main__":sys.exit(main())
