#!/usr/bin/env python3
"""Offline model integrity and storage preflight. Never downloads model weights."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path

def safe_path(root,name):
    if not isinstance(name,str) or not name or name.startswith("/") or "\\" in name or any(p in ("","..",".") for p in name.split("/")):
        raise ValueError("unsafe relative asset path")
    root=root.resolve()
    target=(root/name).resolve()
    if not target.is_relative_to(root) or target==root:
        raise ValueError("asset escapes model directory")
    return target

def verify(root,manifest,free_bytes=None):
    if not isinstance(manifest,dict) or not isinstance(manifest.get("assets"),list) or not manifest["assets"]:
        raise ValueError("manifest requires nonempty assets")
    seen=set();failures=[];verified=[];required_bytes=0
    for entry in manifest["assets"]:
        if not isinstance(entry,dict):raise ValueError("malformed asset")
        name=entry.get("path");size=entry.get("size");digest=entry.get("sha256")
        path=safe_path(root,name)
        if name in seen:raise ValueError("duplicate asset path")
        seen.add(name)
        if type(size) is not int or size<=0:raise ValueError("invalid expected asset size")
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be 64 lowercase hex digits")
        required_bytes+=size
        if not path.is_file():
            failures.append({"path":name,"error":"missing"});continue
        if path.stat().st_size!=size:
            failures.append({"path":name,"error":"size_mismatch"});continue
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b""):h.update(chunk)
        if h.hexdigest()!=digest:failures.append({"path":name,"error":"sha256_mismatch"})
        else:verified.append(name)
    available=free_bytes if free_bytes is not None else shutil.disk_usage(root if root.exists() else root.parent).free
    # Installation preflight: reserve package size + 15% headroom for partial downloads.
    reserve=required_bytes+required_bytes//7
    return {"ok":not failures,"verified":verified,"failures":failures,
            "expected_package_bytes":required_bytes,"free_bytes":available,
            "fresh_install_space_sufficient":available>=reserve,
            "fresh_install_reserve_bytes":reserve,
            "note":"Space preflight is for fresh install, not in-place verification. GPU RAM is NOT checked."}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--report",default="reports/model-integrity.json")
    args=parser.parse_args()
    try:
        result=verify(Path(args.root),json.loads(Path(args.manifest).read_text()))
        code=0 if result["ok"] and result["fresh_install_space_sufficient"] else 2
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        result={"ok":False,"error":str(exc)};code=2
    output=Path(args.report);output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+".tmp")
    temp.write_text(json.dumps(result,indent=2))
    os.replace(temp,output)
    print(json.dumps(result,indent=2))
    return code
if __name__=="__main__":sys.exit(main())
