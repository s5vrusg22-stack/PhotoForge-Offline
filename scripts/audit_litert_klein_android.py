#!/usr/bin/env python3
"""No-download audit for Android FLUX.2 Klein LiteRT edit package.
All outputs are metadata only: never claim on-device execution.
"""
import json,sys,urllib.request
from pathlib import Path
REPO="litert-community/FLUX.2-klein-4B-LiteRT"
REFERENCE_BYTES=10*1024**3  # reference only, not an installation limit
def fetch():
    request=urllib.request.Request("https://huggingface.co/api/models/"+REPO+"?blobs=true",
        headers={"User-Agent":"PhotoForge-LiteRT-Metadata-Audit/1.0"})
    with urllib.request.urlopen(request,timeout=45) as response:return json.load(response)
def inspect(info):
    files={}
    duplicate_paths=[]
    invalid_size_files=[]
    for item in info.get("siblings",[]):
        name=item.get("rfilename","")
        size=item.get("size")
        if size is None and isinstance(item.get("lfs"),dict):size=item["lfs"].get("size")
        if name in files:duplicate_paths.append(name)
        if size is not None and (type(size) is not int or size <= 0):
            invalid_size_files.append(name)
            size=None
        files[name]=size
    # 3 text encoder + 8 generation DiT + 8 editing DiT + 2 VAE = 21 graphs.
    graph_names=([f"ke_enc{i}.tflite" for i in range(3)]
        +["kc_prep.tflite"]+[f"kc_double{i}.tflite" for i in range(2)]
        +[f"kc_single{i}.tflite" for i in range(4)]+["kc_final.tflite"]
        +["kce_prep.tflite"]+[f"kce_double{i}.tflite" for i in range(2)]
        +[f"kce_single{i}.tflite" for i in range(4)]+["kce_final.tflite"]
        +["kv_vae.tflite","kv_vae_enc.tflite"])
    matches={name:[p for p in files if p==name or p.endswith("/"+name)] for name in graph_names}
    ambiguous_graphs={name:paths for name,paths in matches.items() if len(paths)>1}
    found={name:(paths[0] if len(paths)==1 else None) for name,paths in matches.items()}
    missing=[name for name,p in found.items() if p is None]
    unknown=[name for name,p in found.items() if p is not None and not isinstance(files[p],int)]
    graph_bytes=sum(files[p] for p in found.values() if p and isinstance(files[p],int))
    tokenizer={p:size for p,size in files.items() if p.startswith("tokenizer/")}
    required_tokenizer=[
        "tokenizer/qwen_vocab.txt",
        "tokenizer/qwen_merges.txt",
        "tokenizer/qwen_special.txt",
        "tokenizer/qwen_embed_fp16.bin",
    ]
    missing_tokenizer=[name for name in required_tokenizer if name not in tokenizer]
    tokenizer_missing=bool(missing_tokenizer)
    tokenizer_unknown=[p for p,size in tokenizer.items() if not isinstance(size,int)]
    tokenizer_bytes=sum(v for v in tokenizer.values() if isinstance(v,int))
    total=graph_bytes+tokenizer_bytes
    # No runtime inference may be approved from repository metadata alone.
    # These host operations are explicitly required by the upstream model card.
    host_ops=["tokenizer","token_embeddings","causal_padding_mask",
              "text_rotary_tables","image_rotary_tables","flow_scheduler",
              "latent_pack_unpack","image_normalization","gpu_graph_lifecycle"]
    host_ops_verified=[]
    # Editing does not require the eight text-to-image kc_* DiT graphs.
    # This is a prospective edit-only package; real host compatibility
    # and completeness must still be checked before release.
    edit_names=[name for name in graph_names if not name.startswith("kc_")]
    edit_missing=[name for name in edit_names if found[name] is None]
    edit_unknown=[name for name in edit_names if found[name] is not None and not isinstance(files[found[name]],int)]
    edit_graph_bytes=sum(files[found[name]] for name in edit_names
                         if found[name] is not None and isinstance(files[found[name]],int))
    edit_total=edit_graph_bytes+tokenizer_bytes
    generation_names=[name for name in graph_names if name.startswith("ke_") or
        name.startswith("kc_") or name=="kv_vae.tflite"]
    generation_missing=[name for name in generation_names if found[name] is None]
    generation_unknown=[name for name in generation_names
        if found[name] is not None and not isinstance(files[found[name]],int)]
    generation_graph_bytes=sum(files[found[name]] for name in generation_names
        if found[name] is not None and isinstance(files[found[name]],int))
    generation_total=generation_graph_bytes+tokenizer_bytes
    complete=(not missing and not unknown and not tokenizer_missing and not tokenizer_unknown
              and not duplicate_paths and not invalid_size_files and not ambiguous_graphs)
    # The manifest may contain additional required sidecars. A model cannot be
    # approved until all host-side assets and a real Android run are validated.
    return {"repository":REPO,"license":info.get("cardData",{}).get("license") if isinstance(info.get("cardData"),dict) else None,
        "required_graphs":len(graph_names),"found_graphs":len(graph_names)-len(missing),
        "missing_graphs":missing,"unknown_graph_sizes":unknown,
        "duplicate_paths":duplicate_paths,"invalid_size_files":invalid_size_files,
        "host_operations_required":host_ops,
        "host_operations_verified":host_ops_verified,
        "host_pipeline_complete":False,
        "ambiguous_graphs":ambiguous_graphs,
        "tokenizer_files":tokenizer,"tokenizer_missing":tokenizer_missing,
        "missing_tokenizer_files":missing_tokenizer,
        "tokenizer_unknown_sizes":tokenizer_unknown,
        "known_graph_bytes":graph_bytes,"known_tokenizer_bytes":tokenizer_bytes,
        "known_minimum_bytes":total,"known_minimum_gib":round(total/1024**3,3),
        "edit_only_graph_count":len(edit_names),
        "edit_only_missing_graphs":edit_missing,
        "edit_only_unknown_graph_sizes":edit_unknown,
        "edit_only_known_minimum_bytes":edit_total,
        "edit_only_known_minimum_gib":round(edit_total/1024**3,3),
        "edit_only_known_minimum_within_10gib":edit_total<=REFERENCE_BYTES,
        "edit_only_package_complete":False,
        "generation_graph_count":len(generation_names),
        "generation_missing_graphs":generation_missing,
        "generation_unknown_graph_sizes":generation_unknown,
        "generation_known_minimum_bytes":generation_total,
        "generation_known_minimum_gib":round(generation_total/1024**3,3),
        "generation_package_complete":False,
        "generation_real_image_created":False,
        "editing_real_image_created":False,
        "disk_size_limit_enforced":False,
        "known_minimum_within_10gib":total<=REFERENCE_BYTES,
        "graph_and_tokenizer_manifest_complete":complete,
        "all_runtime_sidecars_verified":False,
        "android_gpu_full_pipeline_tested":False,"galaxy_s25_ultra_tested":False,
        "peak_ram_measured":False,"real_photo_edit_generated":False,
        "approved_for_app":False,
        "note":"21 full graphs, 13 edit-only, 12 generation-only (overlap shared). Both are metadata minima, NOT validated deployable packages. User allows >11.4 GB disk. RAM and on-device edit/generation still untested."}
def main():
    out=Path("reports/litert-klein-android-metadata.json");out.parent.mkdir(exist_ok=True,parents=True)
    try:result=inspect(fetch());result["status"]="METADATA_CHECKED"
    except Exception as exc:result={"status":"METADATA_ERROR","error":str(exc),"approved_for_app":False}
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False,indent=2)[:15000])
    if result["status"]=="METADATA_ERROR":sys.exit(1)
    # Missing published model files are actionable CI failures; host-side
    # operations remain release blockers even when metadata is complete.
    if (result["missing_graphs"] or result["missing_tokenizer_files"]
        or result["ambiguous_graphs"] or result["duplicate_paths"]
        or result["invalid_size_files"]):
        print("ERROR: upstream LiteRT graph manifest changed or is invalid",file=sys.stderr)
        sys.exit(2)
if __name__=="__main__":main()
