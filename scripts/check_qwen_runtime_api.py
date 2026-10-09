#!/usr/bin/env python3
"""Runtime API verification for Qwen Image Edit 2511, without base weights."""
import inspect,json,traceback
from pathlib import Path
report={"status":"FAIL","base_weights_downloaded":False,"image_generated":False}
try:
 import torch,diffusers,peft
 from diffusers import QwenImageEditPlusPipeline,QwenImageTransformer2DModel
 from PIL import Image
 import run_qwen_pixelsmile_inference as runner
 report["versions"]={"torch":torch.__version__,"diffusers":diffusers.__version__,"peft":peft.__version__}
 pipeline_sig=inspect.signature(QwenImageEditPlusPipeline.__call__)
 params=pipeline_sig.parameters
 report["call_params"]=list(params)
 for name in ("image","prompt","num_inference_steps","generator"):
  if name not in params:raise RuntimeError(f"QwenImageEditPlusPipeline.__call__ missing {name}")
 for method in ("from_pretrained","load_lora_weights","set_adapters"):
  if not hasattr(QwenImageEditPlusPipeline,method):raise RuntimeError("Missing method "+method)
 source=inspect.getsource(runner.main)
 if 'local_files_only=True' not in source or 'HF_HUB_OFFLINE' not in source:
  raise RuntimeError("Offline safeguards missing")
 # Confirm local test image can be decoded without model weights.
 img=Image.new("RGB",(256,256),(110,130,150))
 Path("runtime-test-input.png").write_bytes(b"")
 img.save("runtime-test-input.png")
 assert Image.open("runtime-test-input.png").size==(256,256)
 report.update(status="PASS",runtime_imports="PASS",api_signatures="PASS",input_image_io="PASS",
               note="Runtime imports, API signatures and input image IO verified; full model weights and inference NOT executed.")
except Exception as e:
 report.update(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc(limit=6))
Path("qwen-runtime-report.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2)[:12000])
if report["status"]!="PASS":raise SystemExit(1)
