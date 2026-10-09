#!/usr/bin/env python3
"""Static preflight: validate image-edit inference wiring without model download."""
import ast
import json
from pathlib import Path
p=Path("scripts/run_qwen_pixelsmile_inference.py")
tree=ast.parse(p.read_text(encoding="utf-8"))
source=p.read_text(encoding="utf-8")
required=("QwenImageEditPlusPipeline","from_pretrained","local_files_only=True",
          "load_lora_weights","set_adapters","torch.inference_mode()",
          "result.images[0].save","HF_HUB_OFFLINE")
missing=[item for item in required if item not in source]
report={"status":"PASS" if not missing else "FAIL","syntax":"PASS",
        "required_wiring_checked":len(required),"missing":missing,
        "base_weights_downloaded":False,"image_generated":False,
        "limitations":"Static preflight only; GPU, full base weights, Diffusers runtime API and visual output not validated."}
Path("inference-preflight-report.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if missing:raise SystemExit(1)
