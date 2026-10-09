#!/usr/bin/env python3
"""Inspect Qwen-Image-Edit / PixelSmile modules, validate adapter, and report conversion blockers.
No false claim of mobile inference or ONNX export.
"""
import json
from pathlib import Path
from safetensors import safe_open

root = Path("models/expression")
base = root / "base"
adapter = root / "adapter"
report = {"base_components": {}, "adapter": {}, "conversion": {"status": "blocked", "reason": ""}}
for file in sorted(base.rglob("config.json")):
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        report["base_components"][str(file.relative_to(base))] = {
            "class": data.get("_class_name"), "architectures": data.get("architectures"),
            "model_type": data.get("model_type"), "hidden_size": data.get("hidden_size"),
            "num_hidden_layers": data.get("num_hidden_layers")
        }
    except (ValueError, OSError) as e:
        report["base_components"][str(file.relative_to(base))] = {"error": str(e)}
files = sorted(adapter.rglob("*.safetensors"))
assert files, "PixelSmile adapter weights not found"
for file in files:
    with safe_open(str(file), framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        prefixes = {}
        for key in keys:
            prefix = key.split(".")[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        report["adapter"][file.name] = {
            "bytes": file.stat().st_size, "tensor_count": len(keys),
            "prefixes": prefixes, "example_keys": keys[:12]
        }
report["conversion"]["reason"] = (
    "PixelSmile is a LoRA adapter, not a standalone image model. "
    "Android conversion requires an implementation of the Qwen-Image-Edit-2511 "
    "transformer, VAE, text/vision encoder, scheduler, and adapter application "
    "with validated mobile backend and memory profile."
)
root.mkdir(parents=True, exist_ok=True)
(root / "architecture-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2)[:12000])
