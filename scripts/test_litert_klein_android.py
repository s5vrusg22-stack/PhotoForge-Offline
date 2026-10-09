#!/usr/bin/env python3
"""Offline regression tests for LiteRT Android package manifest."""
import importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name("audit_litert_klein_android.py")
spec=importlib.util.spec_from_file_location("litert_audit",p)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ManifestTests(unittest.TestCase):
 def test_missing_graphs_never_approve(self):
  r=mod.inspect({"siblings":[{"rfilename":"ke_enc0.tflite","size":123}]})
  self.assertEqual(r["required_graphs"],21)
  self.assertEqual(r["found_graphs"],1)
  self.assertFalse(r["graph_and_tokenizer_manifest_complete"])
  self.assertFalse(r["approved_for_app"])
 def test_complete_graphs_still_need_device_validation(self):
  names=([f"ke_enc{i}.tflite" for i in range(3)]+["kc_prep.tflite"]
    +[f"kc_double{i}.tflite" for i in range(2)]
    +[f"kc_single{i}.tflite" for i in range(4)]+["kc_final.tflite"]
    +["kce_prep.tflite"]+[f"kce_double{i}.tflite" for i in range(2)]
    +[f"kce_single{i}.tflite" for i in range(4)]+["kce_final.tflite"]
    +["kv_vae.tflite","kv_vae_enc.tflite"])
  r=mod.inspect({"siblings":[{"rfilename":n,"size":100} for n in names]+
    [{"rfilename":"tokenizer/qwen_embed_fp16.bin","size":1000}]})
  self.assertEqual(r["found_graphs"],21)
  self.assertTrue(r["graph_and_tokenizer_manifest_complete"])
  self.assertFalse(r["approved_for_app"])
 def test_unknown_sizes_never_complete(self):
  r=mod.inspect({"siblings":[{"rfilename":"tokenizer/qwen_embed_fp16.bin"}]})
  self.assertFalse(r["graph_and_tokenizer_manifest_complete"])
if __name__=="__main__":unittest.main()
