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
    [{"rfilename":"tokenizer/qwen_embed_fp16.bin","size":1000},
    {"rfilename":"tokenizer/qwen_vocab.txt","size":100},
    {"rfilename":"tokenizer/qwen_merges.txt","size":100},
    {"rfilename":"tokenizer/qwen_special.txt","size":100}]})
  self.assertEqual(r["found_graphs"],21)
  self.assertTrue(r["graph_and_tokenizer_manifest_complete"])
  self.assertFalse(r["approved_for_app"])
  self.assertEqual(r["generation_graph_count"],12)
  self.assertEqual(r["generation_known_minimum_bytes"],1200+1300)
  self.assertFalse(r["generation_package_complete"])
  self.assertFalse(r["generation_real_image_created"])
  self.assertFalse(r["editing_real_image_created"])
  self.assertFalse(r["disk_size_limit_enforced"])
 def test_unknown_sizes_never_complete(self):
  r=mod.inspect({"siblings":[{"rfilename":"tokenizer/qwen_embed_fp16.bin"}]})
  self.assertFalse(r["graph_and_tokenizer_manifest_complete"])
  self.assertFalse(r["generation_package_complete"])
  self.assertIn("tokenizer/qwen_vocab.txt",r["missing_tokenizer_files"])
 def test_full_package_above_11_gb_not_rejected(self):
  names=([f"ke_enc{i}.tflite" for i in range(3)]+["kc_prep.tflite"]
    +[f"kc_double{i}.tflite" for i in range(2)]
    +[f"kc_single{i}.tflite" for i in range(4)]+["kc_final.tflite"]
    +["kce_prep.tflite"]+[f"kce_double{i}.tflite" for i in range(2)]
    +[f"kce_single{i}.tflite" for i in range(4)]+["kce_final.tflite"]
    +["kv_vae.tflite","kv_vae_enc.tflite"])
  r=mod.inspect({"siblings":[{"rfilename":n,"size":650*1024**2} for n in names]+
    [{"rfilename":"tokenizer/qwen_embed_fp16.bin","size":1000},
    {"rfilename":"tokenizer/qwen_vocab.txt","size":100},
    {"rfilename":"tokenizer/qwen_merges.txt","size":100},
    {"rfilename":"tokenizer/qwen_special.txt","size":100}]})
  self.assertGreater(r["known_minimum_gib"],11.4)
  self.assertFalse(r["disk_size_limit_enforced"])
  self.assertFalse(r["approved_for_app"])
 def test_invalid_size_and_duplicate_are_reported(self):
  r=mod.inspect({"siblings":[
   {"rfilename":"ke_enc0.tflite","size":0},
   {"rfilename":"ke_enc0.tflite","size":-1},
   {"rfilename":"kc_final.tflite","size":"not-an-integer"},
   {"rfilename":"kv_vae.tflite","size":True}]})
  self.assertIn("ke_enc0.tflite",r["duplicate_paths"])
  self.assertIn("ke_enc0.tflite",r["invalid_size_files"])
  self.assertIn("kc_final.tflite",r["invalid_size_files"])
  self.assertIn("kv_vae.tflite",r["invalid_size_files"])
  self.assertFalse(r["graph_and_tokenizer_manifest_complete"])
  self.assertFalse(r["approved_for_app"])
 def test_ambiguous_graph_paths_never_approved(self):
  r=mod.inspect({"siblings":[
   {"rfilename":"model/ke_enc0.tflite","size":100},
   {"rfilename":"other/ke_enc0.tflite","size":100}]})
  self.assertIn("ke_enc0.tflite",r["ambiguous_graphs"])
  self.assertEqual(r["found_graphs"],0)
  self.assertFalse(r["approved_for_app"])
 def test_reject_malformed_upstream_metadata(self):
  for payload in ({}, {"siblings":None}, {"siblings":"bad"}, {"siblings":[None]},
                  {"siblings":[{"rfilename":"../escape.tflite","size":1}]},
                  {"siblings":[{"rfilename":"/absolute.tflite","size":1}]}):
   with self.subTest(payload=payload),self.assertRaises(ValueError):
    mod.inspect(payload)
 def test_unknown_graph_size_is_reported(self):
  r=mod.inspect({"siblings":[{"rfilename":"kc_prep.tflite"}]})
  self.assertIn("kc_prep.tflite",r["unknown_graph_sizes"])
  self.assertFalse(r["graph_and_tokenizer_manifest_complete"])
 def test_empty_graph_size_is_invalid(self):
  r=mod.inspect({"siblings":[{"rfilename":"ke_enc0.tflite","size":0}]})
  self.assertIn("ke_enc0.tflite",r["invalid_size_files"])
  self.assertFalse(r["approved_for_app"])
if __name__=="__main__":unittest.main()
