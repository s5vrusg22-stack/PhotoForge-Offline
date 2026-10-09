#!/usr/bin/env python3
"""Offline unit tests for metadata audit; no network or model downloads."""
import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("audit_small_image_models.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class AuditTests(unittest.TestCase):
    def test_gguf_variants_are_alternatives(self):
        def gb(n): return n * 1024**3
        info = {"cardData":{"license":"apache-2.0"},"siblings":[
            {"rfilename":"klein-Q2.gguf","size":gb(2)},
            {"rfilename":"klein-Q4_K_M.gguf","size":gb(4)},
            {"rfilename":"klein-Q8.gguf","size":gb(8)}]}
        with patch.object(audit,"fetch",return_value=info):
            result=audit.inspect("test/model")
        self.assertEqual(result["status"],"METADATA_OK")
        self.assertEqual(len(result["gguf_variants"]),3)
        self.assertEqual(len(result["q4_variants"]),1)
        self.assertEqual(result["q4_variants"][0]["bytes"],gb(4))
        self.assertFalse(result["complete_pipeline_size_verified"])
        self.assertFalse(result["android_runtime_confirmed"])
        self.assertFalse(result["model_editing_confirmed"])

    def test_unknown_size_not_approved(self):
        with patch.object(audit,"fetch",return_value={"siblings":[{"rfilename":"model-Q4.gguf"}]}):
            result=audit.inspect("test/model")
        self.assertFalse(result["weight_size_within_10gib"])
        self.assertEqual(result["unknown_size_files"],["model-Q4.gguf"])

    def test_missing_repo_does_not_approve(self):
        with patch.object(audit,"fetch",side_effect=RuntimeError("not found")):
            result=audit.inspect("missing/model")
        self.assertEqual(result["status"],"METADATA_ERROR")
        self.assertFalse(result["android_runtime_confirmed"])

if __name__=="__main__":
    unittest.main()
