import hashlib,importlib.util,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location("prepare",Path(__file__).with_name("prepare_litert_weights.py"))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class FakeFile:
 def __init__(self,name,sha=None):
  self.rfilename=name
  self.size=5
  self.lfs={"sha256":sha} if sha else None

class PrepareTests(unittest.TestCase):
 def setUp(self):
  names=([f"ke_enc{i}.tflite" for i in range(3)]
   +["kc_prep.tflite"]+[f"kc_double{i}.tflite" for i in range(2)]
   +[f"kc_single{i}.tflite" for i in range(4)]+["kc_final.tflite"]
   +["kce_prep.tflite"]+[f"kce_double{i}.tflite" for i in range(2)]
   +[f"kce_single{i}.tflite" for i in range(4)]+["kce_final.tflite"]
   +["kv_vae.tflite","kv_vae_enc.tflite"]
   +["tokenizer/"+n for n in ("qwen_vocab.txt","qwen_merges.txt","qwen_special.txt","qwen_embed_fp16.bin")])
  self.digest=hashlib.sha256(b"valid").hexdigest()
  self.files=[FakeFile(n,self.digest) for n in names]
  self.info=types.SimpleNamespace(sha="a"*40,siblings=self.files)
 def fake_hub(self):
  api=types.SimpleNamespace(model_info=lambda *args,**kwargs:self.info)
  return patch.dict(sys.modules,{"huggingface_hub":types.SimpleNamespace(HfApi=lambda:api,snapshot_download=lambda **kwargs:kwargs["local_dir"])})
 def test_pinned_plan_has_both_edit_and_generation(self):
  with tempfile.TemporaryDirectory() as d,self.fake_hub():
   r=mod.prepare(d)
   self.assertEqual(r["graph_count"],21)
   self.assertFalse(r["downloaded"])
   self.assertEqual(r["revision"],"a"*40)
 def test_missing_upstream_hash_rejected(self):
  self.files[0].lfs=None
  with tempfile.TemporaryDirectory() as d,self.fake_hub(),self.assertRaisesRegex(RuntimeError,"SHA256 unavailable"):
   mod.prepare(d)
 def test_download_checks_upstream_sha256(self):
  with tempfile.TemporaryDirectory() as d,self.fake_hub():
   for f in self.files:
    p=Path(d)/f.rfilename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"valid")
   r=mod.prepare(d,download=True)
   self.assertTrue(r["verified"])
   self.assertEqual(len(__import__("json").loads((Path(d)/"trusted-assets.json").read_text())["assets"]),25)
 def test_corrupt_download_rejected(self):
  with tempfile.TemporaryDirectory() as d,self.fake_hub():
   for f in self.files:
    p=Path(d)/f.rfilename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"wrong")
   with self.assertRaisesRegex(RuntimeError,"SHA256 mismatch"):
    mod.prepare(d,download=True)
if __name__=="__main__":unittest.main()
