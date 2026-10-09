import hashlib,tempfile,unittest
from pathlib import Path
from verify_model_assets import verify,safe_path
class VerifyAssetsTest(unittest.TestCase):
 def test_good_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"weights.bin").write_bytes(b"valid")
   m={"assets":[{"path":"weights.bin","size":5,"sha256":hashlib.sha256(b"valid").hexdigest()}]}
   r=verify(p,m,free_bytes=100)
   self.assertTrue(r["ok"]);self.assertTrue(r["fresh_install_space_sufficient"])
 def test_corrupted_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"weights.bin").write_bytes(b"wrong")
   m={"assets":[{"path":"weights.bin","size":5,"sha256":hashlib.sha256(b"valid").hexdigest()}]}
   self.assertEqual(verify(p,m)["failures"][0]["error"],"sha256_mismatch")
 def test_missing_file(self):
  with tempfile.TemporaryDirectory() as d:
   m={"assets":[{"path":"missing","size":1,"sha256":"0"*64}]}
   self.assertEqual(verify(Path(d),m)["failures"][0]["error"],"missing")
 def test_disk_shortage(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"a").write_bytes(b"x")
   m={"assets":[{"path":"a","size":1,"sha256":hashlib.sha256(b"x").hexdigest()}]}
   self.assertFalse(verify(p,m,free_bytes=0)["fresh_install_space_sufficient"])
 def test_reject_path_traversal(self):
  with tempfile.TemporaryDirectory() as d:
   for name in ("../bad","/tmp/bad","a//b","a/./b","a\\b"):
    with self.assertRaises(ValueError):safe_path(Path(d),name)
 def test_reject_duplicate(self):
  with tempfile.TemporaryDirectory() as d:
   e={"path":"a","size":1,"sha256":"0"*64}
   with self.assertRaises(ValueError):verify(Path(d),{"assets":[e,e]})
if __name__=="__main__":unittest.main()
