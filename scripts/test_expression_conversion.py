#!/usr/bin/env python3
"""Offline regression tests; no model download required."""
import json
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

def run_case(folder, values, dtype, expect_success=True):
    source = folder / "test.safetensors"
    target = folder / "converted.npz"
    if dtype == "BF16":
        payload = (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
    else:
        payload = np.asarray(values, dtype="<f4").tobytes()
    header = json.dumps({"weight": {"dtype": dtype, "shape": [len(values)], "data_offsets": [0, len(payload)]}}).encode()
    source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    command = [sys.executable, "scripts/convert_pixelsmile_adapter.py", "--input", str(source), "--output", str(target)]
    result = subprocess.run(command, capture_output=True, text=True)
    if expect_success:
        assert result.returncode == 0, result.stderr
        with zipfile.ZipFile(target) as archive:
            item = np.load(archive.open("tensors/000000.npy"), allow_pickle=False)
            assert item.dtype == np.float16
            assert np.array_equal(item, np.asarray(values, dtype=np.float16))
        assert json.loads((folder / "adapter-conversion-report.json").read_text())["tensor_count"] == 1
    else:
        assert result.returncode != 0, "Expected conversion rejection"
        assert not target.exists(), "Failed conversion left output file"

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    run_case(root, [1.0, -2.0, 0.5, 100.0], "BF16")
    run_case(root, [0.25, -1.0, 3.0], "F32")
    (root / "converted.npz").unlink()
    run_case(root, [100000.0], "BF16", expect_success=False)
print("PASS: BF16, F32, and overflow rejection regression tests")
