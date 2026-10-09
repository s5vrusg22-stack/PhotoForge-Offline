#!/usr/bin/env python3
"""Inspect FLUX LiteRT FlatBuffer graph signatures without loading model weights.

Usage: python tools/inspect_tflite_graphs.py /path/to/graphs --output graph-signatures.json
Requires: pip install flatbuffers tflite
"""
import argparse
import json
from pathlib import Path

def inspect(path):
    import tflite
    # Memory-map large (up to ~1 GB) graphs instead of duplicating weights in RAM.
    import mmap
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            return inspect_buffer(path, mapped, tflite)

def inspect_buffer(path, data, tflite):
    model = tflite.Model.GetRootAsModel(data, 0)
    graphs = []
    for subgraph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(subgraph_index)
        def tensor_at(index):
            t = graph.Tensors(index)
            name = t.Name()
            return {
                "index": int(index),
                "name": name.decode("utf-8", "replace") if name else "",
                "shape": [int(t.Shape(i)) for i in range(t.ShapeLength())],
                "shape_signature": [int(t.ShapeSignature(i)) for i in range(t.ShapeSignatureLength())],
                "type_code": int(t.Type()),
                "elements_x4_if_fp32": 4 * __import__("math").prod(
                    [int(t.Shape(i)) for i in range(t.ShapeLength())]
                ),
            }
        graphs.append({
            "subgraph": subgraph_index,
            "inputs": [tensor_at(graph.Inputs(i)) for i in range(graph.InputsLength())],
            "outputs": [tensor_at(graph.Outputs(i)) for i in range(graph.OutputsLength())],
        })
    return {"file": path.name, "size_bytes": path.stat().st_size, "subgraphs": graphs}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("graph-signatures.json"))
    args = parser.parse_args()
    paths = sorted(args.directory.rglob("*.tflite"))
    if not paths:
        raise SystemExit("No .tflite graph files found")
    result = {"graphs": [inspect(path) for path in paths]}
    expected = {*(f"ke_enc{i}.tflite" for i in range(3)),
                *(f"{mode}_{part}.tflite" for mode in ("kc", "kce")
                  for part in ("prep", "double0", "double1", "single0", "single1",
                               "single2", "single3", "final")),
                "kv_vae.tflite", "kv_vae_enc.tflite"}
    actual = [p.name for p in paths]
    result["inventory"] = {
        "expected_count": len(expected), "found_count": len(actual),
        "missing": sorted(expected - set(actual)),
        "unexpected": sorted(set(actual) - expected),
        "duplicate_names": sorted({name for name in actual if actual.count(name) > 1}),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for graph in result["graphs"]:
        for sub in graph["subgraphs"]:
            print(graph["file"], "subgraph", sub["subgraph"])
            for label in ("inputs", "outputs"):
                for item in sub[label]:
                    print(" ", label, item["index"], item["name"], item["shape"], "type", item["type_code"])
    print("Inventory:", json.dumps(result["inventory"], ensure_ascii=False))
    print("Wrote", args.output)
    if result["inventory"]["duplicate_names"]:
        raise SystemExit("Duplicate graph basenames found; inspect report")

if __name__ == "__main__":
    main()
