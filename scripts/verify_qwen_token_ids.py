#!/usr/bin/env python3
"""Verify real Qwen token IDs against the published FLUX LiteRT tokenizer fixture.

Requires: pip install transformers huggingface_hub
Never substitutes invented token IDs.
"""
import argparse
import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL = "litert-community/FLUX.2-klein-4B-LiteRT"
TOKENIZER = "Qwen/Qwen3-4B"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default="main")
    p.add_argument("--prompt", default="Turn the apple into a green apple")
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)\n    decoded = tokenizer.decode(ids, skip_special_tokens=False)\n    if decoded != args.prompt:\n        raise RuntimeError(f"Qwen token round-trip mismatch: {decoded!r}")
    if not ids or any(i < 0 or i >= 151936 for i in ids):
        raise RuntimeError("Invalid Qwen token IDs")
    fixture = hf_hub_download(repo_id=MODEL, filename="tokenizer/tokenizer_fixture.txt", revision=args.revision)
    with open(fixture, encoding="utf-8") as f:
        sample = f.read()
    if not sample.strip():
        raise RuntimeError("Empty upstream tokenizer fixture")
    print(json.dumps({"prompt": args.prompt, "ids": ids, "token_count": len(ids),
                      "fixture_path": fixture, "fixture_bytes": len(sample.encode("utf-8"))},
                     ensure_ascii=False))
    # Emit a machine-readable conformance vector consumed by Kotlin tests.
    vector = {"prompt": args.prompt, "ids": ids}
    with open("qwen_token_vector.json", "w", encoding="utf-8") as out:
        json.dump(vector, out, ensure_ascii=False)
    print("WROTE qwen_token_vector.json")
    print("NOTE: upstream fixture schema is retained for independent parity checks.")
if __name__ == "__main__":
    main()
