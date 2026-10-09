# FLUX.2 Klein LiteRT: verified upstream contracts and failure gates

Upstream source: https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT/tree/main
Model card: https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT

## Verified against published file listing (2026-10-09)
- 21 graph **filenames** match the published repository: ke_enc0..2 (3), kc_* (8), kce_* (8), kv_vae and kv_vae_enc (2).
- Required tokenizer assets documented upstream: qwen_vocab.txt, qwen_merges.txt, qwen_special.txt, qwen_embed_fp16.bin.
- Graph paths in the audit are expected to be exact filenames, with a defensive fallback for subdirectories.
- Repository advertises Apache-2.0; confirm the model and upstream licenses before redistribution.
- The public repository displays ~11.4 GB total. This is NOT a measurement of the eventual installed app or peak RAM.
- Upstream demonstrates both text-to-image and image editing, but these are not implemented or tested in PhotoForge Android yet.

## Host-side functionality still missing from PhotoForge
- Qwen2 BPE tokenizer with fixture verification and fixed 512-token padding/truncation.
- mmap of FP16 token embedding table, token IDs and causal/padding attention mask.
- Encoder and image rotary tables, time embedding, scheduler, pack/unpack, VAE preprocessing/postprocessing.
- Different token shapes for image generation vs editing; image editing concatenates reference-image tokens.
- Correct LiteRT CompiledModel GPU integration, graph load/unload lifecycle and safe failure cleanup.
- Cancellation, atomic download/resume, hash validation, storage preflight, offline verification and photo save.
- Subject identity/background preservation requires real visual acceptance testing.

## High-risk runtime compatibility
- Generic TFLite OpenCL delegates are NOT interchangeable with LiteRT CompiledModel GPU.
- Upstream reports failures for some graphs on classic TFLite OpenCL and Qwen encoder graphs on NPU.
- A single loaded graph (~912 MB) is NOT total RAM: GPU compile caches, activations, embeddings, image buffers and host process must be measured.
- Do not load all 21 graphs simultaneously; upstream uses staged execution.
- Pixel 8a and Galaxy S26 measurements do not prove Galaxy S25 Ultra compatibility.

## CI behavior
- `scripts/audit_litert_klein_android.py` fetches only metadata, not model weights.
- It checks all 21 graph filenames, four tokenizer assets, duplicate/ambiguous paths, invalid sizes and produces a report.
- It fails if required filenames are absent or structurally invalid. Unknown sizes are explicitly reported and do not establish package completeness.
- Unit tests use synthetic metadata; passing them does not prove model downloads, Android execution or image quality.
- The report keeps `approved_for_app=false` until real-device acceptance is separately implemented.
