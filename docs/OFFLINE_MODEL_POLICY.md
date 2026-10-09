# PhotoForge Offline — model deployment policy

## Primary target
- Android: Samsung Galaxy S25 Ultra; offline image editing is mandatory.
- User explicitly permits an on-device package larger than 11.4 GB. There is no hard 10 GiB installation cap. Report exact complete download/storage footprint and free-space requirement; do not equate disk space with RAM.
- Primary feature is image-to-image photo editing with identity/background preservation; secondary feature is text-to-image generation. Both must run offline and be tested independently.
- Candidate must support **image-to-image editing** and preserve subject identity and background; text-to-image-only models do not qualify.
- Verify license, actual complete file sizes, Android inference backend, peak RAM, latency and image quality before adopting a candidate.
- 4B parameter count does **not** imply a working 4B Android image-editing model. Do not claim mobile support without device testing.

## Legacy Qwen/PixelSmile
- Keep the Qwen Image Edit 2511 + PixelSmile LoRA code and compatibility reports for future optional online use.
- Do not download the ~54 GB full Qwen weights in CI or to the phone.
- Do not ship a fake or remote-only editing feature as 'offline'.
- Online Qwen is disabled by default and requires an explicitly configured external GPU service; GitHub Actions is not an inference server.
- PixelSmile LoRA is specific to its target architecture and must not be loaded into a different model without verified compatibility.

## Android LiteRT candidate (priority)
- Repository: https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT
- INT8 staged 21-graph pipeline: photo editing uses text encoder + editing DiT + VAE encoder/decoder; image generation uses text encoder + generation DiT + VAE decoder. Android Kotlin LiteRT GPU integration is a candidate, not an app feature yet.
- Reported Pixel 8a full editing ~328-369 seconds. Galaxy S26 per-graph Adreno GPU tests do NOT prove Galaxy S25 Ultra end-to-end performance.
- Include tokenizer embedding sidecar (~778 MB) and all host assets in disk budget.
- NPU path has failures on some text-encoder graphs; GPU preferred pending real device validation.
- Workflow: .github/workflows/litert-klein-android-audit.yml (metadata only; no weight downloads).
- Real S25 Ultra offline edit, peak RAM, output quality and total package size remain release gates.

## Release gates
1. List **all** required weights, tokenizer assets, runtime files and licenses for both editing and generation.
2. Measure full package disk usage, available free space, download integrity and separate runtime RAM; size above 11.4 GB is allowed.
3. Test a real photo-to-photo edit with loaded weights and verify subject/background preservation.
4. Separately test real text-to-image generation with loaded weights.
5. Validate both complete Android execution paths on Galaxy S25 Ultra, record peak RAM, latency, image output and offline network isolation.
6. Verify cancellation, error reporting, model file integrity and image saving; enable features only after real inference passes. No placeholder success results.
