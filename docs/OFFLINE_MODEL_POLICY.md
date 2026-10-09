# PhotoForge Offline — model deployment policy

## Primary target
- Android: Samsung Galaxy S25 Ultra; offline image editing is mandatory.
- Maximum total on-device AI package target: 10 GiB, including all encoders, VAE, tokenizer, and other required assets. This is a target, not an achieved result.
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
- INT8 staged 21-graph on-device image editing pipeline (256x256), Android Kotlin LiteRT GPU.
- Reported Pixel 8a full editing ~328-369 seconds. Galaxy S26 per-graph Adreno GPU tests do NOT prove Galaxy S25 Ultra end-to-end performance.
- Include tokenizer embedding sidecar (~778 MB) and all host assets in disk budget.
- NPU path has failures on some text-encoder graphs; GPU preferred pending real device validation.
- Workflow: .github/workflows/litert-klein-android-audit.yml (metadata only; no weight downloads).
- Real S25 Ultra offline edit, peak RAM, output quality and total package size remain release gates.

## Release gates
1. Select a real image-editing model and list **all** runtime artifacts and licenses.
2. Confirm complete quantized package <= 10 GiB.
3. Test one actual image-to-image edit with loaded weights on a supported GPU/CPU.
4. Confirm Android backend supports the full pipeline; measure peak RAM and time on Galaxy S25 Ultra.
5. Only then enable offline editing in the app. No placeholder success results.
