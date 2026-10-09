# PhotoForge Offline — original-preserving photo editor

## Product scope (October 2026)
The editor prioritizes **accessory/prop placement, clothing changes, and pose changes**, not hair recoloring. Preserve the source photo outside the selected region.

### Implemented source code
- Import a photo, select a transparent PNG prop and composite it at the center.
- Paint/undo/clear masks and run compatible LaMa ONNX **object removal** locally.
- Select prop/clothing/pose modes in the interface, with clear status messages.
- One-step undo after PNG compositing or successful LaMa removal.
- Rotate, mirror, brightness and export PNG.

### Not yet implemented
- Generative clothes replacement or pose transformation. These require an image-conditioned diffusion or pose-guided model, compatible runtime, model weights, and on-device validation.
- Text-prompt prop generation; draggable prop layers; production-ready layer undo stack.
- Verified APK build, model download/checksum, actual Galaxy S25 Ultra inference.

**Do not mistake mode buttons for implemented AI clothes/pose transformations.**

### Development acceptance criteria
1. Install and launch debug APK on Galaxy S25 Ultra.
2. Load a source portrait and overlay PNG without changing unselected pixels.
3. Confirm one-step undo restores the previous bitmap.
4. Confirm a legitimately obtained compatible LaMa ONNX model passes inference on device.
5. Implement pose/clothing AI only after obtaining a suitable licensed model and validating latency, memory, and preservation quality.

## Build
See [GitHub Actions](https://github.com/s5vrusg22-stack/PhotoForge-Offline/actions).

## Text prompt accessory milestone

A Korean text input is now wired to `PromptPropRenderer`, which draws transparent vector accessories for the exact supported keywords **안경 / 모자 / 목걸이** (also basic English synonyms). The result is composited on the photo with one-step undo. This is **not an AI image generator**, and the accessory is not anchored to face/body landmarks.

## Generative AI blocker

LaMa ONNX only removes/fills regions; it cannot synthesize arbitrary prompted objects, clothing or poses. Implementing real prompt-conditioned local generation requires compatible licensed model weights, tokenizer/text encoder, denoising UNet or transformer, scheduler, VAE, image/mask conditioning and possibly pose conditioning. None of those diffusion components is present in this repository yet. Do not advertise free-text AI generation as functional until end-to-end device inference is demonstrated.

## Primary feature: facial expression editing

**Priority order:** 1. expression change, 2. text-entered accessories, 3. clothing change, 4. pose change, 5. LaMa object removal.

The expression UI now exposes six presets (natural smile, broad smile, neutral, sad, surprised, angry) and 15%/35%/65% strength settings. The button does not mutate pixels until an actual generative model is installed and tested. This avoids falsely claiming AI expression edits work.

`GenerativeEditSpec.kt` defines a local image-conditioned model interface and preservation-focused positive/negative prompts. The placeholder backend explicitly reports unavailability. **This is architecture, not a functioning diffusion pipeline.** A real backend must provide compatible model weights, image/mask conditioning, face identity preservation, scheduler and device inference tests. Facial expression editing is more challenging than object removal because preserving the same person's identity matters.
