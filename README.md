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
