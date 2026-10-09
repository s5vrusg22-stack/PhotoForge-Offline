# PhotoForge Offline

An Android photo editor with local image manipulation and an **ONNX Runtime inpainting integration**. Designed for offline operation.

## Implemented source features

- Import a photo from Android document picker.
- Touch-paint an inpainting mask on the displayed image, undo a stroke, clear the mask and change brush size.
- Choose a **user-provided** compatible `.onnx` inpainting model and copy it to app-private storage.
- Run ONNX Runtime locally on CPU on a background thread; composite masked output over the original photo.
- Overlay a transparent PNG (centered), rotate, mirror, adjust brightness and save PNG to `Pictures/PhotoForge`.
- Dark UI with large touch-friendly controls; no INTERNET permission.

## ONNX model compatibility

The adapter currently requires a LaMa-style ONNX graph with two float32 inputs named `image` and `mask` (or `img` for image). Expected tensor shapes:
- `image`: [1, 3, 512, 512] RGB in 0..1
- `mask`: [1, 1, 512, 512] white=erase
- output: [1, 3, 512, 512] RGB in 0..1

**The app does not include model weights.** Not every model named LaMa uses this exact tensor contract. Users must obtain a legitimately distributable compatible ONNX file separately. The adapter does not generate new objects from text prompts; it fills/removes selected regions according to the chosen model.

## Build and validation

Go to [GitHub Actions](https://github.com/s5vrusg22-stack/PhotoForge-Offline/actions) and open the **Build Android APK** workflow. After a successful run, download the `PhotoForge-debug-apk` artifact.

**Important:** Source is committed, but APK build success and actual ONNX inference on Galaxy S25 Ultra have not been verified. This is an implementation milestone, not a validated production release.

## Known limitations

- Fixed 512x512 inference can distort non-square inputs and soften generated areas.
- Large photos and model files may cause out-of-memory failures.
- Model loading currently copies the ONNX file into private storage, requiring extra space.
- There is no text-guided diffusion, SDXL, FLUX or model-weight download.
- Overlay is centered, not yet draggable/resizable.
- Before calling this app finished, verify build logs, APK install, and at least one compatible ONNX model on device.
