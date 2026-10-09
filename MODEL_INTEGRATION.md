# PhotoForge model integration register

## Confirmed published candidate: LaMa ONNX (object removal only)
- Model: https://huggingface.co/sapienkit/LaMa-ONNX
- License: Apache-2.0 (see model card and upstream attribution)
- Weight file: `lama_fp32.onnx`, approximately 208 MB
- SHA-256: `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`
- Input: image float32 [1,3,512,512], RGB / 255; mask float32 [1,1,512,512], white=1
- Output: float32 [1,3,512,512], RGB 0..255
- Build workflow downloads and checks the weight file, runs ONNX Runtime CPU smoke inference, then attempts APK build.
- Status: source integration committed; build/test run result and Android hardware performance **not verified**.

## Expression generation research candidate: PixelSmile
- Weights: https://huggingface.co/PixelSmile/PixelSmile
- Source: https://github.com/Ammmob/PixelSmile
- License on model card: Apache-2.0; check all upstream dependencies separately.
- Uses Qwen-Image-Edit-2511 base + PixelSmile LoRA, not a standalone ONNX graph.
- Status: research candidate only. **Not bundled or runnable in this Android app.** Large base model, mobile quantization, tokenizer, inference runtime, GPU memory and latency must be assessed before integration.

## Prop/clothing/pose generation
- Text field currently supports only keyword-triggered drawn accessories: glasses, hat, necklace.
- Transparent PNG overlay supports user-provided artwork.
- No text-conditioned generative model for arbitrary props, clothing, or pose is integrated.
- Future backend must use the GenerativeEditSpec interface, run without network, and composite only the selected region.

## Device acceptance test
1. APK build succeeds; package installs and launches on Galaxy S25 Ultra.
2. LaMa model SHA check passes and inference produces finite 512x512 output.
3. Paint mask, erase an object, save result, and compare pixels outside mask.
4. Type '안경', paint intended placement, apply, verify one-step undo.
5. For expression generation, demonstrate at least three preset emotions on the same face with stable identity and unchanged surroundings.
6. Repeat with airplane mode on; record peak RAM, time and crash logs.

Never describe a research candidate as an integrated AI generator.
