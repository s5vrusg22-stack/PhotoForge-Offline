package com.example.photoforge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Color
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.min

/**
 * Local LaMa-style ONNX inference adapter.
 * Contract: image [1,3,H,W] RGB 0..1, mask [1,1,H,W] 0/1,
 * output [1,3,H,W] RGB 0..255. Models with other signatures are rejected.
 */
object InpaintEngine {
    private const val SIDE = 512

    fun run(model: File, image: Bitmap, mask: Bitmap): Bitmap {
        require(model.isFile && model.length() > 0L) { "ONNX 모델 파일이 없습니다." }
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(1)
        }
        opts.use {
            env.createSession(model.absolutePath, opts).use { session ->
                val names = session.inputNames
                require(names.size == 2) { "입력 2개(image, mask)를 가진 LaMa ONNX 모델이 필요합니다: $names" }
                val imageName = names.firstOrNull { it.equals("image", true) || it.equals("img", true) }
                    ?: names.firstOrNull { it.contains("image", true) }
                    ?: error("image 입력을 찾을 수 없습니다: $names")
                val maskName = names.firstOrNull { it.contains("mask", true) }
                    ?: error("mask 입력을 찾을 수 없습니다: $names")
                val rgb = Bitmap.createScaledBitmap(image, SIDE, SIDE, true)
                val m = Bitmap.createScaledBitmap(mask, SIDE, SIDE, false)
                val pixels = IntArray(SIDE * SIDE)
                val masks = IntArray(SIDE * SIDE)
                rgb.getPixels(pixels, 0, SIDE, 0, 0, SIDE, SIDE)
                m.getPixels(masks, 0, SIDE, 0, 0, SIDE, SIDE)
                val plane = SIDE * SIDE
                val imageValues = FloatArray(3 * plane)
                val maskValues = FloatArray(plane)
                for (i in 0 until plane) {
                    val p = pixels[i]
                    imageValues[i] = Color.red(p) / 255f
                    imageValues[plane + i] = Color.green(p) / 255f
                    imageValues[plane * 2 + i] = Color.blue(p) / 255f
                    maskValues[i] = if (Color.red(masks[i]) > 127) 1f else 0f
                }
                OnnxTensor.createTensor(env, FloatBuffer.wrap(imageValues), longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong())).use { img ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(maskValues), longArrayOf(1, 1, SIDE.toLong(), SIDE.toLong())).use { matte ->
                        session.run(mapOf(imageName to img, maskName to matte)).use { result ->
                            val output = result[0] as? OnnxTensor ?: error("ONNX 결과가 텐서가 아닙니다.")
                            val shape = output.info.shape
                            require(shape.contentEquals(longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong()))) {
                                "지원하지 않는 출력 크기: ${shape.contentToString()}"
                            }
                            val values = output.floatBuffer
                            val resultPixels = IntArray(plane)
                            fun ch(i: Int): Int = values.get(i).toInt().coerceIn(0, 255)
                            for (i in 0 until plane) {
                                resultPixels[i] = Color.rgb(ch(i), ch(plane + i), ch(2 * plane + i))
                            }
                            val generated = Bitmap.createBitmap(SIDE, SIDE, Bitmap.Config.ARGB_8888)
                            generated.setPixels(resultPixels, 0, SIDE, 0, 0, SIDE, SIDE)
                            val restored = Bitmap.createScaledBitmap(generated, image.width, image.height, true)
                            // Preserve unmasked original pixels, avoiding changes to the entire photo.
                            val original = image.copy(Bitmap.Config.ARGB_8888, false)
                            val canvas = android.graphics.Canvas(original)
                            val fullMask = Bitmap.createScaledBitmap(mask, image.width, image.height, false)
                            val maskPixels = IntArray(image.width * image.height)
                            val restoredPixels = IntArray(image.width * image.height)
                            val basePixels = IntArray(image.width * image.height)
                            fullMask.getPixels(maskPixels, 0, image.width, 0, 0, image.width, image.height)
                            restored.getPixels(restoredPixels, 0, image.width, 0, 0, image.width, image.height)
                            original.getPixels(basePixels, 0, image.width, 0, 0, image.width, image.height)
                            for (i in basePixels.indices) {
                                if (Color.red(maskPixels[i]) > 127) basePixels[i] = restoredPixels[i]
                            }
                            original.setPixels(basePixels, 0, image.width, 0, 0, image.width, image.height)
                            return original
                        }
                    }
                }
            }
        }
    }
}
