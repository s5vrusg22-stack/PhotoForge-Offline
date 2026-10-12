package com.example.photoforge

import android.graphics.Bitmap
import kotlin.math.roundToInt

/** Exact RGB float tensor bridge for the 256x256 VAE graphs. */
object FluxBitmapTensor {
    const val SIZE = 256
    fun toNchw(bitmap: Bitmap): FloatArray {
        val scaled = if (bitmap.width == SIZE && bitmap.height == SIZE) bitmap
            else Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        val pixels = IntArray(SIZE * SIZE)
        scaled.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val out = FloatArray(3 * SIZE * SIZE)
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i] = (((p shr 16) and 255) / 127.5f) - 1f
            out[plane + i] = (((p shr 8) and 255) / 127.5f) - 1f
            out[2 * plane + i] = ((p and 255) / 127.5f) - 1f
        }
        if (scaled !== bitmap) scaled.recycle()
        return out
    }

    fun fromNchw(rgb: FloatArray): Bitmap {
        FluxTensorRouting.requireShape("decoded RGB", rgb, 1, 3, SIZE, SIZE)
        val plane = SIZE * SIZE
        val pixels = IntArray(plane)
        fun byte(v: Float) = (((v.coerceIn(-1f, 1f) + 1f) * 127.5f).roundToInt()).coerceIn(0,255)
        for (i in 0 until plane) {
            pixels[i] = (255 shl 24) or (byte(rgb[i]) shl 16) or
                (byte(rgb[plane+i]) shl 8) or byte(rgb[2*plane+i])
        }
        return Bitmap.createBitmap(pixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }
}
