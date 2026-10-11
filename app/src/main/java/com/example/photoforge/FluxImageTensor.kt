package com.example.photoforge

import android.graphics.Bitmap
import android.graphics.Color

/** Converts Android bitmap to the exported VAE NCHW float tensor and back. */
object FluxImageTensor {
    const val SIZE = 256
    fun encode(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        try {
            val pixels = IntArray(SIZE * SIZE)
            scaled.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
            val plane = SIZE * SIZE
            return FloatArray(plane * 3).also { output ->
                pixels.forEachIndexed { i, pixel ->
                    output[i] = Color.red(pixel) / 127.5f - 1f
                    output[plane + i] = Color.green(pixel) / 127.5f - 1f
                    output[plane * 2 + i] = Color.blue(pixel) / 127.5f - 1f
                }
            }
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
    }

    fun decode(nchw: FloatArray): Bitmap {
        FluxTensorRouting.requireShape("VAE output", nchw, 1, 3, SIZE, SIZE)
        val plane = SIZE * SIZE
        val pixels = IntArray(plane)
        fun channel(value: Float): Int =
            ((value.coerceIn(-1f, 1f) + 1f) * 127.5f).toInt().coerceIn(0, 255)
        for (i in pixels.indices) {
            pixels[i] = Color.rgb(channel(nchw[i]), channel(nchw[plane + i]),
                channel(nchw[2 * plane + i]))
        }
        return Bitmap.createBitmap(pixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }
}
