package com.example.photoforge

import android.graphics.Bitmap

/**
 * Compatibility facade. FluxBitmapTensor is the single canonical implementation
 * so RGB normalization cannot silently diverge between entry points.
 */
@Deprecated("Use FluxBitmapTensor")
object FluxImageTensor {
    const val SIZE = FluxBitmapTensor.SIZE
    fun encode(bitmap: Bitmap): FloatArray = FluxBitmapTensor.toNchw(bitmap)
    fun decode(nchw: FloatArray): Bitmap = FluxBitmapTensor.fromNchw(nchw)
}
