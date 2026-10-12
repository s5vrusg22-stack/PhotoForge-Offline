package com.example.photoforge

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Loads model-exported mask/rotary tensors. No synthetic defaults are permitted. */
object FluxConditioningAssets {
    data class Tensors(
        val encoderMask: FloatArray,
        val encoderCos: FloatArray,
        val encoderSin: FloatArray,
        val ditCos: FloatArray,
        val ditSin: FloatArray
    )
    private fun read(root: File, name: String): FloatArray {
        val f = File(root, name)
        require(f.isFile && f.length() > 0 && f.length() % 4L == 0L) { "Missing/invalid $name" }
        require(f.length() <= Int.MAX_VALUE) { "$name is too large" }
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray((f.length()/4).toInt()) { b.float }.also {
            require(it.all(Float::isFinite)) { "$name contains nonfinite values" }
        }
    }
    fun load(root: File): Tensors {
        FluxConditioningManifest.verify(root)
        val t = Tensors(
            read(root,"encoder_mask_f32le.bin"),
            read(root,"encoder_rotary_cos_f32le.bin"),
            read(root,"encoder_rotary_sin_f32le.bin"),
            read(root,"dit_rotary_cos_f32le.bin"),
            read(root,"dit_rotary_sin_f32le.bin")
        )
        require(t.encoderCos.size == t.encoderSin.size) { "Encoder rotary pair mismatch" }
        require(t.ditCos.size == t.ditSin.size) { "DiT rotary pair mismatch" }
        return t
    }
}
