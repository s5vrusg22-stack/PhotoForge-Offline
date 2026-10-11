package com.example.photoforge

import java.io.File
import java.io.RandomAccessFile

/** Bounded random-access Qwen fp16 embeddings. IDs require a verified tokenizer. */
class FluxEmbeddingLookup(file: File) : AutoCloseable {
    companion object {
        const val VOCAB_SIZE = 151936
        const val HIDDEN_SIZE = 2560
        const val ROW_BYTES = HIDDEN_SIZE * 2
    }
    private val handle = RandomAccessFile(file, "r")
    init {
        require(handle.length() == VOCAB_SIZE.toLong() * ROW_BYTES) {
            "Unexpected Qwen fp16 embedding size: ${handle.length()}"
        }
    }
    fun lookup(ids: IntArray): FloatArray {
        require(ids.size == 512) { "Qwen encoder requires 512 token IDs" }
        val result = FloatArray(ids.size * HIDDEN_SIZE)
        val row = ByteArray(ROW_BYTES)
        ids.forEachIndexed { index, id ->
            require(id in 0 until VOCAB_SIZE) { "Token ID out of range: $id" }
            handle.seek(id.toLong() * ROW_BYTES)
            handle.readFully(row)
            for (column in 0 until HIDDEN_SIZE) {
                val offset = column * 2
                val bits = (row[offset].toInt() and 255) or
                    ((row[offset + 1].toInt() and 255) shl 8)
                result[index * HIDDEN_SIZE + column] = halfToFloat(bits)
            }
        }
        require(result.all { it.isFinite() }) { "Nonfinite Qwen embedding" }
        return result
    }
    private fun halfToFloat(h: Int): Float {
        val sign = (h and 0x8000) shl 16
        val exponent = (h ushr 10) and 0x1f
        val mantissa = h and 0x3ff
        val bits = when (exponent) {
            0 -> {
                if (mantissa == 0) sign else {
                    var m = mantissa
                    var e = -14
                    while ((m and 0x400) == 0) { m = m shl 1; e-- }
                    sign or ((e + 127) shl 23) or ((m and 0x3ff) shl 13)
                }
            }
            31 -> sign or 0x7f800000 or (mantissa shl 13)
            else -> sign or ((exponent - 15 + 127) shl 23) or (mantissa shl 13)
        }
        return Float.fromBits(bits)
    }
    override fun close() = handle.close()
}
