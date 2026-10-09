package com.example.photoforge

/**
 * Spatial 2x2 packing: [1,32,32,32] NCHW -> [1,256,128].
 * Reverse conversion restores [1,32,32,32].
 * This is layout conversion only, not VAE latent normalization/scaling.
 */
object FluxVaePacking {
    const val CHANNELS = 32
    const val HEIGHT = 32
    const val WIDTH = 32
    const val PACKED_TOKENS = 256
    const val PACKED_CHANNELS = 128

    fun pack(nchw: FloatArray): FloatArray {
        FluxTensorRouting.requireShape("VAE NCHW latent", nchw, 1, CHANNELS, HEIGHT, WIDTH)
        val result = FloatArray(PACKED_TOKENS * PACKED_CHANNELS)
        for (y in 0 until 16) for (x in 0 until 16)
            for (dy in 0..1) for (dx in 0..1) for (c in 0 until CHANNELS) {
                val token = y * 16 + x
                val channel = (dy * 2 + dx) * CHANNELS + c
                result[token * PACKED_CHANNELS + channel] =
                    nchw[c * HEIGHT * WIDTH + (y * 2 + dy) * WIDTH + x * 2 + dx]
            }
        return result
    }

    fun unpack(tokens: FloatArray): FloatArray {
        FluxTensorRouting.requireShape("packed VAE latent", tokens, 1, PACKED_TOKENS, PACKED_CHANNELS)
        val result = FloatArray(CHANNELS * HEIGHT * WIDTH)
        for (y in 0 until 16) for (x in 0 until 16)
            for (dy in 0..1) for (dx in 0..1) for (c in 0 until CHANNELS) {
                val token = y * 16 + x
                val channel = (dy * 2 + dx) * CHANNELS + c
                result[c * HEIGHT * WIDTH + (y * 2 + dy) * WIDTH + x * 2 + dx] =
                    tokens[token * PACKED_CHANNELS + channel]
            }
        return result
    }
}
