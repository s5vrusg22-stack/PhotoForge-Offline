package com.example.photoforge

/**
 * Spatial 2x2 packing: [1,32,32,32] NCHW -> [1,256,128].
 * Reverse conversion restores [1,32,32,32].
 * Plain pack/unpack are layout-only. Normalized variants require verified model coefficients.
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
    /**
     * Model-specific normalization is explicit, never inferred. The exported
     * graph must confirm whether this convention is appropriate.
     */
    fun packNormalized(nchw: FloatArray, scale: Float, shift: Float): FloatArray {
        validateNormalization(scale, shift)
        return pack(nchw).also { values ->
            for (i in values.indices) values[i] = (values[i] - shift) * scale
            require(values.all { it.isFinite() }) { "Non-finite normalized VAE latent" }
        }
    }

    fun unpackNormalized(tokens: FloatArray, scale: Float, shift: Float): FloatArray {
        validateNormalization(scale, shift)
        val raw = tokens.copyOf()
        FluxTensorRouting.requireShape("normalized VAE tokens", raw, 1, PACKED_TOKENS, PACKED_CHANNELS)
        for (i in raw.indices) raw[i] = raw[i] / scale + shift
        require(raw.all { it.isFinite() }) { "Non-finite denormalized VAE latent" }
        return unpack(raw)
    }

    /**
     * The FLUX.2 VAE uses 32 channel-specific batch-normalization statistics.
     * Apply them in NCHW space, before packing and after unpacking respectively.
     * Callers must supply the exact exported checkpoint statistics.
     */
    fun packWithChannelStats(nchw: FloatArray, mean: FloatArray, std: FloatArray): FloatArray {
        validateChannelStats(mean, std)
        require(nchw.size == CHANNELS * HEIGHT * WIDTH)
        val normalized = FloatArray(nchw.size)
        val plane = HEIGHT * WIDTH
        for (c in 0 until CHANNELS) for (i in 0 until plane) {
            val index = c * plane + i
            normalized[index] = (nchw[index] - mean[c]) / std[c]
        }
        return pack(normalized)
    }

    fun unpackWithChannelStats(tokens: FloatArray, mean: FloatArray, std: FloatArray): FloatArray {
        validateChannelStats(mean, std)
        val nchw = unpack(tokens)
        val plane = HEIGHT * WIDTH
        for (c in 0 until CHANNELS) for (i in 0 until plane) {
            val index = c * plane + i
            nchw[index] = nchw[index] * std[c] + mean[c]
        }
        require(nchw.all { it.isFinite() }) { "Nonfinite VAE denormalized latent" }
        return nchw
    }

    private fun validateChannelStats(mean: FloatArray, std: FloatArray) {
        require(mean.size == CHANNELS && std.size == CHANNELS) {
            "VAE requires 32 channel means and standard deviations"
        }
        require(mean.all { it.isFinite() } && std.all { it.isFinite() && it > 0f }) {
            "Invalid VAE channel statistics"
        }
    }

    private fun validateNormalization(scale: Float, shift: Float) {
        require(scale.isFinite() && scale > 0f && shift.isFinite()) {
            "VAE normalization requires a finite positive scale and finite shift"
        }
    }
}
