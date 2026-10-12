package com.example.photoforge

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Loads exact 32-channel VAE latent mean/std exported with the model package. */
data class FluxVaeStatistics(val mean: FloatArray, val std: FloatArray) {
    init {
        require(mean.size == FluxVaePacking.CHANNELS && std.size == FluxVaePacking.CHANNELS)
        require(mean.all { it.isFinite() })
        require(std.all { it.isFinite() && it > 0f })
    }
    companion object {
        private const val FLOATS = FluxVaePacking.CHANNELS * 2
        private const val BYTES = FLOATS * 4
        fun load(file: File): FluxVaeStatistics {
            require(file.isFile && file.length() == BYTES.toLong()) {
                "VAE stats must contain exactly $FLOATS little-endian float32 values"
            }
            val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val mean = FloatArray(FluxVaePacking.CHANNELS) { buffer.float }
            val std = FloatArray(FluxVaePacking.CHANNELS) { buffer.float }
            return FluxVaeStatistics(mean, std)
        }
    }
}
