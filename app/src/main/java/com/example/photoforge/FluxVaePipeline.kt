package com.example.photoforge

import java.io.File

/** Verified VAE bridge between encoder/decoder graph tensors and packed FLUX image tokens. */
object FluxVaePipeline {
    fun encodeLatentToReference(
        encoderLatentNchw: FloatArray,
        modelRoot: File
    ): FloatArray {
        val stats = FluxVaeStatistics.load(File(modelRoot, "vae_stats_f32le.bin"))
        val packed = FluxVaePacking.packWithChannelStats(
            encoderLatentNchw, stats.mean, stats.std)
        FluxTensorRouting.requireShape(
            "reference latent", packed, 1,
            FluxTensorRouting.IMAGE_TOKENS, FluxTensorRouting.IMAGE_CHANNELS)
        return packed
    }

    fun decodeTokensToLatent(
        denoisedTokens: FloatArray,
        modelRoot: File
    ): FloatArray {
        FluxTensorRouting.requireShape(
            "denoised latent", denoisedTokens, 1,
            FluxTensorRouting.IMAGE_TOKENS, FluxTensorRouting.IMAGE_CHANNELS)
        val stats = FluxVaeStatistics.load(File(modelRoot, "vae_stats_f32le.bin"))
        return FluxVaePacking.unpackWithChannelStats(
            denoisedTokens, stats.mean, stats.std)
    }
}
