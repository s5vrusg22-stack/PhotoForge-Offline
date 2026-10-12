package com.example.photoforge

import android.graphics.Bitmap
import java.io.File
import kotlin.random.Random

/**
 * End-to-end edit entry point. It owns tensor assembly but delegates every
 * learned operation to verified tokenizer/embedding/LiteRT graph assets.
 */
class FluxEndToEndEditor(
    private val modelRoot: File,
    private val graphRunner: FluxGraphRunner,
    private val qwen: QwenEmbeddingPipeline
) {
    fun editWithInstalledConditioning(
        source: Bitmap,
        prompt: String,
        sigmas: FloatArray,
        seed: Int
    ): Bitmap {
        val c = FluxConditioningAssets.load(modelRoot)
        return edit(source, prompt, c.encoderMask, c.encoderCos, c.encoderSin,
            c.ditCos, c.ditSin, sigmas, seed)
    }

    fun edit(
        source: Bitmap,
        prompt: String,
        encoderMask: FloatArray,
        encoderCos: FloatArray,
        encoderSin: FloatArray,
        ditCos: FloatArray,
        ditSin: FloatArray,
        sigmas: FloatArray,
        seed: Int
    ): Bitmap {
        require(prompt.isNotBlank()) { "Edit prompt must not be blank" }
        require(sigmas.size == 5) { "FLUX Klein requires exactly five sigma boundaries" }
        FlowMatchScheduler.validateTimesteps(sigmas)
        require(sigmas.last() == 0f) { "Final sigma must be zero" }
        FluxRuntimePreflight.requireReady(modelRoot, editing = true)
        val text = qwen.encode(prompt)
        val image = FluxBitmapTensor.toNchw(source)
        val random = Random(seed)
        val noise = FloatArray(FluxTensorRouting.IMAGE_TOKENS * FluxTensorRouting.IMAGE_CHANNELS) {
            // Box-Muller Gaussian noise, deterministic for a caller-supplied seed.
            val u1 = random.nextDouble().coerceAtLeast(1e-12)
            val u2 = random.nextDouble()
            (kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) *
                kotlin.math.cos(2.0 * Math.PI * u2)).toFloat()
        }
        val host = FluxGraphExecutor.HostTensors(
            inputsEmbeds = text.embeddings,
            encoderMask = encoderMask,
            encoderCos = encoderCos,
            encoderSin = encoderSin,
            ditCos = ditCos,
            ditSin = ditSin,
            timeEmbeddings = emptyList(),
            sigmas = sigmas.copyOf(),
            initialNoise = noise,
            referenceImageChw = image
        )
        val stats = FluxVaeStatistics.load(File(modelRoot, "vae_stats_f32le.bin"))
        val withVae = FluxGraphExecutor.withVaeChannelStats(host, stats.mean, stats.std)
        val decoded = FluxEditRuntime.execute(modelRoot, graphRunner, withVae)
        return FluxBitmapTensor.fromNchw(decoded)
    }
}
