package com.example.photoforge

/**
 * Host-side tensor routing for the published 256x256 FLUX.2 Klein LiteRT graph family.
 * Inputs are float32 row-major. Model-specific mask/rotary/time tensors must be
 * supplied by validated host-preparation code, never substituted with zero tensors.
 */
object FluxTensorRouting {
    const val TEXT_TOKENS = 512
    const val TEXT_HIDDEN = 2560
    const val CONTEXT_HIDDEN = 7680
    const val IMAGE_TOKENS = 256
    const val EDIT_IMAGE_TOKENS = 512
    const val IMAGE_CHANNELS = 128
    const val TRANSFORMER_HIDDEN = 3072

    fun requireShape(name: String, data: FloatArray, vararg shape: Int) {
        require(shape.all { it > 0 }) { "Invalid dimensions for $name" }
        val count = shape.fold(1L) { acc, dim -> acc * dim }
        require(count <= Int.MAX_VALUE && data.size.toLong() == count) {
            "$name: expected ${shape.joinToString("x")} ($count), got ${data.size}"
        }
        require(data.all { it.isFinite() }) { "$name contains NaN or infinity" }
    }

    /** Interleave Qwen layer 9, 18 and 27 outputs in the channel dimension. */
    fun interleaveTextTaps(taps: List<FloatArray>): FloatArray {
        require(taps.size == 3) { "Expected three encoder taps" }
        taps.forEachIndexed { i, tap -> requireShape("encoder$i", tap, 1, TEXT_TOKENS, TEXT_HIDDEN) }
        val output = FloatArray(TEXT_TOKENS * CONTEXT_HIDDEN)
        for (token in 0 until TEXT_TOKENS) {
            val src = token * TEXT_HIDDEN
            val dst = token * CONTEXT_HIDDEN
            for (layer in 0 until 3) {
                taps[layer].copyInto(output, dst + layer * TEXT_HIDDEN, src, src + TEXT_HIDDEN)
            }
        }
        return output
    }

    /** Text tokens precede image tokens in the DiT single-stream blocks. */
    fun joinTextImage(text: FloatArray, image: FloatArray, edit: Boolean): FloatArray {
        val imageTokens = if (edit) EDIT_IMAGE_TOKENS else IMAGE_TOKENS
        requireShape("text", text, 1, TEXT_TOKENS, TRANSFORMER_HIDDEN)
        requireShape("image", image, 1, imageTokens, TRANSFORMER_HIDDEN)
        return FloatArray(text.size + image.size).also {
            text.copyInto(it)
            image.copyInto(it, text.size)
        }
    }

    fun imageFromJoint(joint: FloatArray, edit: Boolean): FloatArray {
        val imageTokens = if (edit) EDIT_IMAGE_TOKENS else IMAGE_TOKENS
        requireShape("joint", joint, 1, TEXT_TOKENS + imageTokens, TRANSFORMER_HIDDEN)
        return joint.copyOfRange(TEXT_TOKENS * TRANSFORMER_HIDDEN, joint.size)
    }

    /** Editing appends reference tokens, and scheduler updates only noise tokens. */
    fun concatenateNoiseReference(noise: FloatArray, reference: FloatArray): FloatArray {
        requireShape("noise", noise, 1, IMAGE_TOKENS, IMAGE_CHANNELS)
        requireShape("reference", reference, 1, IMAGE_TOKENS, IMAGE_CHANNELS)
        return noise + reference
    }

    fun noiseVelocity(editOutput: FloatArray, edit: Boolean): FloatArray {
        requireShape("DiT velocity", editOutput, 1,
            if (edit) EDIT_IMAGE_TOKENS else IMAGE_TOKENS, IMAGE_CHANNELS)
        return editOutput.copyOfRange(0, IMAGE_TOKENS * IMAGE_CHANNELS)
    }

    fun applyVelocity(noise: FloatArray, predicted: FloatArray, sigma: Float, nextSigma: Float) {
        requireShape("noise", noise, 1, IMAGE_TOKENS, IMAGE_CHANNELS)
        require(predicted.size == IMAGE_TOKENS * IMAGE_CHANNELS ||
            predicted.size == EDIT_IMAGE_TOKENS * IMAGE_CHANNELS) {
            "Unexpected velocity length: ${predicted.size}"
        }
        val velocity = noiseVelocity(predicted, predicted.size == EDIT_IMAGE_TOKENS * IMAGE_CHANNELS)
        FlowMatchScheduler.eulerStep(noise, velocity, sigma, nextSigma)
    }
}
