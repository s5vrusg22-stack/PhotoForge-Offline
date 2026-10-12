package com.example.photoforge

/** Validated host tensors used by the exported FLUX transformer graphs. */
object FluxConditioning {
    data class Rotary(val cos: FloatArray, val sin: FloatArray)
    data class Step(val timestep: FloatArray, val textRotary: Rotary, val imageRotary: Rotary)

    fun build(
        sigma: Float,
        textHeadDimension: Int,
        imageHeadDimension: Int,
        textBase: Double = 10000.0,
        imageBase: Double = 10000.0
    ): Step {
        require(sigma.isFinite() && sigma >= 0f)
        val time = FluxHostMath.timestepEmbedding(sigma, 256)
        FluxTensorRouting.requireShape("timestep embedding", time, 1, 256)
        val text = FluxHostMath.rotaryTables(
            FluxTensorRouting.TEXT_TOKENS, textHeadDimension, textBase)
        val image = FluxHostMath.rotaryTables(
            FluxTensorRouting.EDIT_IMAGE_TOKENS, imageHeadDimension, imageBase)
        require(text.first.all { it.isFinite() } && text.second.all { it.isFinite() })
        require(image.first.all { it.isFinite() } && image.second.all { it.isFinite() })
        return Step(time, Rotary(text.first, text.second), Rotary(image.first, image.second))
    }
}
