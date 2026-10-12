package com.example.photoforge

/**
 * Model-compatible conditioning contract.
 *
 * The exported kce_prep and kce_final graphs consume a projected 3072-wide
 * timestep embedding, NOT the raw 256-wide sinusoidal Fourier features.
 * The projection weights are checkpoint-specific and cannot be synthesized by
 * repeating or zero-padding 256 values.
 */
object FluxConditioning {
    const val PROJECTED_TIME_WIDTH = 3072
    data class Rotary(val cos: FloatArray, val sin: FloatArray)
    data class Step(val timestep: FloatArray, val textRotary: Rotary, val imageRotary: Rotary)

    fun fromModelProjected(
        projectedTime: FloatArray,
        textCos: FloatArray,
        textSin: FloatArray,
        imageCos: FloatArray,
        imageSin: FloatArray
    ): Step {
        FluxTensorRouting.requireShape("projected timestep", projectedTime, 1, PROJECTED_TIME_WIDTH)
        require(projectedTime.all { it.isFinite() }) { "Non-finite projected timestep" }
        require(textCos.isNotEmpty() && textCos.size == textSin.size &&
            textCos.all { it.isFinite() } && textSin.all { it.isFinite() }) {
            "Invalid text rotary tables"
        }
        require(imageCos.isNotEmpty() && imageCos.size == imageSin.size &&
            imageCos.all { it.isFinite() } && imageSin.all { it.isFinite() }) {
            "Invalid image rotary tables"
        }
        return Step(
            projectedTime.copyOf(),
            Rotary(textCos.copyOf(), textSin.copyOf()),
            Rotary(imageCos.copyOf(), imageSin.copyOf())
        )
    }

    @Deprecated("256-wide Fourier features are not the 3072-wide projected model input. Supply verified model-projected conditioning.")
    fun build(
        sigma: Float,
        textHeadDimension: Int,
        imageHeadDimension: Int,
        textBase: Double = 10000.0,
        imageBase: Double = 10000.0
    ): Step = error("FLUX requires model-projected [1,3072] timestep; raw sinusoidal features cannot be passed to kce_prep")
}
