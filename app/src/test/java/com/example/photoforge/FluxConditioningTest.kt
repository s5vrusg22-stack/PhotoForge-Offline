package com.example.photoforge

import org.junit.Assert.assertEquals
import org.junit.Test

class FluxConditioningTest {
    @Test fun acceptsModelProjectedTime() {
        val c = FluxConditioning.fromModelProjected(
            FloatArray(3072) { 0.25f }, floatArrayOf(1f), floatArrayOf(0f),
            floatArrayOf(1f), floatArrayOf(0f))
        assertEquals(3072, c.timestep.size)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsRawFourierTime() {
        FluxConditioning.fromModelProjected(
            FloatArray(256), floatArrayOf(1f), floatArrayOf(0f),
            floatArrayOf(1f), floatArrayOf(0f))
    }

    @Test(expected = IllegalStateException::class)
    fun oldBuilderCannotGenerateFakeProjectedTime() {
        @Suppress("DEPRECATION")
        FluxConditioning.build(1f, 64, 64)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsRotaryShapeMismatch() {
        FluxConditioning.fromModelProjected(
            FloatArray(3072), floatArrayOf(1f, 1f), floatArrayOf(0f),
            floatArrayOf(1f), floatArrayOf(0f))
    }
}
