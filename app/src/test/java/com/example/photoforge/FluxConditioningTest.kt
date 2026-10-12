package com.example.photoforge

import org.junit.Assert.assertEquals
import org.junit.Test

class FluxConditioningTest {
    @Test fun buildsFiniteTimeAndRotaryTensors() {
        val c = FluxConditioning.build(1f, 64, 64)
        assertEquals(256, c.timestep.size)
        assertEquals(FluxTensorRouting.TEXT_TOKENS * 64, c.textRotary.cos.size)
        assertEquals(FluxTensorRouting.EDIT_IMAGE_TOKENS * 64, c.imageRotary.sin.size)
    }
}
