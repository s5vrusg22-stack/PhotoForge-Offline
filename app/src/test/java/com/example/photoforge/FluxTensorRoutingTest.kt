package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test

class FluxTensorRoutingTest {
    @Test fun interleavesThreeQwenTaps() {
        val taps = listOf(FloatArray(512 * 2560) { 1f },
            FloatArray(512 * 2560) { 2f }, FloatArray(512 * 2560) { 3f })
        val out = FluxTensorRouting.interleaveTextTaps(taps)
        assertEquals(512 * 7680, out.size)
        assertEquals(1f, out[0])
        assertEquals(2f, out[2560])
        assertEquals(3f, out[5120])
        assertEquals(2f, out[7680 + 2560])
    }

    @Test fun generationAndEditingJoinHaveExpectedLengths() {
        val text = FloatArray(512 * 3072) { 7f }
        val image = FloatArray(256 * 3072) { 9f }
        val gen = FluxTensorRouting.joinTextImage(text, image, false)
        assertEquals(768 * 3072, gen.size)
        assertEquals(9f, gen[512 * 3072])
        val edit = FluxTensorRouting.joinTextImage(text, FloatArray(512 * 3072) { 4f }, true)
        assertEquals(1024 * 3072, edit.size)
        assertEquals(4f, edit[512 * 3072])
        assertEquals(512 * 3072, FluxTensorRouting.imageFromJoint(edit, true).size)
    }

    @Test fun editVelocityUpdatesNoiseOnly() {
        val noise = FloatArray(256 * 128) { 1f }
        val prediction = FloatArray(512 * 128) { i -> if (i < 256 * 128) 2f else 999f }
        FluxTensorRouting.applyVelocity(noise, prediction, 1f, 0.5f)
        assertEquals(0f, noise[0], 0f)
        assertEquals(0f, noise.last(), 0f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsInvalidJointTensor() {
        FluxTensorRouting.imageFromJoint(FloatArray(100), false)
    }
}
