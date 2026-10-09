package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test

class FluxVaePackingTest {
    @Test fun roundTripPreservesAllChannelsAndPixels() {
        val source = FloatArray(32 * 32 * 32) { (it % 997).toFloat() }
        val packed = FluxVaePacking.pack(source)
        assertEquals(256 * 128, packed.size)
        assertArrayEquals(source, FluxVaePacking.unpack(packed), 0f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun wrongShapeRejected() {
        FluxVaePacking.pack(FloatArray(42))
    }
    @Test fun normalizedRoundTripPreservesLatents() {
        val source = FloatArray(32 * 32 * 32) { (it % 37 - 18) / 19f }
        val packed = FluxVaePacking.packNormalized(source, 0.7f, 0.25f)
        val restored = FluxVaePacking.unpackNormalized(packed, 0.7f, 0.25f)
        assertArrayEquals(source, restored, 0.000001f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsZeroNormalizationScale() {
        FluxVaePacking.packNormalized(FloatArray(32 * 32 * 32), 0f, 0f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonfiniteNormalizedLatents() {
        FluxVaePacking.unpackNormalized(FloatArray(256 * 128) { Float.MAX_VALUE }, 0.000001f, 0f)
    }
}
