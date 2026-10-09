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
}
