package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test

class FluxHostMathTest {
    @Test fun timestepZeroHasUnitCosineAndZeroSine() {
        val embedding = FluxHostMath.timestepEmbedding(0f, 8)
        assertArrayEquals(floatArrayOf(1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f),
            embedding, 0f)
    }

    @Test fun rotaryTablesAtPositionZeroAreIdentity() {
        val (cos, sin) = FluxHostMath.rotaryTables(3, 8)
        assertEquals(24, cos.size)
        for (i in 0 until 8) {
            assertEquals(1f, cos[i], 0f)
            assertEquals(0f, sin[i], 0f)
        }
        assertTrue(sin[8] != 0f)
    }

    @Test fun causalPaddingMaskBlocksFutureAndInvalidKeys() {
        val bias = FluxHostMath.attentionBias(
            booleanArrayOf(true, true, false), causal = true)
        assertEquals(9, bias.size)
        assertEquals(0f, bias[0], 0f)
        assertTrue(bias[1] < -1000f)
        assertEquals(0f, bias[3], 0f)
        assertTrue(bias[5] < -1000f)
        assertTrue(bias[6] < -1000f)
        val repeated = FluxHostMath.repeatHeads(bias, 3, 2)
        assertEquals(18, repeated.size)
        assertArrayEquals(bias, repeated.copyOfRange(9, 18), 0f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOddRotaryHeadSize() {
        FluxHostMath.rotaryTables(8, 7)
    }
}
