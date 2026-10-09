package com.example.photoforge

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.pow

/**
 * CPU-side deterministic building blocks. Frequencies, axis layout and masks
 * must be matched to the exported model before passing them into GPU graphs.
 */
object FluxHostMath {
    /** Sinusoidal flow timestep embedding, shape [1, dimension]. */
    fun timestepEmbedding(timestep: Float, dimension: Int = 256,
                          maxPeriod: Double = 10000.0): FloatArray {
        require(timestep.isFinite() && dimension > 0 && dimension % 2 == 0)
        require(maxPeriod > 1.0 && maxPeriod.isFinite())
        val half = dimension / 2
        return FloatArray(dimension) { index ->
            val k = index % half
            val frequency = maxPeriod.pow(-k.toDouble() / half)
            val angle = timestep.toDouble() * frequency
            if (index < half) cos(angle).toFloat() else sin(angle).toFloat()
        }
    }

    /**
     * One-dimensional rotary position tables in interleaved [cos,sin] pairs.
     * Each position occupies headDimension values; caller maps axes to graph
     * tensors based on the actual exported signature.
     */
    fun rotaryTables(tokens: Int, headDimension: Int, base: Double = 10000.0):
            Pair<FloatArray, FloatArray> {
        require(tokens > 0 && headDimension > 0 && headDimension % 2 == 0)
        require(base > 1.0 && base.isFinite())
        val cosTable = FloatArray(tokens * headDimension)
        val sinTable = FloatArray(tokens * headDimension)
        for (position in 0 until tokens) {
            for (pair in 0 until headDimension / 2) {
                val theta = position * base.pow(-2.0 * pair / headDimension)
                val c = cos(theta).toFloat()
                val s = sin(theta).toFloat()
                val offset = position * headDimension + 2 * pair
                cosTable[offset] = c
                cosTable[offset + 1] = c
                sinTable[offset] = s
                sinTable[offset + 1] = s
            }
        }
        return cosTable to sinTable
    }

    /**
     * Additive attention bias: valid queries can attend to valid keys only.
     * A causal mask blocks future keys; padding keys are always blocked.
     */
    fun attentionBias(validTokens: BooleanArray, causal: Boolean,
                      blocked: Float = -1e9f): FloatArray {
        require(validTokens.isNotEmpty() && blocked.isFinite() && blocked < 0f)
        val count = validTokens.size
        return FloatArray(count * count) { index ->
            val query = index / count
            val key = index % count
            if (validTokens[query] && validTokens[key] && (!causal || key <= query))
                0f else blocked
        }
    }

    /** Expands one [T,T] attention bias into [heads,T,T]. */
    fun repeatHeads(bias: FloatArray, tokens: Int, heads: Int): FloatArray {
        require(tokens > 0 && heads > 0 && bias.size.toLong() == tokens.toLong() * tokens &&
            bias.size.toLong() * heads <= Int.MAX_VALUE)
        return FloatArray(bias.size * heads).also { output ->
            repeat(heads) { h -> bias.copyInto(output, h * bias.size) }
        }
    }
}
