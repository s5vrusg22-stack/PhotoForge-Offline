package com.example.photoforge

import kotlin.math.exp

/**
 * Host-side flow-matching Euler scheduler for FLUX-family inference.
 * Not a complete FLUX.2 scheduler: exact upstream shift/timestep schedule
 * must be validated against the model export before production use.
 */
object FlowMatchScheduler {
    fun validateTimesteps(timesteps: FloatArray) {
        require(timesteps.isNotEmpty()) { "Empty timestep schedule" }
        require(timesteps.all { it.isFinite() && it in 0f..1f }) { "Invalid timestep" }
        for (i in 1 until timesteps.size) {
            require(timesteps[i] < timesteps[i - 1]) { "Timesteps must decrease strictly" }
        }
    }

    /** x_next = x + (sigma_next - sigma) * velocity; modifies x in-place. */
    fun eulerStep(latents: FloatArray, velocity: FloatArray, sigma: Float, nextSigma: Float) {
        require(latents.size == velocity.size) { "Latent/velocity tensor mismatch" }
        require(sigma.isFinite() && nextSigma.isFinite() && sigma in 0f..1f &&
                nextSigma in 0f..1f && nextSigma < sigma) { "Invalid sigma interval" }
        val dt = nextSigma - sigma
        for (i in latents.indices) {
            val v = velocity[i]
            require(v.isFinite() && latents[i].isFinite()) { "Non-finite latent/velocity" }
            latents[i] += dt * v
        }
    }

    fun linearSchedule(steps: Int): FloatArray {
        require(steps in 1..200) { "Unsupported number of steps" }
        return FloatArray(steps + 1) { i -> 1f - i.toFloat() / steps }
    }
}
