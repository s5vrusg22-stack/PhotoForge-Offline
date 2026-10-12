package com.example.photoforge

import org.junit.Test

class FluxExecutionGuardTest {
    private fun host() = FluxGraphExecutor.HostTensors(
        inputsEmbeds = FloatArray(512 * 2560),
        encoderMask = floatArrayOf(0f),
        encoderCos = floatArrayOf(1f), encoderSin = floatArrayOf(0f),
        ditCos = floatArrayOf(1f), ditSin = floatArrayOf(0f),
        timeEmbeddings = List(4) { FloatArray(3072) },
        sigmas = floatArrayOf(1f, .75f, .5f, .25f, 0f),
        initialNoise = FloatArray(256 * 128)
    )
    @Test fun acceptsInternallyConsistentContract() { FluxExecutionGuard.validate(host()) }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsRotaryPairMismatch() {
        FluxExecutionGuard.validate(host().copy(ditSin = floatArrayOf(0f, 0f)))
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonzeroTerminalSigma() {
        FluxExecutionGuard.validate(host().copy(sigmas=floatArrayOf(1f,.8f,.6f,.4f,.2f)))
    }
}
