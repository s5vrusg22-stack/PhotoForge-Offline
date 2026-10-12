package com.example.photoforge

/** Cross-checks all caller-provided host tensors before any GPU graph is created. */
object FluxExecutionGuard {
    fun validate(host: FluxGraphExecutor.HostTensors) {
        FluxTensorRouting.requireShape("Qwen embeddings", host.inputsEmbeds, 1, 512, 2560)
        FluxTensorRouting.requireShape("initial noise", host.initialNoise, 1, 256, 128)
        require(host.sigmas.size == 5) { "Expected five sigma boundaries" }
        FlowMatchScheduler.validateTimesteps(host.sigmas)
        require(host.sigmas.last() == 0f) { "Final sigma must be zero" }
        require(host.timeEmbeddings.size == 4) { "Expected four projected timestep rows" }
        host.timeEmbeddings.forEachIndexed { i, t ->
            FluxTensorRouting.requireShape("projected timestep $i", t, 1, 3072)
        }
        listOf(host.encoderMask, host.encoderCos, host.encoderSin, host.ditCos, host.ditSin)
            .forEachIndexed { i, t -> require(t.isNotEmpty() && t.all(Float::isFinite)) {
                "Conditioning tensor $i is empty/nonfinite"
            }}
        require(host.encoderCos.size == host.encoderSin.size) { "Encoder rotary pair mismatch" }
        require(host.ditCos.size == host.ditSin.size) { "DiT rotary pair mismatch" }
        host.referenceImageChw?.let {
            FluxTensorRouting.requireShape("reference image", it, 1, 3, 256, 256)
        }
    }
}
