package com.example.photoforge

import java.io.File

typealias FluxGraphRunner = (String, List<FloatArray>) -> List<FloatArray>

/** Final assembly point for verified model assets and the graph executor. */
object FluxEditRuntime {
    fun execute(
        modelRoot: File,
        runner: FluxGraphRunner,
        hostWithoutTime: FluxGraphExecutor.HostTensors
    ): FloatArray {
        FluxRuntimePreflight.requireReady(modelRoot, editing = true)
        val host = FluxTimestepInstaller.attachFromModelRoot(hostWithoutTime, modelRoot)
        return FluxGraphExecutor(runner).run(host)
    }
}
