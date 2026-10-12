package com.example.photoforge

import java.io.File

/** Final assembly point for verified model assets and the graph executor. */
object FluxEditRuntime {
    typealias GraphRunner = (String, List<FloatArray>) -> List<FloatArray>

    fun execute(
        modelRoot: File,
        runner: GraphRunner,
        hostWithoutTime: FluxGraphExecutor.HostTensors
    ): FloatArray {
        FluxRuntimePreflight.requireReady(modelRoot, editing = true)
        val host = FluxTimestepInstaller.attachFromModelRoot(hostWithoutTime, modelRoot)
        return FluxGraphExecutor(runner).run(host)
    }
}
