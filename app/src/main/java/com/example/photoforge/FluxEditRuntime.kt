package com.example.photoforge

import java.io.File

/** Final assembly point for verified model assets and the graph executor. */
object FluxEditRuntime {
    fun execute(
        modelRoot: File,
        runner: FluxGraphExecutor.GraphRunner,
        hostWithoutTime: FluxGraphExecutor.HostTensors
    ): FloatArray {
        FluxRuntimePreflight.requireReady(modelRoot, editing = true)
        val host = FluxTimestepInstaller.attachFromModelRoot(hostWithoutTime, modelRoot)
        return FluxGraphExecutor(runner).execute(host)
    }
}
