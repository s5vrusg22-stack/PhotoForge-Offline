package com.example.photoforge

import java.io.File

/**
 * Production entry gate. Qwen stays disabled on user request.
 * Never call the graph runner with placeholder tokenizer/mask/rotary inputs.
 */
object FluxInferenceEntry {
    fun runWithVerifiedHostTensors(
        modelDirectory: File,
        host: FluxGraphExecutor.HostTensors,
        onStage: (String) -> Unit = {}
    ): FloatArray {
        ExperimentalModelFeatures.requireQwenEnabled()
        LiteRtSequentialRunner(modelDirectory).use { gpu ->
            return FluxGraphExecutor(gpu::run).run(host, onStage)
        }
    }
}
