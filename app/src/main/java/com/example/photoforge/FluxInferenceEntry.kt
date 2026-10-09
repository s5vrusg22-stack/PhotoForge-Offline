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
        require(modelDirectory.isDirectory) { "FLUX model directory does not exist" }
        // Verified embeddings may be provided by an external host; the disabled
        // Qwen text-encoding feature must not block graph-only inference.
        LiteRtSequentialRunner(modelDirectory).use { gpu ->
            return FluxGraphExecutor(gpu::run).run(host, onStage)
        }
    }
}
