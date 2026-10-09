package com.example.photoforge

import java.io.File

/**
 * Production entry gate. Qwen stays disabled on user request.
 * Never call the graph runner with placeholder tokenizer/mask/rotary inputs.
 */
object FluxInferenceEntry {
    fun validateGraphFiles(directory: File, editing: Boolean) {
        require(directory.isDirectory) { "FLUX model directory does not exist" }
        val prefix = if (editing) "kce" else "kc"
        val names = buildList {
            for (i in 0..2) add("ke_enc$i.tflite")
            add("${prefix}_prep.tflite")
            for (i in 0..1) add("${prefix}_double$i.tflite")
            for (i in 0..3) add("${prefix}_single$i.tflite")
            add("${prefix}_final.tflite")
            add("kv_vae.tflite")
            if (editing) add("kv_vae_enc.tflite")
        }
        val root = directory.canonicalFile
        val missing = names.filter { name ->
            val graph = File(root, name).canonicalFile
            graph.parentFile != root || !graph.isFile || graph.length() == 0L
        }
        require(missing.isEmpty()) {
            "Missing FLUX graph files: ${missing.joinToString(", ")}"
        }
    }

    fun runWithVerifiedHostTensors(
        modelDirectory: File,
        host: FluxGraphExecutor.HostTensors,
        onStage: (String) -> Unit = {}
    ): FloatArray {
        require(modelDirectory.isDirectory) { "FLUX model directory does not exist" }
        validateGraphFiles(modelDirectory, host.referenceImageChw != null)
        // Verified embeddings may be provided by an external host; the disabled
        // Qwen text-encoding feature must not block graph-only inference.
        LiteRtSequentialRunner(modelDirectory).use { gpu ->
            return FluxGraphExecutor(gpu::run).run(host, onStage)
        }
    }
}
