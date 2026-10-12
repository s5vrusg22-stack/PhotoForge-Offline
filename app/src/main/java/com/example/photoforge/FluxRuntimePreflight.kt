package com.example.photoforge

import java.io.File

/** Fail-fast gate before any multi-GB FLUX graph is allocated. */
object FluxRuntimePreflight {
    data class Result(val ready: Boolean, val problems: List<String>)

    fun check(modelRoot: File, editing: Boolean = true): Result {
        val problems = mutableListOf<String>()
        val graphNames = GraphPipelineContract.required(
            if (editing) GraphPipelineContract.Mode.EDIT else GraphPipelineContract.Mode.GENERATE)
        graphNames.forEach { name ->
            val f = File(modelRoot, name)
            if (!f.isFile || f.length() <= 0L) problems += "missing graph: $name"
        }
        if (editing) {
            val timestepBin = File(modelRoot, FluxProjectedTimestepConfig.FILE_NAME)
            val timestepMeta = File(modelRoot, FluxProjectedTimestepConfig.META_NAME)
            if (!timestepBin.isFile || !timestepMeta.isFile) {
                problems += "missing verified FLUX timestep projection assets"
            }
            val vaeStats = File(modelRoot, "vae_stats_f32le.bin")
            if (!vaeStats.isFile || vaeStats.length() != 256L) {
                problems += "missing or invalid VAE 32-channel statistics"
            }
            try {
                val assets = QwenTokenizerAssets(modelRoot).requireComplete()
                if (assets.embeddings.length() !=
                    FluxEmbeddingLookup.VOCAB_SIZE.toLong() * FluxEmbeddingLookup.ROW_BYTES) {
                    problems += "Qwen embedding table size mismatch"
                }
            } catch (e: IllegalArgumentException) {
                problems += (e.message ?: "tokenizer assets invalid")
            }
        }
        return Result(problems.isEmpty(), problems)
    }

    fun requireReady(modelRoot: File, editing: Boolean = true) {
        val result = check(modelRoot, editing)
        require(result.ready) { result.problems.joinToString("; ") }
    }
}
