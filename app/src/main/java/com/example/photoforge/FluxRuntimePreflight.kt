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
        if (editing) {\n            val timestepBin = File(modelRoot, FluxProjectedTimestepConfig.FILE_NAME)\n            val timestepMeta = File(modelRoot, FluxProjectedTimestepConfig.META_NAME)\n            if (!timestepBin.isFile || !timestepMeta.isFile) {\n                problems += "missing verified FLUX timestep projection assets"\n            }\n            try {
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
