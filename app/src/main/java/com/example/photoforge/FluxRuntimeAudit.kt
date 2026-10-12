package com.example.photoforge

import java.io.File
import java.security.MessageDigest

/** Central fail-fast audit for the production edit runtime. */
object FluxRuntimeAudit {
    data class Report(val checks: List<Pair<String, Boolean>>) {
        val ok get() = checks.all { it.second }
        fun requireOk() = require(ok) {
            checks.filterNot { it.second }.joinToString("; ") { "failed: ${it.first}" }
        }
    }

    fun audit(root: File): Report {
        val checks = mutableListOf<Pair<String, Boolean>>()
        fun add(name: String, value: Boolean) { checks += name to value }
        add("model root", root.isDirectory && !java.nio.file.Files.isSymbolicLink(root.toPath()))
        GraphPipelineContract.required(GraphPipelineContract.Mode.EDIT).forEach { name ->
            val f = File(root, name)
            add("graph $name", f.isFile && f.length() > 0L && !java.nio.file.Files.isSymbolicLink(f.toPath()))
        }
        val tb = File(root, FluxProjectedTimestepConfig.FILE_NAME)
        val tm = File(root, FluxProjectedTimestepConfig.META_NAME)
        add("timestep binary size", tb.isFile && tb.length() == FluxTimestepAssets.FILE_BYTES)
        add("timestep metadata", tm.isFile && tm.length() in 1..65536)
        val vs = File(root, "vae_stats_f32le.bin")
        add("VAE stats size", vs.isFile && vs.length() == 256L)
        try { FluxVaeStatistics.load(vs); add("VAE stats finite", true) }
        catch (_: Exception) { add("VAE stats finite", false) }
        try { QwenTokenizerAssets(root).requireComplete(); add("Qwen assets", true) }
        catch (_: Exception) { add("Qwen assets", false) }
        return Report(checks)
    }
}
