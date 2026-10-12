package com.example.photoforge

import android.content.Context
import java.io.File

/** Sequentially compiles every required edit graph to expose GPU/model incompatibility without running inference. */
object FluxModelHealthCheck {
    data class Report(
        val assetAudit: FluxRuntimeAudit.Report,
        val graphs: List<LiteRtGraphProbe.Result>
    ) {
        val ok: Boolean get() = assetAudit.ok && graphs.all { it.success }
        val failedGraphs: List<String> get() = graphs.filterNot { it.success }.map { it.graph }
    }

    fun run(context: Context, modelRoot: File): Report {
        val audit = FluxRuntimeAudit.audit(modelRoot)
        if (!audit.ok) return Report(audit, emptyList())
        val results = GraphPipelineContract.required(GraphPipelineContract.Mode.EDIT).map { name ->
            LiteRtGraphProbe.compileGpu(context, File(modelRoot, name))
        }
        return Report(audit, results)
    }
}
