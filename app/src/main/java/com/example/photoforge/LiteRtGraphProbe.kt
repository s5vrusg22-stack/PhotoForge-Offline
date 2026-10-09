package com.example.photoforge

import android.content.Context
import android.os.Debug
import android.os.SystemClock
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File

/**
 * Real GPU graph compile/load probe, not a diffusion inference pipeline.
 * Uses a single graph at a time and closes it on every path.
 */
object LiteRtGraphProbe {
    data class Result(
        val graph: String,
        val success: Boolean,
        val milliseconds: Long,
        val pssKbBefore: Int,
        val pssKbAfter: Int,
        val error: String?
    )

    fun compileGpu(context: Context, model: File): Result {
        require(model.isFile && model.length() > 0L) { "그래프 파일이 없습니다." }
        require(model.extension == "tflite") { "TFLite 그래프를 선택하세요." }
        val before = Debug.getPss().toInt()
        val start = SystemClock.elapsedRealtime()
        return try {
            CompiledModel.create(
                model.absolutePath,
                CompiledModel.Options(Accelerator.GPU)
            ).use { compiled ->
                // Loading/compiling only. Real run requires graph-specific tensors.
                val inputs = compiled.createInputBuffers()
                try {
                    check(inputs.isNotEmpty()) { "입력 텐서 없음" }
                } finally {
                    inputs.forEach { it.close() }
                }
            }
            Result(model.name, true, SystemClock.elapsedRealtime() - start,
                before, Debug.getPss().toInt(), null)
        } catch (e: Exception) {
            Result(model.name, false, SystemClock.elapsedRealtime() - start,
                before, Debug.getPss().toInt(), e.javaClass.simpleName + ": " + e.message)
        } catch (e: OutOfMemoryError) {
            Result(model.name, false, SystemClock.elapsedRealtime() - start,
                before, Debug.getPss().toInt(), "GPU/메모리 부족: " + e.message)
        }
    }
}
