package com.example.photoforge

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File

/**
 * Runs one exported graph at a time. FP32 is required by the published
 * FLUX.2 Klein GPU contract; buffer and model lifetimes are bounded per call.
 */
class LiteRtSequentialRunner(private val root: File) : AutoCloseable {

    fun run(name: String, tensors: List<FloatArray>): List<FloatArray> {
        require(name.matches(Regex("(ke_enc[0-2]|kc_(prep|double[01]|single[0-3]|final)|kce_(prep|double[01]|single[0-3]|final)|kv_vae(_enc)?)\\.tflite"))) {
            "Unexpected graph name"
        }
        val rootPath = root.toPath()
        require(java.nio.file.Files.isDirectory(rootPath) &&
            !java.nio.file.Files.isSymbolicLink(rootPath)) {
            "Invalid or symlinked FLUX model directory"
        }
        val candidate = File(root, name)
        require(!java.nio.file.Files.isSymbolicLink(candidate.toPath())) {
            "Symlinked FLUX graph is not allowed: $name"
        }
        val file = candidate.canonicalFile
        require(file.parentFile == root.canonicalFile && file.isFile && file.length() > 0L) {
            "Missing or empty model graph: $name"
        }
        val options = CompiledModel.Options(Accelerator.GPU)
        val model = CompiledModel.create(file.absolutePath, options)
        try {
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            try {
                require(inputs.size == tensors.size) {
                    "$name expects ${inputs.size} inputs, got ${tensors.size}"
                }
                tensors.forEachIndexed { i, tensor ->
                    require(tensor.isNotEmpty() && tensor.all { it.isFinite() }) {
                        "$name input[$i] must be a nonempty finite FP32 tensor"
                    }
                    try {
                        inputs[i].writeFloat(tensor)
                    } catch (e: IllegalArgumentException) {
                        throw IllegalArgumentException("$name input[$i] rejected ${tensor.size} FP32 elements; verify exported tensor signature", e)
                    }
                }
                model.run(inputs, outputs)
                return outputs.mapIndexed { i, buffer ->
                    val values = buffer.readFloat()
                    require(values.isNotEmpty() && values.all { it.isFinite() }) {
                        "$name output[$i] is empty or contains nonfinite values"
                    }
                    values
                }
            } finally {
                inputs.forEach { it.close() }
                outputs.forEach { it.close() }
            }
        } finally {
            model.close()
        }
    }

    override fun close() { /* Graph models and buffers are closed per invocation. */ }
}
