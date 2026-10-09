package com.example.photoforge

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * Runs one exported graph at a time. FP32 is required by the published
 * FLUX.2 Klein GPU contract; buffer and model lifetimes are bounded per call.
 */
class LiteRtSequentialRunner(private val root: File) : AutoCloseable {
    private val environment = Environment.create()

    fun run(name: String, tensors: List<FloatArray>): List<FloatArray> {
        require(name.matches(Regex("(ke_enc[0-2]|kc_(prep|double[01]|single[0-3]|final)|kce_(prep|double[01]|single[0-3]|final)|kv_vae(_enc)?)\\.tflite"))) {
            "Unexpected graph name"
        }
        val file = File(root, name).canonicalFile
        require(file.parentFile == root.canonicalFile && file.isFile) { "Missing model graph: $name" }
        val options = CompiledModel.Options(Accelerator.GPU)
        options.gpuOptions = CompiledModel.GpuOptions(
            precision = CompiledModel.GpuOptions.Precision.FP32)
        val model = CompiledModel.create(file.absolutePath, options, environment)
        try {
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            try {
                require(inputs.size == tensors.size) {
                    "$name expects ${inputs.size} inputs, got ${tensors.size}"
                }
                tensors.forEachIndexed { i, tensor -> inputs[i].writeFloat(tensor) }
                model.run(inputs, outputs)
                return outputs.map { it.readFloat() }
            } finally {
                inputs.forEach { it.close() }
                outputs.forEach { it.close() }
            }
        } finally {
            model.close()
        }
    }

    override fun close() {
        environment.close()
    }
}
