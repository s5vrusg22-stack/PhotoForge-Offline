package com.example.photoforge

/**
 * Real graph invocation orchestration. Caller supplies upstream-compatible host
 * tensors; this class never invents masks, rotary tables, embeddings or VAE stats.
 * A GraphRunner must run each graph and close its GPU buffers before returning.
 */
class FluxGraphExecutor(private val runner: (String, List<FloatArray>) -> List<FloatArray>) {
    data class HostTensors(
        val inputsEmbeds: FloatArray,
        val encoderMask: FloatArray,
        val encoderCos: FloatArray,
        val encoderSin: FloatArray,
        val ditCos: FloatArray,
        val ditSin: FloatArray,
        val timeEmbeddings: List<FloatArray>,
        val sigmas: FloatArray,
        val initialNoise: FloatArray,
        val referenceImageChw: FloatArray? = null,
        val encodeReference: (FloatArray) -> FloatArray = { error("Reference VAE packing unavailable") },
        val decodeLatents: (FloatArray) -> FloatArray = { error("Latent unpack and VAE normalization unavailable") }
    )

    private fun invoke(name: String, inputs: List<FloatArray>, outputs: Int): List<FloatArray> {
        val result = runner("$name.tflite", inputs)
        check(result.size == outputs) { "$name returned ${result.size} outputs, expected $outputs" }
        check(result.all { tensor -> tensor.all { it.isFinite() } }) { "$name produced nonfinite tensor" }
        return result
    }

    fun run(host: HostTensors, onStage: (String) -> Unit = {}): FloatArray {
        val editing = host.referenceImageChw != null
        // Do not allocate multi-GB GPU graphs when mandatory VAE transforms are absent.
        val vaeProbe = host.decodeLatents(host.initialNoise.copyOf())
        FluxTensorRouting.requireShape("VAE decoder input preflight", vaeProbe, 1, 32, 32, 32)
        if (editing) {
            val referenceProbe = host.encodeReference(FloatArray(32 * 32 * 32))
            FluxTensorRouting.requireShape("VAE reference tokens preflight", referenceProbe, 1, 256, 128)
        }
        val mode = if (editing) "kce" else "kc"
        FluxTensorRouting.requireShape("input embeddings", host.inputsEmbeds, 1, 512, 2560)
        FluxTensorRouting.requireShape("noise", host.initialNoise, 1, 256, 128)
        require(host.timeEmbeddings.size == 4 && host.sigmas.size == 5) {
            "FLUX Klein requires four denoising steps and five sigmas"
        }
        FlowMatchScheduler.validateTimesteps(host.sigmas)
        // Fail before loading GPU graphs if caller-supplied conditioning is absent
        // or corrupt. Exact dimensions are model-export-specific and checked
        // against each LiteRT graph input signature at invocation time.
        listOf(
            "encoder mask" to host.encoderMask,
            "encoder rotary cosine" to host.encoderCos,
            "encoder rotary sine" to host.encoderSin,
            "DiT rotary cosine" to host.ditCos,
            "DiT rotary sine" to host.ditSin
        ).forEach { (name, tensor) ->
            require(tensor.isNotEmpty() && tensor.all { it.isFinite() }) {
                "$name must be a nonempty finite tensor"
            }
        }
        require(host.sigmas.last() == 0f) {
            "Final diffusion sigma must be zero before VAE decoding"
        }
        host.timeEmbeddings.forEachIndexed { index, value ->
            FluxTensorRouting.requireShape("temb$index", value, 1, 3072)
        }

        var hidden = host.inputsEmbeds
        val taps = ArrayList<FloatArray>(3)
        for (i in 0..2) {
            onStage("ke_enc$i")
            hidden = invoke("ke_enc$i", listOf(hidden, host.encoderMask,
                host.encoderCos, host.encoderSin), 1)[0]
            FluxTensorRouting.requireShape("ke_enc$i output", hidden, 1, 512, 2560)
            taps.add(hidden)
        }
        val prompt = FluxTensorRouting.interleaveTextTaps(taps)
        val reference = host.referenceImageChw?.let { image ->
            FluxTensorRouting.requireShape("reference image", image, 1, 3, 256, 256)
            onStage("kv_vae_enc")
            val encoded = invoke("kv_vae_enc", listOf(image), 1)[0]
            FluxTensorRouting.requireShape("VAE encoded", encoded, 1, 32, 32, 32)
            host.encodeReference(encoded).also {
                FluxTensorRouting.requireShape("reference tokens", it, 1, 256, 128)
            }
        }
        val noise = host.initialNoise.copyOf()
        repeat(4) { step ->
            val modelInput = if (reference == null) noise.copyOf()
                else FluxTensorRouting.concatenateNoiseReference(noise, reference)
            onStage("${mode}_prep step $step")
            val prep = invoke("${mode}_prep", listOf(modelInput, prompt,
                host.timeEmbeddings[step]), 5)
            var image = prep[0]
            var text = prep[1]
            val modImage = prep[2]
            val modText = prep[3]
            val modSingle = prep[4]
            FluxTensorRouting.requireShape("prep image", image, 1,
                if (editing) 512 else 256, 3072)
            FluxTensorRouting.requireShape("prep text", text, 1, 512, 3072)
            repeat(2) { i ->
                onStage("${mode}_double$i step $step")
                val pair = invoke("${mode}_double$i", listOf(image, text,
                    host.ditCos, host.ditSin, modImage, modText), 2)
                image = pair[0]
                text = pair[1]
                FluxTensorRouting.requireShape("double image", image, 1,
                    if (editing) 512 else 256, 3072)
                FluxTensorRouting.requireShape("double text", text, 1, 512, 3072)
            }
            var joint = FluxTensorRouting.joinTextImage(text, image, editing)
            repeat(4) { i ->
                onStage("${mode}_single$i step $step")
                joint = invoke("${mode}_single$i", listOf(joint, host.ditCos,
                    host.ditSin, modSingle), 1)[0]
                FluxTensorRouting.requireShape("single joint", joint, 1,
                    if (editing) 1024 else 768, 3072)
            }
            onStage("${mode}_final step $step")
            val predicted = invoke("${mode}_final", listOf(joint,
                host.timeEmbeddings[step]), 1)[0]
            FluxTensorRouting.applyVelocity(noise, predicted,
                host.sigmas[step], host.sigmas[step + 1])
        }
        onStage("kv_vae")
        val vaeInput = host.decodeLatents(noise)
        FluxTensorRouting.requireShape("VAE decoder input", vaeInput, 1, 32, 32, 32)
        val output = invoke("kv_vae", listOf(vaeInput), 1)[0]
        FluxTensorRouting.requireShape("VAE output", output, 1, 3, 256, 256)
        return output
    }
}
