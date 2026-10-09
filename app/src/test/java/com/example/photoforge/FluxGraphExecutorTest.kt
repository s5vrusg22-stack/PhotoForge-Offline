package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test

class FluxGraphExecutorTest {
    @Test fun generationInvokesExpectedGraphsFourTimes() {
        val names = mutableListOf<String>()
        val runner = FluxGraphExecutor { name, inputs ->
            names += name
            when {
                name.startsWith("ke_enc") -> listOf(FloatArray(512 * 2560))
                name == "kc_prep.tflite" -> listOf(
                    FloatArray(256 * 3072), FloatArray(512 * 3072),
                    floatArrayOf(1f), floatArrayOf(1f), floatArrayOf(1f))
                name.startsWith("kc_double") -> listOf(inputs[0], inputs[1])
                name.startsWith("kc_single") -> listOf(inputs[0])
                name == "kc_final.tflite" -> listOf(FloatArray(256 * 128))
                name == "kv_vae.tflite" -> listOf(FloatArray(3 * 256 * 256))
                else -> error("Unexpected graph $name")
            }
        }
        val host = FluxGraphExecutor.HostTensors(
            inputsEmbeds = FloatArray(512 * 2560),
            encoderMask = floatArrayOf(0f),
            encoderCos = floatArrayOf(0f),
            encoderSin = floatArrayOf(0f),
            ditCos = floatArrayOf(0f),
            ditSin = floatArrayOf(0f),
            timeEmbeddings = List(4) { FloatArray(3072) },
            sigmas = floatArrayOf(1f, .75f, .5f, .25f, 0f),
            initialNoise = FloatArray(256 * 128),
            decodeLatents = { FloatArray(32 * 32 * 32) }
        )
        val image = runner.run(host)
        assertEquals(3 * 256 * 256, image.size)
        assertEquals(3, names.count { it.startsWith("ke_enc") })
        assertEquals(4, names.count { it == "kc_prep.tflite" })
        assertEquals(8, names.count { it.startsWith("kc_double") })
        assertEquals(16, names.count { it.startsWith("kc_single") })
        assertEquals(4, names.count { it == "kc_final.tflite" })
        val expected = mutableListOf("ke_enc0.tflite", "ke_enc1.tflite", "ke_enc2.tflite")
        repeat(4) {
            expected += listOf("kc_prep.tflite", "kc_double0.tflite", "kc_double1.tflite",
                "kc_single0.tflite", "kc_single1.tflite", "kc_single2.tflite",
                "kc_single3.tflite", "kc_final.tflite")
        }
        expected += "kv_vae.tflite"
        assertEquals(expected, names)
    }
    @Test fun editingInvokesVaeEncoderAndEditingGraphsInOrder() {
        val names = mutableListOf<String>()
        val runner = FluxGraphExecutor { name, inputs ->
            names += name
            when {
                name.startsWith("ke_enc") -> listOf(FloatArray(512 * 2560))
                name == "kv_vae_enc.tflite" -> listOf(FloatArray(32 * 32 * 32))
                name == "kce_prep.tflite" -> listOf(
                    FloatArray(512 * 3072), FloatArray(512 * 3072),
                    floatArrayOf(1f), floatArrayOf(1f), floatArrayOf(1f))
                name.startsWith("kce_double") -> listOf(inputs[0], inputs[1])
                name.startsWith("kce_single") -> listOf(inputs[0])
                name == "kce_final.tflite" -> listOf(FloatArray(512 * 128))
                name == "kv_vae.tflite" -> listOf(FloatArray(3 * 256 * 256))
                else -> error("Unexpected graph $name")
            }
        }
        val host = FluxGraphExecutor.HostTensors(
            inputsEmbeds = FloatArray(512 * 2560),
            encoderMask = floatArrayOf(0f), encoderCos = floatArrayOf(0f),
            encoderSin = floatArrayOf(0f), ditCos = floatArrayOf(0f),
            ditSin = floatArrayOf(0f),
            timeEmbeddings = List(4) { FloatArray(3072) },
            sigmas = floatArrayOf(1f, .75f, .5f, .25f, 0f),
            initialNoise = FloatArray(256 * 128),
            referenceImageChw = FloatArray(3 * 256 * 256),
            encodeReference = { FloatArray(256 * 128) },
            decodeLatents = { FloatArray(32 * 32 * 32) }
        )
        assertEquals(3 * 256 * 256, runner.run(host).size)
        val expected = mutableListOf("ke_enc0.tflite", "ke_enc1.tflite",
            "ke_enc2.tflite", "kv_vae_enc.tflite")
        repeat(4) {
            expected += listOf("kce_prep.tflite", "kce_double0.tflite",
                "kce_double1.tflite", "kce_single0.tflite", "kce_single1.tflite",
                "kce_single2.tflite", "kce_single3.tflite", "kce_final.tflite")
        }
        expected += "kv_vae.tflite"
        assertEquals(expected, names)
    }

}
