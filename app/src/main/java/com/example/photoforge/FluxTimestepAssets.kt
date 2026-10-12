package com.example.photoforge

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * Loads four precomputed, checkpoint-projected FLUX Klein time embeddings.
 * Expected format: four consecutive little-endian float32 rows [1,3072].
 * A caller MUST provide the SHA-256 of a trusted reference export.
 * This does not pretend to derive the model's learned time_in weights.
 */
object FluxTimestepAssets {
    const val STEPS = 4
    const val WIDTH = 3072
    const val FILE_BYTES = STEPS * WIDTH * 4L

    fun load(file: File, expectedSha256: String): List<FloatArray> {
        require(expectedSha256.matches(Regex("[a-fA-F0-9]{64}"))) {
            "Trusted timestep SHA-256 required"
        }
        require(file.isFile && file.length() == FILE_BYTES) {
            "Expected $FILE_BYTES bytes for 4x3072 float32 timestep embeddings"
        }
        val bytes = file.readBytes()
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }
        require(digest.equals(expectedSha256, ignoreCase = true)) {
            "Timestep export SHA-256 mismatch"
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return List(STEPS) { index ->
            FloatArray(WIDTH) { buffer.float }.also {
                require(it.all(Float::isFinite)) {
                    "Nonfinite projected timestep at step $index"
                }
            }
        }
    }

    fun attach(
        host: FluxGraphExecutor.HostTensors,
        file: File,
        expectedSha256: String
    ): FluxGraphExecutor.HostTensors =
        host.copy(timeEmbeddings = load(file, expectedSha256))
}
