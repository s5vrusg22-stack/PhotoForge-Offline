package com.example.photoforge

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** SHA-256 manifest verification for exported conditioning tensors. */
object FluxConditioningManifest {
    const val NAME = "conditioning_manifest.json"
    private val required = listOf(
        "encoder_mask_f32le.bin",
        "encoder_rotary_cos_f32le.bin", "encoder_rotary_sin_f32le.bin",
        "dit_rotary_cos_f32le.bin", "dit_rotary_sin_f32le.bin"
    )
    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
    fun verify(root: File) {
        val manifestFile = File(root, NAME)
        require(manifestFile.isFile && manifestFile.length() in 1..65536) { "Missing conditioning manifest" }
        val json = JSONObject(manifestFile.readText(Charsets.UTF_8))
        require(json.getString("dtype") == "float32" && json.getString("byte_order") == "little") {
            "Unsupported conditioning encoding"
        }
        val hashes = json.getJSONObject("sha256")
        required.forEach { name ->
            val file = File(root, name)
            require(file.isFile) { "Missing conditioning asset $name" }
            val expected = hashes.getString(name).lowercase()
            require(expected.matches(Regex("[0-9a-f]{64}"))) { "Invalid SHA-256 for $name" }
            require(sha256(file) == expected) { "Conditioning SHA-256 mismatch: $name" }
        }
    }
}
