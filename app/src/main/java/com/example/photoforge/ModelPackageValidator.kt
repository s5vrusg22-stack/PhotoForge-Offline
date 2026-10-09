package com.example.photoforge

import java.io.File
import java.security.MessageDigest

/**
 * Verifies the selected model package against pinned upstream SHA-256 digests.
 * Rejects missing files, symlink escapes, truncated files and mismatched hashes.
 * Manifest parsing is intentionally separate: caller must provide trusted entries.
 */
object ModelPackageValidator {
    data class Asset(val path: String, val size: Long, val sha256: String)
    data class Validation(val verified: Int, val errors: List<String>) {
        val ok get() = errors.isEmpty()
    }

    fun verify(root: File, mode: GraphPipelineContract.Mode, assets: List<Asset>): Validation {
        val errors = mutableListOf<String>()
        val required = GraphPipelineContract.required(mode)
        val byName = assets.groupBy { it.path }
        val rootCanonical = root.canonicalFile
        for (name in required) {
            val matches = byName[name]
            if (matches == null) { errors += "Missing manifest entry: $name"; continue }
            if (matches.size != 1) { errors += "Duplicate manifest entry: $name"; continue }
            val entry = matches.single()
            if (entry.size <= 0 || !Regex("[0-9a-f]{64}").matches(entry.sha256)) {
                errors += "Invalid metadata: $name"; continue
            }
            val file = File(root, name).canonicalFile
            if (!file.path.startsWith(rootCanonical.path + File.separator)) {
                errors += "Path escapes model root: $name"; continue
            }
            if (!file.isFile || file.length() != entry.size) {
                errors += "Missing or wrong size: $name"; continue
            }
            val hash = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val chunk = ByteArray(1024 * 1024)
                while (true) {
                    val count = input.read(chunk)
                    if (count < 0) break
                    hash.update(chunk, 0, count)
                }
            }
            val actual = hash.digest().joinToString("") { "%02x".format(it) }
            if (actual != entry.sha256) errors += "SHA256 mismatch: $name"
        }
        return Validation(required.size - errors.size, errors)
    }
}
