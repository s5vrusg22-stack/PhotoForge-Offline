package com.example.photoforge

import java.io.File
import java.security.MessageDigest

object FluxSha256 {
    fun digest(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                if (count > 0) md.update(buffer, 0, count)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    fun verify(file: File, expected: String): Boolean {
        require(Regex("[0-9a-fA-F]{64}").matches(expected)) { "Invalid SHA-256" }
        return file.isFile && file.length() > 0L &&
            digest(file).equals(expected, ignoreCase = true)
    }
}
