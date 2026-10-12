package com.example.photoforge

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

class FluxTimestepAssetsTest {
    @Test fun loadsVerifiedLittleEndianProjectedRows() {
        val file = File.createTempFile("flux-temb", ".bin")
        try {
            val buffer = ByteBuffer.allocate(FluxTimestepAssets.FILE_BYTES.toInt())
                .order(ByteOrder.LITTLE_ENDIAN)
            repeat(FluxTimestepAssets.STEPS * FluxTimestepAssets.WIDTH) {
                buffer.putFloat(it.toFloat() / 1000f)
            }
            val bytes = buffer.array()
            file.writeBytes(bytes)
            val sha = MessageDigest.getInstance("SHA-256").digest(bytes)
                .joinToString("") { "%02x".format(it.toInt() and 255) }
            val result = FluxTimestepAssets.load(file, sha)
            assertEquals(4, result.size)
            assertEquals(3072, result[0].size)
            assertEquals(0f, result[0][0], 0f)
            assertEquals(3.072f, result[1][0], 0.0001f)
        } finally { file.delete() }
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsIncorrectDigest() {
        val file = File.createTempFile("flux-temb", ".bin")
        try {
            file.writeBytes(ByteArray(FluxTimestepAssets.FILE_BYTES.toInt()))
            FluxTimestepAssets.load(file, "0".repeat(64))
        } finally { file.delete() }
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsMissingProjection() {
        val file = File.createTempFile("flux-temb", ".bin")
        try {
            FluxTimestepAssets.load(file, "0".repeat(64))
        } finally { file.delete() }
    }
}
