package com.example.photoforge

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.io.File
import java.util.Base64

class QwenBpeTablesTest {
    private fun line(bytes: String, n: Int) =
        Base64.getEncoder().encodeToString(bytes.toByteArray()) + " " + n

    @Test fun tablesDriveRealIdEncoder() {
        val vocab = File.createTempFile("qwen-vocab", ".txt")
        val merges = File.createTempFile("qwen-merges", ".txt")
        try {
            vocab.writeText(listOf(line("a",1), line("b",2), line("ab",7)).joinToString("\n"))
            merges.writeText(line("ab",0))
            val tables = QwenBpeTables.load(vocab, merges)
            val bpe = QwenByteBpe(tables.vocabulary, tables.mergeRanks)
            assertArrayEquals(intArrayOf(7), bpe.encodePretoken("ab".toByteArray()))
        } finally {
            vocab.delete(); merges.delete()
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun malformedTableFailsClosed() {
        val vocab = File.createTempFile("qwen-vocab", ".txt")
        val merges = File.createTempFile("qwen-merges", ".txt")
        try {
            vocab.writeText("not-valid")
            merges.writeText(line("ab",0))
            QwenBpeTables.load(vocab, merges)
        } finally {
            vocab.delete(); merges.delete()
        }
    }
}
