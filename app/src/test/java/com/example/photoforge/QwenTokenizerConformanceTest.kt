package com.example.photoforge

import org.junit.Test

class QwenTokenizerConformanceTest {
    private fun key(s: String) = QwenByteBpe.ByteKey(s.toByteArray(Charsets.UTF_8))

    @Test fun acceptsExactTokenIds() {
        val bpe = QwenByteBpe(mapOf(key("a") to 7), emptyMap())
        val tokenizer = QwenVerifiedTokenizer(bpe, padId = 0)
        QwenTokenizerConformance.verify(
            tokenizer, listOf(QwenTokenizerConformance.Vector("a", intArrayOf(7))))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsAnyTokenIdMismatch() {
        val bpe = QwenByteBpe(mapOf(key("a") to 7), emptyMap())
        val tokenizer = QwenVerifiedTokenizer(bpe, padId = 0)
        QwenTokenizerConformance.verify(
            tokenizer, listOf(QwenTokenizerConformance.Vector("a", intArrayOf(8))))
    }
}
