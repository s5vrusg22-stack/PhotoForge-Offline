package com.example.photoforge

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class QwenByteBpeTest {
    private fun key(value: String) = QwenByteBpe.ByteKey(value.toByteArray(Charsets.UTF_8))

    @Test fun mergesLowestRankPairFirst() {
        val vocab = mapOf(key("a") to 1, key("b") to 2, key("c") to 3,
            key("ab") to 4, key("bc") to 5)
        val ranks = mapOf(key("ab") to 0, key("bc") to 1)
        assertArrayEquals(intArrayOf(4, 3),
            QwenByteBpe(vocab, ranks).encodePretoken("abc".toByteArray()))
    }

    @Test fun noMergePreservesBytes() {
        val vocab = mapOf(key("a") to 1, key("b") to 2)
        assertArrayEquals(intArrayOf(1, 2),
            QwenByteBpe(vocab, emptyMap()).encodePretoken("ab".toByteArray()))
    }

    @Test(expected = IllegalStateException::class)
    fun missingVocabularyFailsClosed() {
        QwenByteBpe(emptyMap(), emptyMap()).encodePretoken(byteArrayOf(42))
    }
}
