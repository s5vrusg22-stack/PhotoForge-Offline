package com.example.photoforge

import java.nio.charset.StandardCharsets

/**
 * Byte-level ranked BPE core. Vocabulary decoding, GPT/Qwen pre-tokenization,
 * chat template and special token parsing must be supplied by a verified loader.
 * Never use this directly on an entire prompt without pre-tokenization.
 */
class QwenByteBpe(
    private val vocabulary: Map<ByteKey, Int>,
    private val mergeRanks: Map<ByteKey, Int>
) {
    class ByteKey(bytes: ByteArray) {
        private val value = bytes.copyOf()
        fun bytes(): ByteArray = value.copyOf()
        override fun equals(other: Any?): Boolean =
            other is ByteKey && value.contentEquals(other.value)
        override fun hashCode(): Int = value.contentHashCode()
    }

    fun encodePretoken(bytes: ByteArray): IntArray {
        if (bytes.isEmpty()) return IntArray(0)
        val parts = bytes.map { byteArrayOf(it) }.toMutableList()
        while (parts.size > 1) {
            var bestIndex = -1
            var bestRank = Int.MAX_VALUE
            for (i in 0 until parts.lastIndex) {
                val rank = mergeRanks[ByteKey(parts[i] + parts[i + 1])] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if (bestIndex < 0) break
            parts[bestIndex] = parts[bestIndex] + parts[bestIndex + 1]
            parts.removeAt(bestIndex + 1)
        }
        return parts.map { part ->
            vocabulary[ByteKey(part)] ?: error(
                "BPE vocabulary missing merged byte sequence; upstream assets incompatible"
            )
        }.toIntArray()
    }
}
