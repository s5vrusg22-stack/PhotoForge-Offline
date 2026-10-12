package com.example.photoforge

import java.io.File
import java.util.Base64

object QwenBpeTables {
    data class Tables(
        val vocabulary: Map<QwenByteBpe.ByteKey, Int>,
        val mergeRanks: Map<QwenByteBpe.ByteKey, Int>
    )
    fun load(vocabFile: File, mergesFile: File): Tables {
        require(vocabFile.isFile && mergesFile.isFile) { "Qwen BPE tables missing" }
        val vocabulary = parse(vocabFile, "vocab")
        val ranks = parse(mergesFile, "merges")
        require(vocabulary.isNotEmpty() && ranks.isNotEmpty()) { "Empty Qwen BPE table" }
        require(vocabulary.values.all { it in 0 until FluxEmbeddingLookup.VOCAB_SIZE }) {
            "Qwen vocab ID outside embedding table"
        }
        return Tables(vocabulary, ranks)
    }
    private fun parse(file: File, label: String): Map<QwenByteBpe.ByteKey, Int> {
        val output = LinkedHashMap<QwenByteBpe.ByteKey, Int>()
        file.useLines(Charsets.UTF_8) { lines ->
            lines.forEachIndexed { index, raw ->
                val line = raw.trim()
                if (line.isEmpty() || line.startsWith("#")) return@forEachIndexed
                val split = line.lastIndexOf(' ')
                require(split > 0 && split < line.lastIndex) {
                    "Malformed Qwen " + label + " line " + (index + 1)
                }
                val bytes = try { Base64.getDecoder().decode(line.substring(0, split)) }
                    catch (e: IllegalArgumentException) {
                        throw IllegalArgumentException("Invalid base64 in Qwen " + label + " line " + (index + 1), e)
                    }
                require(bytes.isNotEmpty()) { "Empty byte token in Qwen " + label }
                val number = line.substring(split + 1).toIntOrNull()
                    ?: throw IllegalArgumentException("Invalid integer in Qwen " + label + " line " + (index + 1))
                require(number >= 0) { "Negative value in Qwen " + label }
                val previous = output.put(QwenByteBpe.ByteKey(bytes), number)
                require(previous == null) { "Duplicate byte token in Qwen " + label }
            }
        }
        return output
    }
}
