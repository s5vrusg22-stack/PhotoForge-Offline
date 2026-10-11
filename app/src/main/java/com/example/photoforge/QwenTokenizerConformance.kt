package com.example.photoforge

/**
 * Fail-closed comparison between Kotlin token IDs and an upstream-generated
 * conformance vector. Production inference must not proceed on mismatch.
 */
object QwenTokenizerConformance {
    data class Vector(val prompt: String, val ids: IntArray)

    fun verify(tokenizer: QwenVerifiedTokenizer, vectors: List<Vector>) {
        require(vectors.isNotEmpty()) { "No upstream tokenizer conformance vectors" }
        vectors.forEachIndexed { index, vector ->
            require(vector.prompt.isNotEmpty() && vector.ids.isNotEmpty()) {
                "Invalid tokenizer vector $index"
            }
            val actual = tokenizer.tokenize(vector.prompt)
            require(actual.contentEquals(vector.ids)) {
                "Qwen tokenizer mismatch at vector $index: expected " +
                    vector.ids.joinToString(",") + " got " + actual.joinToString(",")
            }
        }
    }
}
