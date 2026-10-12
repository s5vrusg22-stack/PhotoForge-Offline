package com.example.photoforge

/** Fixed-length token batch used by the exported Qwen LiteRT encoder. */
data class QwenPromptEncoding(
    val tokenIds: IntArray,
    val validTokens: BooleanArray
) {
    init {
        require(tokenIds.size == FluxTensorRouting.TEXT_TOKENS)
        require(validTokens.size == FluxTensorRouting.TEXT_TOKENS)
        require(tokenIds.indices.all { tokenIds[it] >= 0 })
    }
    fun attentionBias(heads: Int = 32): FloatArray {
        val base = FluxHostMath.attentionBias(validTokens, causal = true)
        return FluxHostMath.repeatHeads(base, FluxTensorRouting.TEXT_TOKENS, heads)
    }
    fun embeddings(lookup: FluxEmbeddingLookup): FloatArray = lookup.lookup(tokenIds)
}
