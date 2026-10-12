package com.example.photoforge

import java.text.Normalizer

class QwenTokenIdEncoder(
    private val encodeVerified: (String) -> IntArray,
    private val padTokenId: Int,
    private val bosTokenId: Int? = null,
    private val eosTokenId: Int? = null
) {
    init {
        require(padTokenId in 0 until FluxEmbeddingLookup.VOCAB_SIZE)
        listOfNotNull(bosTokenId, eosTokenId).forEach {
            require(it in 0 until FluxEmbeddingLookup.VOCAB_SIZE)
        }
    }

    fun encode(prompt: String): QwenPromptEncoding {
        val normalized = Normalizer.normalize(prompt.trim(), Normalizer.Form.NFC)
        require(normalized.isNotEmpty()) { "Prompt is empty" }
        val body = encodeVerified(normalized)
        require(body.all { it in 0 until FluxEmbeddingLookup.VOCAB_SIZE }) {
            "Tokenizer emitted token ID outside embedding vocabulary"
        }
        val extra = (if (bosTokenId == null) 0 else 1) + (if (eosTokenId == null) 0 else 1)
        require(body.size + extra <= FluxTensorRouting.TEXT_TOKENS) {
            "Prompt exceeds " + FluxTensorRouting.TEXT_TOKENS + " tokens"
        }
        val ids = IntArray(FluxTensorRouting.TEXT_TOKENS) { padTokenId }
        val valid = BooleanArray(FluxTensorRouting.TEXT_TOKENS)
        var cursor = 0
        bosTokenId?.let { ids[cursor] = it; valid[cursor++] = true }
        body.forEach { ids[cursor] = it; valid[cursor++] = true }
        eosTokenId?.let { ids[cursor] = it; valid[cursor++] = true }
        return QwenPromptEncoding(ids, valid)
    }
}
