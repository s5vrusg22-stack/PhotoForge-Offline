package com.example.photoforge

import java.io.File

/** Production text front-end: verified Qwen IDs -> fixed batch -> fp16 embedding table. */
class QwenEmbeddingPipeline private constructor(
    private val tokenizer: QwenVerifiedTokenizer,
    private val lookup: FluxEmbeddingLookup
) : AutoCloseable {
    data class Output(
        val tokenIds: IntArray,
        val validTokens: BooleanArray,
        val embeddings: FloatArray,
        val attentionBias: FloatArray
    )
    fun encode(prompt: String): Output {
        val batch = tokenizer.encode(prompt)
        val embeddings = batch.embeddings(lookup)
        require(embeddings.size == FluxTensorRouting.TEXT_TOKENS * FluxEmbeddingLookup.HIDDEN_SIZE)
        require(embeddings.all { it.isFinite() }) { "Non-finite Qwen embeddings" }
        return Output(batch.tokenIds, batch.validTokens, embeddings, batch.attentionBias())
    }
    override fun close() = lookup.close()
    companion object {
        fun open(modelRoot: File, padId: Int, bosId: Int? = null, eosId: Int? = null): QwenEmbeddingPipeline {
            val pair = QwenTokenizerFactory.create(modelRoot, padId, bosId, eosId)
            return QwenEmbeddingPipeline(pair.first, pair.second)
        }
    }
}
