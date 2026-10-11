package com.example.photoforge

import java.io.File

/** Builds the production tokenizer only from validated model-package assets. */
object QwenTokenizerFactory {
    fun create(modelRoot: File, padId: Int, bosId: Int? = null, eosId: Int? = null):
            Pair<QwenVerifiedTokenizer, FluxEmbeddingLookup> {
        val assets = QwenTokenizerAssets(modelRoot).requireComplete()
        val tables = QwenBpeTables.load(assets.vocab, assets.merges)
        val tokenizer = QwenVerifiedTokenizer(
            QwenByteBpe(tables.vocabulary, tables.mergeRanks),
            padId, bosId, eosId)
        return tokenizer to FluxEmbeddingLookup(assets.embeddings)
    }
}
