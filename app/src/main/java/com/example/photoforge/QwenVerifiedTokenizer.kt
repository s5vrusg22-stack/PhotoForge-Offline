package com.example.photoforge

/**
 * Connects Qwen pre-tokenization to byte-level BPE and the fixed 512-ID batch.
 * The BPE tables must be decoded from the exact exported model vocabulary.
 */
class QwenVerifiedTokenizer(
    private val bpe: QwenByteBpe,
    padId: Int,
    bosId: Int? = null,
    eosId: Int? = null
) {
    private val encoder = QwenTokenIdEncoder(::tokenize, padId, bosId, eosId)

    fun tokenize(prompt: String): IntArray {
        val pieces = QwenPreTokenizer.split(prompt)
        val ids = ArrayList<Int>()
        for (piece in pieces) {
            val encoded = bpe.encodePretoken(piece.toByteArray(Charsets.UTF_8))
            for (id in encoded) ids.add(id)
        }
        return ids.toIntArray()
    }

    fun encode(prompt: String): QwenPromptEncoding = encoder.encode(prompt)

    fun encodeAndEmbed(prompt: String, lookup: FluxEmbeddingLookup): Pair<QwenPromptEncoding, FloatArray> {
        val batch = encode(prompt)
        return batch to batch.embeddings(lookup)
    }
}
