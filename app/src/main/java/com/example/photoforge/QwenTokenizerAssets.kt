package com.example.photoforge

import java.io.File

/**
 * Validates tokenizer assets and exposes special-token metadata.
 * Actual text -> token-ID encoding is intentionally blocked until the exact
 * upstream tokenizer serialization format is verified.
 */
class QwenTokenizerAssets(private val root: File) {
    data class Assets(
        val vocab: File,
        val merges: File,
        val special: File,
        val embeddings: File
    )
    fun requireComplete(): Assets {
        val tokenizer = File(root, "tokenizer").canonicalFile
        require(tokenizer.path.startsWith(root.canonicalPath + File.separator))
        fun required(name: String): File = File(tokenizer, name).canonicalFile.also {
            require(it.parentFile == tokenizer && it.isFile && it.length() > 0L) {
                "Missing tokenizer asset: tokenizer/$name"
            }
        }
        return Assets(
            required("qwen_vocab.txt"),
            required("qwen_merges.txt"),
            required("qwen_special.txt"),
            required("qwen_embed_fp16.bin")
        )
    }
}
