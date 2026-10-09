package com.example.photoforge

/**
 * Unicode-preserving, offline prompt input stage.
 * Does not pretend to tokenize or generate Qwen embeddings.
 */
object PromptInputPipeline {
    const val MAX_CODE_POINTS = 2048

    data class Prepared(val text: String, val codePoints: Int, val hasKorean: Boolean,
                        val hasEnglish: Boolean)

    fun prepare(raw: String): Prepared {
        val normalized = java.text.Normalizer.normalize(raw, java.text.Normalizer.Form.NFC)
            .trim()
        val count = normalized.codePointCount(0, normalized.length)
        require(count in 1..MAX_CODE_POINTS) {
            "Prompt must contain 1..$MAX_CODE_POINTS Unicode characters"
        }
        require(normalized.none { it == '\u0000' || (it.isISOControl() && it != '\n' && it != '\t') }) {
            "Prompt contains unsupported control characters"
        }
        val korean = normalized.any { it in '\uAC00'..'\uD7A3' || it in '\u1100'..'\u11FF' ||
            it in '\u3130'..'\u318F' }
        val english = normalized.any { it in 'A'..'Z' || it in 'a'..'z' }
        return Prepared(normalized, count, korean, english)
    }
}
