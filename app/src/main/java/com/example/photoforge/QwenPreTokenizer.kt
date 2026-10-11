package com.example.photoforge

import java.text.Normalizer
import java.util.regex.Pattern

/**
 * Qwen2-compatible pre-token splitting before byte-level BPE.
 * Regex matches the upstream Qwen2Tokenizer pretokenization rule.
 */
object QwenPreTokenizer {
    private val regex = Pattern.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    )
    fun split(text: String): List<String> {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
        val matcher = regex.matcher(normalized)
        val result = ArrayList<String>()
        var end = 0
        while (matcher.find()) {
            require(matcher.start() == end) {
                "Unrecognized Qwen pretoken span at offset $end"
            }
            result.add(matcher.group())
            end = matcher.end()
        }
        require(end == normalized.length) { "Unrecognized trailing Qwen pretoken span" }
        return result
    }
}
