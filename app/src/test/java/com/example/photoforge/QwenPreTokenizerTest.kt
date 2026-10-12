package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test

class QwenPreTokenizerTest {
    @Test fun separatesEnglishContractionsAndPunctuation() {
        assertEquals(listOf("I", "'m", " happy", "!"),
            QwenPreTokenizer.split("I'm happy!"))
    }
    @Test fun preservesKoreanAndCombiningMarks() {
        assertEquals(listOf("안녕하세요", " 세계", "!"),
            QwenPreTokenizer.split("안녕하세요 세계!"))
    }
    @Test fun keepsDigitsSeparate() {
        assertEquals(listOf("a", "1", "2"), QwenPreTokenizer.split("a12"))
    }
    @Test fun normalizesUnicodeNfc() {
        assertEquals(listOf("é"), QwenPreTokenizer.split("e\u0301"))
    }
}
