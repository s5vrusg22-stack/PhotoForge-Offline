package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test

class PromptInputPipelineTest {
    @Test fun preservesMixedKoreanEnglish() {
        val result = PromptInputPipeline.prepare("  눈물 어린 smile  ")
        assertEquals("눈물 어린 smile", result.text)
        assertTrue(result.hasKorean)
        assertTrue(result.hasEnglish)
    }

    @Test fun normalizesDecomposedHangul() {
        assertEquals("가", PromptInputPipeline.prepare("\u1100\u1161").text)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsBlank() { PromptInputPipeline.prepare("   ") }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsControlCharacters() { PromptInputPipeline.prepare("hello\u0000world") }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOversizedPrompt() { PromptInputPipeline.prepare("a".repeat(2049)) }

    @Test fun countsEmojiAsOneCodePoint() {
        assertEquals(2, PromptInputPipeline.prepare("웃🙂").codePoints)
    }
}
