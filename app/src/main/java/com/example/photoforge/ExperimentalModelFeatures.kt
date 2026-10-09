package com.example.photoforge

/** Opt-in gates: Qwen prompt path remains disabled until requested. */
object ExperimentalModelFeatures {
    const val QWEN_TEXT_ENCODER_ENABLED = false

    fun requireQwenEnabled() {
        check(QWEN_TEXT_ENCODER_ENABLED) {
            "Qwen/QN text encoding is disabled by user request; prompt-based FLUX inference unavailable."
        }
    }
}
