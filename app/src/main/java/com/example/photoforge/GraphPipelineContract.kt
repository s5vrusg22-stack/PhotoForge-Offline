package com.example.photoforge

/**
 * Explicit stage order and graph requirements for model package validation.
 * Shapes and tensor routing MUST be sourced from actual TFLite signatures;
 * this class deliberately does not invent dimensions or run dummy inference.
 */
object GraphPipelineContract {
    enum class Mode { EDIT, GENERATE }
    val textGraphs = listOf("ke_enc0.tflite", "ke_enc1.tflite", "ke_enc2.tflite")
    val editGraphs = listOf("kce_prep.tflite", "kce_double0.tflite", "kce_double1.tflite",
        "kce_single0.tflite", "kce_single1.tflite", "kce_single2.tflite",
        "kce_single3.tflite", "kce_final.tflite")
    val generateGraphs = listOf("kc_prep.tflite", "kc_double0.tflite", "kc_double1.tflite",
        "kc_single0.tflite", "kc_single1.tflite", "kc_single2.tflite",
        "kc_single3.tflite", "kc_final.tflite")
    val tokenizerFiles = listOf("tokenizer/qwen_vocab.txt", "tokenizer/qwen_merges.txt",
        "tokenizer/qwen_special.txt", "tokenizer/qwen_embed_fp16.bin")

    fun required(mode: Mode): List<String> = textGraphs +
        (if (mode == Mode.EDIT) listOf("kv_vae_enc.tflite") + editGraphs
         else generateGraphs) + "kv_vae.tflite" + tokenizerFiles

    fun missing(mode: Mode, present: Set<String>): List<String> =
        required(mode).filterNot { it in present }

    /**
     * Encoder: token ids -> text embeddings.
     * Edit: source image -> VAE encoder -> conditioned denoiser.
     * Generation: noise latents -> denoiser.
     * Both: scheduler integrates velocity -> VAE decoder.
     *
     * Exact host ops (Qwen tokenization, rotary embeddings, packing,
     * normalization and tensor names) remain mandatory blockers.
     */
}
