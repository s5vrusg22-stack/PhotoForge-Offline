package com.example.photoforge

/** Model-independent contract for future image-conditioned local generation. */
enum class GenerativeEditKind { EXPRESSION, PROP, CLOTHING, POSE }

data class GenerativeEditSpec(
    val kind: GenerativeEditKind,
    val prompt: String,
    val strength: Float,
    val preserveOutsideMask: Boolean = true
) {
    init {
        require(prompt.isNotBlank()) { "편집 지시문이 필요합니다." }
        require(strength in 0f..1f) { "강도는 0..1 사이여야 합니다." }
    }

    fun positivePrompt(): String {
        val preservation = "same person, preserve identity, same camera, same background, photorealistic, natural lighting"
        return when (kind) {
            GenerativeEditKind.EXPRESSION -> "$preservation, subtle realistic facial expression: $prompt"
            GenerativeEditKind.PROP -> "$preservation, add only the accessory: $prompt"
            GenerativeEditKind.CLOTHING -> "$preservation, change only clothing: $prompt"
            GenerativeEditKind.POSE -> "$preservation, natural anatomically correct body pose: $prompt"
        }
    }

    fun negativePrompt(): String =
        "different identity, changed background, face distortion, extra limbs, unnatural anatomy, blur, artifacts"
}

interface LocalGenerativeEditBackend {
    /** Must fail explicitly when model components are absent; never silently fake AI output. */
    fun isReady(): Boolean
    fun generate(
        image: android.graphics.Bitmap,
        mask: android.graphics.Bitmap,
        edit: GenerativeEditSpec
    ): android.graphics.Bitmap
}

object MissingGenerativeModelBackend : LocalGenerativeEditBackend {
    override fun isReady() = false
    override fun generate(
        image: android.graphics.Bitmap,
        mask: android.graphics.Bitmap,
        edit: GenerativeEditSpec
    ): android.graphics.Bitmap {
        error("표정/소품/의상/자세 생성형 모델 가중치 및 추론 파이프라인이 아직 설치되지 않았습니다.")
    }
}
