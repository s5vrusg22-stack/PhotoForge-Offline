package com.example.photoforge

object QwenOfficialTokenVectors {
    val english = QwenTokenizerConformance.Vector(
        "Turn the apple into a green apple",
        intArrayOf(19389, 279, 23268, 1119, 264, 6176, 23268)
    )
    val korean = QwenTokenizerConformance.Vector(
        "이 사람을 자연스럽게 웃는 표정으로 바꿔줘",
        intArrayOf(12802, 124889, 17877, 140577, 24897, 134798, 57801, 85413, 225,
            16560, 139052, 29281, 42039, 81718, 144379, 144131)
    )
    val all = listOf(english, korean)
}
