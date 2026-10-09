package com.example.photoforge

import android.app.Activity
import android.os.Bundle
import android.content.Intent
import android.graphics.ImageDecoder
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import android.view.ViewGroup

class MainActivity : Activity() {
    private lateinit var preview: ImageView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 24)
        }
        root.addView(TextView(this).apply {
            text = "PhotoForge Offline\n사진 편집 개발 버전"
            textSize = 24f
        })
        root.addView(Button(this).apply {
            text = "사진 열기"
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    type = "image/*"
                    addCategory(Intent.CATEGORY_OPENABLE)
                }, 100)
            }
        })
        preview = ImageView(this).apply { adjustViewBounds = true }
        root.addView(preview, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        root.addView(TextView(this).apply {
            text = "기본 빌드 검증 버전입니다. AI 모델은 아직 포함되지 않았습니다."
        })
        setContentView(root)
    }
    @Deprecated("Deprecated in Android")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100 && resultCode == RESULT_OK) {
            val uri = data?.data ?: return
            try {
                preview.setImageBitmap(ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)))
            } catch (e: Exception) {
                Toast.makeText(this, "사진 열기 실패: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
}
