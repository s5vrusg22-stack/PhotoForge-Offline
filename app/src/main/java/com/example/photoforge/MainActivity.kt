package com.example.photoforge

import android.app.Activity
import android.os.Bundle
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import android.view.ViewGroup
import java.io.ByteArrayOutputStream

class MainActivity : Activity() {
    private lateinit var preview: ImageView
    private lateinit var status: TextView
    private var original: Bitmap? = null
    private var current: Bitmap? = null
    private var rotation = 0f
    private var mirrored = false
    private var brightness = 0
    private val pickImage = 100
    private val pickOverlay = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(18, 30, 18, 16)
            setBackgroundColor(Color.rgb(19, 23, 32))
        }
        val title = TextView(this).apply {
            text = "PhotoForge Offline"
            textSize = 25f
            setTextColor(Color.WHITE)
        }
        root.addView(title)
        status = TextView(this).apply {
            text = "사진을 열어 편집하세요. 인터넷 연결이 필요하지 않습니다."
            setTextColor(Color.LTGRAY)
        }
        root.addView(status)
        preview = ImageView(this).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.rgb(35, 39, 48))
        }
        root.addView(preview, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        val scroll = ScrollView(this)
        val controls = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        fun addButton(label: String, action: () -> Unit) {
            controls.addView(Button(this).apply {
                text = label
                setOnClickListener { action() }
            })
        }
        addButton("사진 열기") { pick(pickImage) }
        addButton("PNG 소품 이미지 겹치기 (중앙)") { pick(pickOverlay) }
        addButton("오른쪽 90° 회전") { if (original != null) { rotation += 90f; render() } }
        addButton("좌우 반전") { if (original != null) { mirrored = !mirrored; render() } }
        addButton("밝기 +15") { if (original != null) { brightness = (brightness + 15).coerceAtMost(90); render() } }
        addButton("밝기 -15") { if (original != null) { brightness = (brightness - 15).coerceAtLeast(-90); render() } }
        addButton("원본으로 초기화") { rotation = 0f; mirrored = false; brightness = 0; render() }
        addButton("PNG로 저장") { saveImage() }
        controls.addView(TextView(this).apply {
            text = "AI 생성 기능은 아직 포함되지 않았습니다. 현재 버전은 오프라인 사진 편집기입니다."
            setTextColor(Color.LTGRAY)
        })
        scroll.addView(controls)
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 0.75f))
        setContentView(root)
    }

    private fun pick(request: Int) {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "image/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }, request)
    }

    @Deprecated("Deprecated in Android")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        val uri: Uri = data?.data ?: return
        try {
            val decoded = contentResolver.openInputStream(uri).use { input ->
                BitmapFactory.decodeStream(input) ?: error("지원되지 않는 이미지")
            }
            if (requestCode == pickImage) {
                original = decoded.copy(Bitmap.Config.ARGB_8888, false)
                rotation = 0f
                mirrored = false
                brightness = 0
                render()
            } else if (requestCode == pickOverlay) {
                val base = current ?: return
                val merged = Bitmap.createBitmap(base.width, base.height, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(merged)
                canvas.drawBitmap(base, 0f, 0f, null)
                val scale = minOf(base.width * 0.5f / decoded.width, base.height * 0.5f / decoded.height)
                val w = decoded.width * scale
                val h = decoded.height * scale
                val dst = android.graphics.RectF((base.width-w)/2, (base.height-h)/2, (base.width+w)/2, (base.height+h)/2)
                canvas.drawBitmap(decoded, null, dst, Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG))
                original = merged
                rotation = 0f
                mirrored = false
                brightness = 0
                render()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "이미지 처리 실패: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun render() {
        val source = original ?: return
        try {
            val matrix = Matrix().apply {
                if (mirrored) postScale(-1f, 1f)
                postRotate(rotation)
            }
            val transformed = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
            val result = if (brightness == 0) transformed else {
                val output = Bitmap.createBitmap(transformed.width, transformed.height, Bitmap.Config.ARGB_8888)
                val paint = Paint().apply {
                    colorFilter = android.graphics.ColorMatrixColorFilter(floatArrayOf(
                        1f,0f,0f,0f,brightness.toFloat(),
                        0f,1f,0f,0f,brightness.toFloat(),
                        0f,0f,1f,0f,brightness.toFloat(),
                        0f,0f,0f,1f,0f
                    ))
                }
                Canvas(output).drawBitmap(transformed, 0f, 0f, paint)
                output
            }
            current = result
            preview.setImageBitmap(result)
            status.text = "${result.width} × ${result.height} · 밝기 ${brightness}"
        } catch (e: OutOfMemoryError) {
            Toast.makeText(this, "이미지가 너무 큽니다. 작은 사진으로 시도하세요.", Toast.LENGTH_LONG).show()
        }
    }

    private fun saveImage() {
        val image = current ?: return
        try {
            val values = android.content.ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "PhotoForge_${System.currentTimeMillis()}.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PhotoForge")
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("저장 공간 접근 실패")
            contentResolver.openOutputStream(uri).use { out ->
                if (out == null || !image.compress(Bitmap.CompressFormat.PNG, 100, out)) error("PNG 저장 실패")
            }
            Toast.makeText(this, "Pictures/PhotoForge에 저장했습니다", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "저장 실패: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
}
