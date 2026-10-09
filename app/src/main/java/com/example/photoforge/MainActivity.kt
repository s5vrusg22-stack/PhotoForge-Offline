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
    private lateinit var maskView: MaskCanvas
    private var modelFile: java.io.File? = null
    private lateinit var status: TextView
    private var original: Bitmap? = null
    private var current: Bitmap? = null
    private var rotation = 0f
    private var mirrored = false
    private var brightness = 0
    private var localStrength = 0.45f
    private var hairTint = Color.rgb(125, 76, 49)
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
        maskView = MaskCanvas(this)
        root.addView(maskView, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        val scroll = ScrollView(this)
        val controls = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        fun addButton(label: String, action: () -> Unit) {
            controls.addView(Button(this).apply {
                text = label
                setOnClickListener { action() }
            })
        }
        addButton("사진 열기") { pick(pickImage) }
        addButton("ONNX 모델 선택 (LaMa 호환)") { pickModel() }
        addButton("마스크 한 획 취소") { maskView.undo() }
        addButton("마스크 전체 지우기") { maskView.clearMask() }
        addButton("브러시 작게") { maskView.brushPx = (maskView.brushPx - 12f).coerceAtLeast(12f) }
        addButton("브러시 크게") { maskView.brushPx = (maskView.brushPx + 12f).coerceAtMost(160f) }
        addButton("선택 영역만 AI로 지우기") { executeInpaint() }
        addButton("머리카락 색상: 갈색") { hairTint = Color.rgb(125, 76, 49); applyLocalTint() }
        addButton("머리카락 색상: 검정") { hairTint = Color.rgb(26, 25, 30); applyLocalTint() }
        addButton("머리카락 색상: 금발") { hairTint = Color.rgb(214, 176, 92); applyLocalTint() }
        addButton("부분 편집 강도: 약하게") { localStrength = 0.25f; status.text = "부분 편집 강도 25%" }
        addButton("부분 편집 강도: 보통") { localStrength = 0.45f; status.text = "부분 편집 강도 45%" }
        addButton("부분 편집 강도: 강하게") { localStrength = 0.70f; status.text = "부분 편집 강도 70%" }
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
        prepareBundledModel()
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
        if (requestCode == 102) { loadModel(data?.data); return }
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
            maskView.photo = result
            status.text = "${result.width} × ${result.height} · 밝기 ${brightness}"
        } catch (e: OutOfMemoryError) {
            Toast.makeText(this, "이미지가 너무 큽니다. 작은 사진으로 시도하세요.", Toast.LENGTH_LONG).show()
        }
    }


    private fun prepareBundledModel() {
        Thread {
            try {
                val target = java.io.File(filesDir, "bundled_lama_fp32.onnx")
                if (!target.exists() || target.length() == 0L) {
                    assets.open("lama_fp32.onnx").use { input ->
                        target.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
                    }
                }
                require(target.length() > 0L) { "모델 파일이 비어 있습니다." }
                modelFile = target
                runOnUiThread { status.text = "오프라인 AI 모델 준비 완료 · 사진을 선택하세요" }
            } catch (e: Exception) {
                runOnUiThread { status.text = "내장 모델 준비 실패: ${e.message}" }
            }
        }.start()
    }

    private fun pickModel() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "*/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }, 102)
    }

    private fun loadModel(uri: Uri?) {
        if (uri == null) return
        status.text = "모델 파일을 복사하는 중..."
        Thread {
            try {
                val target = java.io.File(filesDir, "inpaint.onnx")
                contentResolver.openInputStream(uri).use { input ->
                    requireNotNull(input) { "파일을 읽을 수 없습니다." }
                    target.outputStream().use { input.copyTo(it) }
                }
                require(target.length() > 0L) { "빈 파일입니다." }
                modelFile = target
                runOnUiThread { status.text = "모델 준비됨: ${target.length()/1048576} MB" }
            } catch (e: Exception) {
                runOnUiThread { status.text = "모델 읽기 실패: ${e.message}" }
            }
        }.start()
    }

    private fun applyLocalTint() {
        val source = current ?: return
        if (!maskView.hasMask()) {
            Toast.makeText(this, "머리카락 부분을 먼저 칠하세요.", Toast.LENGTH_LONG).show()
            return
        }
        val mask = maskView.exportMask()
        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val originalPixels = IntArray(source.width * source.height)
        val maskPixels = IntArray(source.width * source.height)
        source.getPixels(originalPixels, 0, source.width, 0, 0, source.width, source.height)
        mask.getPixels(maskPixels, 0, source.width, 0, 0, source.width, source.height)
        val tintHsv = FloatArray(3)
        val hsv = FloatArray(3)
        Color.colorToHSV(hairTint, tintHsv)
        for (i in originalPixels.indices) {
            val coverage = Color.red(maskPixels[i]) / 255f
            if (coverage <= 0f) continue
            val color = originalPixels[i]
            Color.colorToHSV(color, hsv)
            val tinted = Color.HSVToColor(Color.alpha(color), floatArrayOf(
                tintHsv[0], (hsv[1] * 0.3f + tintHsv[1] * 0.7f).coerceIn(0f, 1f), hsv[2]
            ))
            val blend = (localStrength * coverage).coerceIn(0f, 1f)
            fun mix(a: Int, b: Int): Int = (a * (1f - blend) + b * blend).toInt().coerceIn(0, 255)
            originalPixels[i] = Color.argb(Color.alpha(color),
                mix(Color.red(color), Color.red(tinted)),
                mix(Color.green(color), Color.green(tinted)),
                mix(Color.blue(color), Color.blue(tinted)))
        }
        output.setPixels(originalPixels, 0, source.width, 0, 0, source.width, source.height)
        original = output
        rotation = 0f
        mirrored = false
        brightness = 0
        render()
        status.text = "선택 영역 색상 변경 완료 · 다른 부분은 보존됨"
    }

    private fun executeInpaint() {
        val photo = current ?: return
        val model = modelFile
        if (model == null) {
            Toast.makeText(this, "먼저 LaMa 호환 ONNX 모델을 선택하세요.", Toast.LENGTH_LONG).show()
            return
        }
        if (!maskView.hasMask()) {
            Toast.makeText(this, "지울 영역을 손가락으로 칠하세요.", Toast.LENGTH_LONG).show()
            return
        }
        val mask = maskView.exportMask()
        status.text = "기기에서 AI 인페인팅 처리 중..."
        Thread {
            try {
                val output = InpaintEngine.run(model, photo, mask)
                runOnUiThread {
                    original = output
                    rotation = 0f
                    mirrored = false
                    brightness = 0
                    render()
                    status.text = "AI 인페인팅 완료 · PNG 저장 가능"
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "AI 실행 실패: ${e.message}" }
            }
        }.start()
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
