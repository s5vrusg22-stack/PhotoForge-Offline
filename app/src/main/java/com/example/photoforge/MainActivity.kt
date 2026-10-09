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
    private lateinit var promptInput: android.widget.EditText
    private lateinit var expressionInput: android.widget.EditText
    private var expressionReference: Bitmap? = null
    private val pickExpressionReference = 103
    private var original: Bitmap? = null
    private var current: Bitmap? = null
    private var rotation = 0f
    private var mirrored = false
    private var brightness = 0
    private var editMode = "표정 변경"
    private var expression = "자연스러운 미소"
    private var expressionStrength = 0.25f
    private var undoSnapshot: Bitmap? = null
    private val pickImage = 100
    private val pickOverlay = 101
    private val pickLiteRtGraph = 104
    private val pickFluxDirectory = 105

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
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(8, 8, 8, 24)
        }
        fun addButton(label: String, action: () -> Unit) {
            controls.addView(Button(this).apply {
                text = label
                setOnClickListener { action() }
            })
        }
        addButton("사진 열기") { pick(pickImage) }
        controls.addView(TextView(this).apply {
            text = "1. 표정 변경 · 최우선"
            textSize = 19f
            setTextColor(Color.WHITE)
        })
        expressionInput = android.widget.EditText(this).apply {
            hint = "표정 직접 입력: 눈물이 맺힌 채 살짝 웃기"
            setSingleLine(false)
            minLines = 2
            setTextColor(Color.WHITE)
            setHintTextColor(Color.LTGRAY)
        }
        controls.addView(expressionInput)
        expressionInput.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                val raw = s?.toString().orEmpty()
                status.text = if (raw.isBlank()) "한국어 또는 영어 프롬프트를 입력하세요."
                else try {
                    val prepared = PromptInputPipeline.prepare(raw)
                    "프롬프트 준비됨: ${prepared.codePoints}자 · 토크나이저 연결 대기"
                } catch (e: IllegalArgumentException) {
                    "프롬프트 오류: ${e.message}"
                }
            }
            override fun afterTextChanged(s: android.text.Editable?) {}
        })
        addButton("표정 예시 사진 선택") { pick(pickExpressionReference) }
        addButton("입력한 표정 사용") {
            val description = expressionInput.text.toString().trim()
            if (description.isBlank()) {
                status.text = "원하는 표정을 문장으로 입력하세요."
            } else {
                selectExpression(description)
            }
        }
        addButton("🙂 자연스러운 미소") { selectExpression("자연스러운 미소") }
        addButton("😄 활짝 웃기") { selectExpression("활짝 웃기") }
        addButton("😐 무표정") { selectExpression("무표정") }
        addButton("😢 슬픈 표정") { selectExpression("슬픈 표정") }
        addButton("😮 놀란 표정") { selectExpression("놀란 표정") }
        addButton("😠 화난 표정") { selectExpression("화난 표정") }
        addButton("표정 변화: 약하게") { expressionStrength = 0.15f; updateExpressionStatus() }
        addButton("표정 변화: 보통") { expressionStrength = 0.35f; updateExpressionStatus() }
        addButton("표정 변화: 강하게") { expressionStrength = 0.65f; updateExpressionStatus() }
        addButton("선택한 표정으로 AI 수정") { requestExpressionEdit() }
        controls.addView(TextView(this).apply {
            text = "2. 소품 추가 · 3. 옷 변경 · 4. 자세 변경"
            textSize = 17f
            setTextColor(Color.WHITE)
        })
        promptInput = android.widget.EditText(this).apply {
            hint = "추가할 소품 입력: 안경, 모자, 목걸이"
            setSingleLine(true)
            setTextColor(Color.WHITE)
            setHintTextColor(Color.LTGRAY)
            inputType = android.text.InputType.TYPE_CLASS_TEXT
        }
        controls.addView(promptInput)
        addButton("입력한 소품 적용 (기본 도형)") { applyPromptProp() }
        controls.addView(TextView(this).apply { text = "소품 위치: 사진에서 원하는 곳을 손가락으로 칠한 후 적용하세요."; setTextColor(Color.LTGRAY) })
        addButton("ONNX 모델 선택 (LaMa 호환)") { pickModel() }
        addButton("FLUX 모델 폴더 가져오기 (6GB 이상 공간 필요)") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            }, pickFluxDirectory)
        }
        addButton("FLUX 모델 12개 설치 확인") {
            val directory = java.io.File(filesDir, "flux_models")
            try {
                FluxInferenceEntry.validateGraphFiles(directory, false)
                status.text = "FLUX 생성용 그래프 12개 확인됨 · GPU 추론은 별도 검증 필요"
            } catch (e: Exception) { status.text = "FLUX 파일 검사 실패: ${e.message}" }
        }
        addButton("기기 메모리·발열 상태 확인") {
            try {
                val snapshot = DevicePerformanceMonitor.sample(this)
                status.text = snapshot.summary() + " · GPU 사용률은 측정되지 않음"
            } catch (e: Exception) {
                status.text = "기기 상태 조회 실패: ${e.message}"
            }
        }
        addButton("설치된 FLUX 그래프 GPU 컴파일 검사") {
            val graph = java.io.File(filesDir, "flux_models/kc_prep.tflite")
            if (!graph.isFile || graph.length() == 0L) {
                status.text = "FLUX 모델 폴더를 먼저 가져오세요."
            } else {
                status.text = "FLUX kc_prep GPU 컴파일 검사 중..."
                Thread {
                    val result = try {
                        LiteRtGraphProbe.compileGpu(this, graph)
                    } catch (e: Exception) {
                        LiteRtGraphProbe.Result("kc_prep", false, 0, 0, 0, e.message)
                    } catch (e: OutOfMemoryError) {
                        LiteRtGraphProbe.Result("kc_prep", false, 0, 0, 0, "GPU 메모리 부족")
                    }
                    runOnUiThread {
                        status.text = if (result.success)
                            "FLUX kc_prep GPU 그래프 컴파일 성공 (${result.milliseconds}ms) · 추론 실행은 별도"
                        else "FLUX kc_prep GPU 컴파일 실패: ${result.error}"
                    }
                }.start()
            }
        }
        addButton("LiteRT GPU 그래프 로딩 테스트") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"
                addCategory(Intent.CATEGORY_OPENABLE)
            }, pickLiteRtGraph)
        }
        addButton("마스크 한 획 취소") { maskView.undo() }
        addButton("마스크 전체 지우기") { maskView.clearMask() }
        addButton("브러시 작게") { maskView.brushPx = (maskView.brushPx - 12f).coerceAtLeast(12f) }
        addButton("브러시 크게") { maskView.brushPx = (maskView.brushPx + 12f).coerceAtMost(160f) }
        addButton("선택 영역만 AI로 지우기") { executeInpaint() }
        addButton("표정 변경 모드") { editMode = "표정 변경"; updateExpressionStatus() }
        addButton("소품 추가 모드") { editMode = "소품 추가"; status.text = "소품 PNG를 선택해 합성하세요." }
        addButton("옷 변경 모드") { editMode = "옷 변경"; status.text = "옷 부분을 마스크로 지정하세요. AI 의상 생성 모델은 아직 연결되지 않았습니다." }
        addButton("자세 변경 모드") { editMode = "자세 변경"; status.text = "자세 생성 모델은 아직 연결되지 않았습니다. 원본 보존을 우선합니다." }
        addButton("한 단계 되돌리기") {
            undoSnapshot?.let { original = it; undoSnapshot = null; rotation = 0f; mirrored = false; brightness = 0; render() }
        }
        addButton("PNG 소품 이미지 겹치기 (중앙)") { pick(pickOverlay) }
        addButton("오른쪽 90° 회전") { if (original != null) { rotation += 90f; render() } }
        addButton("좌우 반전") { if (original != null) { mirrored = !mirrored; render() } }
        addButton("밝기 +15") { if (original != null) { brightness = (brightness + 15).coerceAtMost(90); render() } }
        addButton("밝기 -15") { if (original != null) { brightness = (brightness - 15).coerceAtLeast(-90); render() } }
        addButton("원본으로 초기화") { rotation = 0f; mirrored = false; brightness = 0; render() }
        addButton("PNG로 저장") { saveImage() }
        controls.addView(TextView(this).apply {
            text = "소품 단어 입력은 현재 기본 도형 3종만 지원합니다. 자유로운 AI 소품 생성 및 의상·자세 생성은 모델 통합 전입니다."
            setTextColor(Color.LTGRAY)
        })
        scroll.addView(controls)
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.35f))
        setContentView(root)
        prepareBundledModel()
    }

    private fun selectExpression(value: String) {
        editMode = "표정 변경"
        expression = value
        updateExpressionStatus()
    }

    private fun updateExpressionStatus() {
        status.text = "표정: $expression · 강도 ${(expressionStrength * 100).toInt()}% · AI 생성 모델 연결 전"
    }

    private fun requestExpressionEdit() {
        if (current == null) {
            Toast.makeText(this, "먼저 사진을 열어주세요.", Toast.LENGTH_SHORT).show()
            return
        }
        editMode = "표정 변경"
        val custom = expressionInput.text.toString().trim()
        if (custom.isNotBlank()) {
            expression = try { PromptInputPipeline.prepare(custom).text }
            catch (e: IllegalArgumentException) {
                status.text = "프롬프트 오류: ${e.message}"
                return
            }
        }
        val referenceInfo = if (expressionReference != null) " · 예시 사진 준비됨" else ""
        status.text = "표정 '$expression' 선택됨$referenceInfo. 실제 얼굴 표정 생성 AI는 아직 연결되지 않아 사진을 변경하지 않았습니다."
    }

    private fun applyPromptProp() {
        val source = current ?: run {
            Toast.makeText(this, "먼저 사진을 열어주세요.", Toast.LENGTH_SHORT).show()
            return
        }
        if (editMode != "소품 추가") {
            status.text = "옷/자세의 AI 생성 모델은 아직 연결되지 않았습니다."
            return
        }
        val prompt = promptInput.text.toString().trim()
        if (prompt.isEmpty()) {
            status.text = "소품 이름을 입력하세요."
            return
        }
        val layer = PromptPropRenderer.render(prompt, source.width, source.height)
        if (layer == null) {
            status.text = "지원되는 기본 소품: 안경, 모자, 목걸이. 임의 단어 AI 생성은 아직 미지원."
            return
        }
        val merged = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(merged)
        // A painted mask is an optional placement target. Without one, use image center.
        if (maskView.hasMask()) {
            val placementMask = maskView.exportMask()
            val pixels = IntArray(source.width * source.height)
            placementMask.getPixels(pixels, 0, source.width, 0, 0, source.width, source.height)
            var sx = 0.0
            var sy = 0.0
            var count = 0
            for (i in pixels.indices) {
                if (Color.red(pixels[i]) > 127) {
                    sx += (i % source.width)
                    sy += (i / source.width)
                    count++
                }
            }
            if (count > 0) {
                val dx = (sx / count).toFloat() - source.width / 2f
                val dy = (sy / count).toFloat() - source.height / 2f
                canvas.drawBitmap(layer, dx, dy, null)
            } else {
                canvas.drawBitmap(layer, 0f, 0f, null)
            }
        } else {
            canvas.drawBitmap(layer, 0f, 0f, null)
        }
        undoSnapshot = source.copy(Bitmap.Config.ARGB_8888, false)
        original = merged
        rotation = 0f
        mirrored = false
        brightness = 0
        render()
        status.text = "기본 도형 소품 적용됨 · AI 생성 아님 · 되돌리기 가능"
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
        if (requestCode == pickFluxDirectory) {
            data?.data?.let { importFluxDirectory(it) }
            return
        }
        if (requestCode == pickLiteRtGraph) {
            val uri = data?.data ?: return
            status.text = "LiteRT GPU 그래프 파일을 읽고 있습니다..."
            Thread {
                val result = try {
                    val graph = java.io.File(cacheDir, "photoforge_graph_probe.tflite")
                    contentResolver.openInputStream(uri).use { input ->
                        requireNotNull(input) { "파일을 열 수 없습니다." }
                        graph.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
                    }
                    LiteRtGraphProbe.compileGpu(this, graph)
                } catch (e: Exception) {
                    LiteRtGraphProbe.Result("선택 파일", false, 0, 0, 0, e.message)
                } catch (e: OutOfMemoryError) {
                    LiteRtGraphProbe.Result("선택 파일", false, 0, 0, 0, "메모리 부족: " + e.message)
                }
                runOnUiThread {
                    status.text = if (result.success)
                        "GPU 그래프 로딩 성공: ${result.graph}, ${result.milliseconds}ms, PSS ${result.pssKbBefore}→${result.pssKbAfter}KB (이미지 생성 아님)"
                    else "GPU 그래프 로딩 실패: ${result.error}"
                }
            }.start()
            return
        }
        val uri: Uri = data?.data ?: return
        try {
            val decoded = decodeScaledImage(uri)
            if (requestCode == pickExpressionReference) {
                expressionReference = decoded.copy(Bitmap.Config.ARGB_8888, false)
                status.text = "표정 예시 사진 준비됨 · 실제 참조 이미지 AI 적용은 모델 연동 전입니다."
                return
            }
            if (requestCode == pickImage) {
                maskView.clearMask()
                undoSnapshot = null
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
                undoSnapshot = base.copy(Bitmap.Config.ARGB_8888, false)
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

    private fun importFluxDirectory(treeUri: Uri) {
        status.text = "FLUX 모델 파일을 검사하고 복사하는 중... (약 6.6GB)"
        Thread {
            val directory = java.io.File(filesDir, "flux_models")
            try {
                val rootId = android.provider.DocumentsContract.getTreeDocumentId(treeUri)
                val childrenUri = android.provider.DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, rootId)
                val available = mutableMapOf<String, String>()
                contentResolver.query(childrenUri, arrayOf(
                    android.provider.DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                    android.provider.DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                    android.provider.DocumentsContract.Document.COLUMN_MIME_TYPE
                ), null, null, null)?.use { cursor ->
                    while (cursor.moveToNext()) {
                        if (cursor.getString(2) != android.provider.DocumentsContract.Document.MIME_TYPE_DIR) {
                            available[cursor.getString(1)] = cursor.getString(0)
                        }
                    }
                }
                val names = buildList {
                    for (i in 0..2) add("ke_enc$i.tflite")
                    add("kc_prep.tflite")
                    for (i in 0..1) add("kc_double$i.tflite")
                    for (i in 0..3) add("kc_single$i.tflite")
                    add("kc_final.tflite")
                    add("kv_vae.tflite")
                }
                val missing = names.filterNot { available.containsKey(it) }
                require(missing.isEmpty()) { "폴더에 없는 모델: ${missing.joinToString()}" }
                // Fail before copying multi-GB graphs if internal storage is insufficient.
                // Existing files are not counted, but remain subject to later validation.
                var requiredBytes = 0L
                for (name in names) {
                    val existing = java.io.File(directory, name)
                    if (existing.isFile && existing.length() > 0L) continue
                    val id = available.getValue(name)
                    val uri = android.provider.DocumentsContract.buildDocumentUriUsingTree(treeUri, id)
                    contentResolver.query(uri, arrayOf(android.provider.DocumentsContract.Document.COLUMN_SIZE),
                        null, null, null)?.use { cursor ->
                        if (cursor.moveToFirst() && !cursor.isNull(0)) {
                            val bytes = cursor.getLong(0)
                            require(bytes > 0) { "모델 크기가 잘못됨: $name" }
                            requiredBytes = Math.addExact(requiredBytes, bytes)
                        } else {
                            throw IllegalStateException("모델 크기를 확인할 수 없음: $name")
                        }
                    } ?: throw IllegalStateException("모델 크기 조회 실패: $name")
                }
                val availableBytes = android.os.StatFs(filesDir.absolutePath).availableBytes
                val reserve = 512L * 1024 * 1024
                require(requiredBytes <= availableBytes - reserve) {
                    "저장 공간 부족: 필요 ${requiredBytes / 1048576} MiB, 여유 ${availableBytes / 1048576} MiB (512 MiB 예약)"
                }
                require(directory.isDirectory || directory.mkdirs()) { "모델 저장 폴더 생성 실패" }
                for ((index, name) in names.withIndex()) {
                    val id = available.getValue(name)
                    val uri = android.provider.DocumentsContract.buildDocumentUriUsingTree(treeUri, id)
                    val target = java.io.File(directory, name)
                    val partial = java.io.File(directory, "$name.partial")
                    if (target.isFile && target.length() > 0L) continue
                    runOnUiThread { status.text = "FLUX 모델 복사 중 ${index + 1}/${names.size}: $name" }
                    try {
                        contentResolver.openInputStream(uri).use { input ->
                            requireNotNull(input) { "읽을 수 없는 모델: $name" }
                            partial.outputStream().buffered(1024 * 1024).use { output ->
                                input.copyTo(output, 1024 * 1024)
                            }
                        }
                        require(partial.length() > 0L) { "빈 모델: $name" }
                        require(partial.renameTo(target)) { "모델 저장 실패: $name" }
                    } finally {
                        partial.delete()
                    }
                }
                FluxInferenceEntry.validateGraphFiles(directory, false)
                runOnUiThread {
                    status.text = "FLUX 모델 12개 설치 완료 · 실제 GPU 추론은 아직 검증되지 않음"
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "FLUX 모델 설치 실패: ${e.message}" }
            } catch (e: OutOfMemoryError) {
                runOnUiThread { status.text = "FLUX 모델 설치 중 메모리 부족" }
            }
        }.start()
    }

    private fun decodeScaledImage(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "파일을 열 수 없습니다." }
            BitmapFactory.decodeStream(input, null, bounds)
        }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "지원되지 않는 이미지" }
        var sample = 1
        while (bounds.outWidth / sample > 2048 || bounds.outHeight / sample > 2048) {
            sample *= 2
        }
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "파일을 다시 열 수 없습니다." }
            BitmapFactory.decodeStream(input, null, options) ?: error("이미지 디코딩 실패")
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
                val available = assets.list("")?.contains("lama_fp32.onnx") == true
                if (!available) {
                    runOnUiThread {
                        status.text = "기본 편집 준비 완료 · LaMa 모델 미포함: ONNX 모델을 직접 선택하세요."
                    }
                    return@Thread
                }
                val target = java.io.File(filesDir, "bundled_lama_fp32.onnx")
                if (!target.isFile || target.length() == 0L) {
                    assets.open("lama_fp32.onnx").use { input ->
                        target.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
                    }
                }
                ai.onnxruntime.OrtSession.SessionOptions().use { options ->
                    ai.onnxruntime.OrtEnvironment.getEnvironment().createSession(target.absolutePath, options).use { session ->
                        val names = session.inputNames
                        require(names.size == 2 && names.any { it.contains("mask", true) }) {
                            "내장 ONNX 모델 입력 구조가 호환되지 않습니다: $names"
                        }
                    }
                }
                modelFile = target
                runOnUiThread { status.text = "오프라인 LaMa 객체 제거 모델 준비 완료 · 사진을 선택하세요" }
            } catch (e: Exception) {
                modelFile = null
                runOnUiThread { status.text = "내장 모델 준비 실패: ${e.message} · ONNX 파일 직접 선택 가능" }
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
                ai.onnxruntime.OrtSession.SessionOptions().use { options ->
                    ai.onnxruntime.OrtEnvironment.getEnvironment().createSession(target.absolutePath, options).use { session ->
                        val names = session.inputNames
                        require(names.size == 2 && names.any { it.contains("mask", true) }) {
                            "LaMa 호환 모델이 아닙니다. 입력: $names"
                        }
                    }
                }
                modelFile = target
                runOnUiThread { status.text = "LaMa 호환 ONNX 모델 확인됨: ${target.length()/1048576} MB" }
            } catch (e: Exception) {
                modelFile = null
                runOnUiThread { status.text = "모델 검증 실패: ${e.message}" }
            }
        }.start()
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
                    undoSnapshot = photo.copy(Bitmap.Config.ARGB_8888, false)
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

    private fun putPending(values: android.content.ContentValues) {
        values.put(MediaStore.Images.Media.IS_PENDING, 1)
    }

    private fun saveImage() {
        val image = current ?: return
        try {
            val values = android.content.ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "PhotoForge_${System.currentTimeMillis()}.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PhotoForge")
            }
            putPending(values)
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("저장 공간 접근 실패")
            try {
                contentResolver.openOutputStream(uri).use { out ->
                    if (out == null || !image.compress(Bitmap.CompressFormat.PNG, 100, out)) error("PNG 저장 실패")
                }
                val done = android.content.ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
                contentResolver.update(uri, done, null, null)
            } catch (e: Exception) {
                contentResolver.delete(uri, null, null)
                throw e
            }
            Toast.makeText(this, "Pictures/PhotoForge에 저장했습니다", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "저장 실패: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
}
