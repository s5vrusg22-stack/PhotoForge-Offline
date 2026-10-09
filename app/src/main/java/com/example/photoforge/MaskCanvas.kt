package com.example.photoforge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View

/** Paints an editable mask over an image displayed with FIT_CENTER geometry. */
class MaskCanvas(context: Context) : View(context) {
    var photo: Bitmap? = null
        set(value) { field = value; clearMask(); invalidate() }
    var brushPx: Float = 36f
    private val paths = mutableListOf<Pair<Path, Float>>()
    private var active: Path? = null
    private val imagePaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(160, 255, 82, 120)
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private fun area(): RectF {
        val b = photo ?: return RectF()
        val scale = minOf(width.toFloat() / b.width, height.toFloat() / b.height)
        val w = b.width * scale
        val h = b.height * scale
        return RectF((width-w)/2, (height-h)/2, (width+w)/2, (height+h)/2)
    }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(35,39,48))
        val b = photo ?: return
        val rect = area()
        canvas.drawBitmap(b, null, rect, imagePaint)
        canvas.save()
        canvas.clipRect(rect)
        for ((path, brush) in paths) {
            strokePaint.strokeWidth = brush
            canvas.drawPath(path, strokePaint)
        }
        active?.let { strokePaint.strokeWidth = brushPx; canvas.drawPath(it, strokePaint) }
        canvas.restore()
    }
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (photo == null) return false
        val rect = area()
        val x = event.x.coerceIn(rect.left, rect.right)
        val y = event.y.coerceIn(rect.top, rect.bottom)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                active = Path().apply { moveTo(x, y); lineTo(x + 0.1f, y + 0.1f) }
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                active?.lineTo(x, y)
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP -> {
                active?.lineTo(x, y)
                active?.let { paths.add(it to brushPx) }
                active = null
                invalidate()
                return true
            }
        }
        return true
    }
    fun clearMask() { paths.clear(); active = null; invalidate() }
    fun undo() { if (paths.isNotEmpty()) paths.removeAt(paths.lastIndex); invalidate() }
    fun hasMask() = paths.isNotEmpty()
    fun exportMask(): Bitmap {
        val b = photo ?: error("사진을 먼저 여세요.")
        val output = Bitmap.createBitmap(b.width, b.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(Color.BLACK)
        val rect = area()
        require(rect.width() > 0f) { "마스크 영역이 준비되지 않았습니다." }
        canvas.scale(b.width / rect.width(), b.height / rect.height())
        canvas.translate(-rect.left, -rect.top)
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }
        for ((path, brush) in paths) {
            p.strokeWidth = brush
            canvas.drawPath(path, p)
        }
        return output
    }
}
