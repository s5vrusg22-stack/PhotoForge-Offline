package com.example.photoforge

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF

/**
 * Explicitly non-generative, offline keyword-driven accessory renderer.
 * Produces a transparent layer that can be composited without altering other pixels.
 * This is a useful fallback while a licensed text-conditioned diffusion model is integrated.
 */
object PromptPropRenderer {
    fun render(prompt: String, width: Int, height: Int): Bitmap? {
        val p = prompt.lowercase().trim()
        val kind = when {
            listOf("안경", "glasses", "spectacles").any { p.contains(it) } -> "glasses"
            listOf("모자", "hat", "cap").any { p.contains(it) } -> "hat"
            listOf("목걸이", "necklace").any { p.contains(it) } -> "necklace"
            else -> return null
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val c = Canvas(bitmap)
        val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(38, 35, 40)
            style = Paint.Style.STROKE
            strokeWidth = width * 0.009f
            strokeCap = Paint.Cap.ROUND
        }
        val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(45, 48, 58)
            style = Paint.Style.FILL
        }
        val x = width / 2f
        val y = height / 2f
        when (kind) {
            "glasses" -> {
                val w = width * 0.13f
                val h = height * 0.065f
                c.drawRoundRect(RectF(x - w * 2.2f, y - h, x - w * 0.2f, y + h), h * 0.5f, h * 0.5f, stroke)
                c.drawRoundRect(RectF(x + w * 0.2f, y - h, x + w * 2.2f, y + h), h * 0.5f, h * 0.5f, stroke)
                c.drawLine(x - w * 0.2f, y - h * 0.25f, x + w * 0.2f, y - h * 0.25f, stroke)
            }
            "hat" -> {
                val path = Path().apply {
                    moveTo(x - width * 0.24f, y)
                    lineTo(x - width * 0.17f, y - height * 0.17f)
                    lineTo(x + width * 0.17f, y - height * 0.17f)
                    lineTo(x + width * 0.24f, y)
                    close()
                }
                c.drawPath(path, fill)
                c.drawRoundRect(RectF(x-width*0.34f,y-height*0.025f,x+width*0.34f,y+height*0.028f),height*0.02f,height*0.02f,fill)
            }
            "necklace" -> {
                val necklace = Path().apply {
                    moveTo(x-width*0.16f, y-height*0.11f)
                    quadTo(x, y+height*0.22f, x+width*0.16f, y-height*0.11f)
                }
                stroke.color = Color.rgb(210, 170, 74)
                stroke.strokeWidth = width*0.006f
                c.drawPath(necklace, stroke)
                c.drawCircle(x,y+height*0.04f,width*0.018f,Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.rgb(228,192,94) })
            }
        }
        return bitmap
    }
}
