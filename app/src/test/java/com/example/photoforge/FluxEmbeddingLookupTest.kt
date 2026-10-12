package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class FluxEmbeddingLookupTest {
    @Test fun readsFp16RowsWithoutLoadingWholeTable() {
        val file = File.createTempFile("qwen-embedding-test", ".bin")
        try {
            java.io.RandomAccessFile(file, "rw").use { data ->
                data.setLength(FluxEmbeddingLookup.VOCAB_SIZE.toLong() * FluxEmbeddingLookup.ROW_BYTES)
                data.seek(FluxEmbeddingLookup.ROW_BYTES.toLong() * 7)
                repeat(FluxEmbeddingLookup.HIDDEN_SIZE) {
                    data.writeByte(0x00)
                    data.writeByte(0x3c) // 1.0 in little-endian fp16
                }
            }
            FluxEmbeddingLookup(file).use { lookup ->
                val ids = IntArray(512) { 7 }
                val values = lookup.lookup(ids)
                assertEquals(512 * 2560, values.size)
                assertEquals(1f, values[0], 0f)
                assertEquals(1f, values.last(), 0f)
            }
        } finally {
            file.delete()
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsWrongEmbeddingFileSize() {
        val file = File.createTempFile("invalid-embedding", ".bin")
        try { FluxEmbeddingLookup(file).close() } finally { file.delete() }
    }
}
