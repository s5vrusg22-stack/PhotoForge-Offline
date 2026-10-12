package com.example.photoforge

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class FluxRuntimePreflightTest {
    @Test fun emptyDirectoryIsNotReady() {
        val dir = createTempDir(prefix = "flux-preflight")
        try {
            val result = FluxRuntimePreflight.check(dir, editing = true)
            assertFalse(result.ready)
            assertTrue(result.problems.isNotEmpty())
        } finally { dir.deleteRecursively() }
    }
}
