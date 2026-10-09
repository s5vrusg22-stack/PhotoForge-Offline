package com.example.photoforge

import org.junit.Assert.*
import org.junit.Test

class FlowMatchSchedulerTest {
    @Test fun linearScheduleHasCorrectEndpoints() {
        val schedule = FlowMatchScheduler.linearSchedule(4)
        assertArrayEquals(floatArrayOf(1f, 0.75f, 0.5f, 0.25f, 0f), schedule, 0f)
        FlowMatchScheduler.validateTimesteps(schedule)
    }

    @Test fun eulerStepUpdatesLatents() {
        val latents = floatArrayOf(1f, -1f)
        FlowMatchScheduler.eulerStep(latents, floatArrayOf(2f, -4f), 1f, 0.5f)
        assertArrayEquals(floatArrayOf(0f, 1f), latents, 0.00001f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsMismatchedTensorLengths() {
        FlowMatchScheduler.eulerStep(floatArrayOf(1f), floatArrayOf(), 1f, 0.5f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsAscendingSchedule() {
        FlowMatchScheduler.validateTimesteps(floatArrayOf(0.5f, 0.75f))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNaN() {
        FlowMatchScheduler.eulerStep(floatArrayOf(Float.NaN), floatArrayOf(1f), 1f, 0f)
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsFiniteInputsThatOverflowEulerUpdate() {
        FlowMatchScheduler.eulerStep(
            floatArrayOf(Float.MAX_VALUE),
            floatArrayOf(-Float.MAX_VALUE),
            1f, 0f
        )
    }
}
