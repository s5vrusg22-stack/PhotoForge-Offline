package com.example.photoforge

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.Debug
import android.os.SystemClock

/**
 * Lightweight diagnostics. PSS includes some graphics allocations but is NOT GPU utilization
 * or total device GPU memory. Samples are snapshots, not continuous hardware counters.
 */
object DevicePerformanceMonitor {
    data class Snapshot(
        val elapsedMs: Long,
        val appPssKb: Int,
        val nativeHeapKb: Long,
        val availableRamBytes: Long,
        val lowMemory: Boolean,
        val thermalStatus: Int?,
        val batteryTemperatureC: Float?
    ) {
        fun summary(): String =
            "PSS ${appPssKb / 1024} MiB · native ${nativeHeapKb / 1024} MiB · " +
            "RAM free ${availableRamBytes / 1048576} MiB · lowMemory=$lowMemory · " +
            "thermal=${thermalStatus ?: "unknown"} · battery=${batteryTemperatureC ?: "unknown"} °C"
    }

    fun sample(context: Context): Snapshot {
        val activity = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val ram = ActivityManager.MemoryInfo().also(activity::getMemoryInfo)
        val pss = Debug.getPss().toInt()
        val battery = context.registerReceiver(
            null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED)
        )
        val tempTenths = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        val thermal = if (android.os.Build.VERSION.SDK_INT >= 29) {
            (context.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager).currentThermalStatus
        } else null
        return Snapshot(
            SystemClock.elapsedRealtime(),
            pss,
            Debug.getNativeHeapAllocatedSize() / 1024,
            ram.availMem,
            ram.lowMemory,
            thermal,
            if (tempTenths >= 0) tempTenths / 10f else null
        )
    }
}
