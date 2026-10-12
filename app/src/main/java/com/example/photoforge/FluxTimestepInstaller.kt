package com.example.photoforge

import org.json.JSONObject
import java.io.File

/** Installs a verified exporter artifact into the runtime HostTensors. */
object FluxTimestepInstaller {
    fun attachFromModelRoot(
        host: FluxGraphExecutor.HostTensors,
        modelRoot: File
    ): FluxGraphExecutor.HostTensors {
        val binary = File(modelRoot, FluxProjectedTimestepConfig.FILE_NAME)
        val metadata = File(modelRoot, FluxProjectedTimestepConfig.META_NAME)
        require(metadata.isFile) { "Missing FLUX timestep metadata" }
        val json = JSONObject(metadata.readText(Charsets.UTF_8))
        val sha = json.getString("sha256")
        val shape = json.getJSONArray("shape")
        require(shape.length() == 2 && shape.getInt(0) == 4 && shape.getInt(1) == 3072) {
            "Unexpected FLUX timestep projection shape"
        }
        require(json.getString("dtype") == "float32") { "Unexpected timestep dtype" }
        require(json.getString("byte_order") == "little") { "Unexpected timestep byte order" }
        return FluxTimestepAssets.attach(host, binary, sha)
    }
}
