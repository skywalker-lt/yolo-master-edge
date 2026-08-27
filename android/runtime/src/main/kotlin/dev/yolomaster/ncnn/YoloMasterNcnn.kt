package dev.yolomaster.ncnn

import android.graphics.Bitmap
import kotlin.math.abs

/**
 * On-device YOLO-Master inference on ncnn.
 *
 * Wraps the shared C++ core (the same letterbox -> ncnn -> decode -> NMS path as the
 * desktop runners). Detection and segmentation, CPU-fp32 by default with an opt-in
 * Vulkan fast path. Not thread-safe: use one instance per thread, or serialize calls.
 *
 * The CPU path is pinned to fp32 in native code; do not attempt to force fp16 on CPU,
 * or the mixture-of-experts routing underflows and returns zero detections on ARM.
 *
 * Usage:
 * ```
 * YoloMasterNcnn().use { rt ->
 *     rt.init(modelDir, useVulkan = false)
 *     rt.setConfig(conf = 0.25f, iou = 0.45f)
 *     val dets = rt.infer(bitmap)
 * }
 * ```
 */
class YoloMasterNcnn : AutoCloseable {

    private var handle: Long = 0L
    private var names: Array<String> = emptyArray()

    val isLoaded: Boolean get() = handle != 0L

    /** "ncnn-CPU-fp32", "ncnn-Vulkan", or "none". */
    val activeBackend: String get() = if (handle != 0L) nativeActiveBackend(handle) else "none"

    /** Message from the last failed native call ("" if none). */
    val lastError: String get() = nativeLastError()

    /**
     * Load an ncnn model directory containing `model.ncnn.param`, `model.ncnn.bin`, and
     * `metadata.yaml`. If [useVulkan] is true but no usable GPU exists, it transparently
     * falls back to CPU-fp32 (check [activeBackend]). Returns true on success.
     */
    fun init(modelDir: String, useVulkan: Boolean = false, threads: Int = 0): Boolean {
        close()
        handle = nativeInit(modelDir, useVulkan, threads)
        if (handle != 0L) names = nativeMetaNames(handle)
        return handle != 0L
    }

    /**
     * Robust GPU policy for one model: load on Vulkan, verify it agrees with the CPU-fp32
     * reference on [probe] (guards the per-device fp16/driver risk), and fall back to
     * CPU-fp32 if the GPU produces no detections or diverges. Ends loaded either way.
     */
    fun initBest(modelDir: String, probe: Bitmap, threads: Int = 0): Boolean {
        if (!init(modelDir, useVulkan = false, threads = threads)) return false
        val cpuCount = infer(probe).size
        if (!init(modelDir, useVulkan = true, threads = threads)) {
            return init(modelDir, useVulkan = false, threads = threads)
        }
        if (activeBackend != "ncnn-Vulkan") return true // no GPU: already CPU-fp32
        val gpu = infer(probe)
        val agrees = gpu.isNotEmpty() && abs(gpu.size - cpuCount) <= 1
        if (!agrees) init(modelDir, useVulkan = false, threads = threads)
        return isLoaded
    }

    /** Tune confidence / IoU / max detections. Cheap: reuses the cached forward pass. */
    fun setConfig(conf: Float = 0.25f, iou: Float = 0.45f, maxDet: Int = 300) {
        check(handle != 0L) { "runtime not loaded" }
        nativeSetConfig(handle, conf, iou, maxDet)
    }

    /** Run detection. Throws if the runtime is not loaded or inference fails. */
    fun infer(bitmap: Bitmap): List<Detection> {
        check(handle != 0L) { "runtime not loaded" }
        val src = bitmap.ensureArgb8888()
        val flat = nativeInfer(handle, src) ?: throw RuntimeException("infer failed: $lastError")
        return decode(flat)
    }

    /**
     * Run detection and, for segmentation models, also composite the mask overlay from the
     * same forward pass. For detection models the overlay is empty.
     */
    fun inferSeg(bitmap: Bitmap): SegResult {
        val dets = infer(bitmap) // runs the forward and caches candidates+proto natively
        val dims = IntArray(2)
        val rgba = nativeSegOverlay(handle, dims) ?: return SegResult(dets, ByteArray(0), 0, 0)
        return SegResult(dets, rgba, dims[0], dims[1])
    }

    private fun decode(flat: FloatArray): List<Detection> {
        if (flat.isEmpty()) return emptyList()
        val n = flat[0].toInt()
        val out = ArrayList<Detection>(n)
        for (i in 0 until n) {
            val p = 1 + i * 6
            val cls = flat[p + 5].toInt()
            out += Detection(
                x1 = flat[p], y1 = flat[p + 1], x2 = flat[p + 2], y2 = flat[p + 3],
                score = flat[p + 4], classId = cls,
                label = names.getOrElse(cls) { cls.toString() },
            )
        }
        return out
    }

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
            names = emptyArray()
        }
    }

    private fun Bitmap.ensureArgb8888(): Bitmap =
        if (config == Bitmap.Config.ARGB_8888) this else copy(Bitmap.Config.ARGB_8888, false)

    private external fun nativeInit(modelDir: String, useVulkan: Boolean, threads: Int): Long
    private external fun nativeSetConfig(handle: Long, conf: Float, iou: Float, maxDet: Int)
    private external fun nativeInfer(handle: Long, bitmap: Bitmap): FloatArray?
    private external fun nativeSegOverlay(handle: Long, dimsOut: IntArray): ByteArray?
    private external fun nativeActiveBackend(handle: Long): String
    private external fun nativeMetaNames(handle: Long): Array<String>
    private external fun nativeLastError(): String
    private external fun nativeRelease(handle: Long)

    companion object {
        init { System.loadLibrary("yolomaster_ncnn") }
    }
}
