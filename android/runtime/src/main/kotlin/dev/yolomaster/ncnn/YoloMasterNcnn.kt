package dev.yolomaster.ncnn

import android.graphics.Bitmap
import kotlin.math.abs

/**
 * On-device YOLO-Master inference on ncnn.
 *
 * Wraps the shared C++ core (the same letterbox -> ncnn -> decode -> NMS path as the
 * desktop runners). Detection and segmentation, CPU by default with an opt-in Vulkan
 * fast path. Not thread-safe: use one instance per thread, or serialize calls.
 *
 * CPU precision is decided PER MODEL by native code (see [Precision]): fp16-safe (dense)
 * models run fp16 on armv8.2 CPUs; the emulated-router mixture models are pinned fp32
 * because their export constants (1e-9 / 1e30) are unrepresentable in fp16 and would zero
 * the routing. Pass [Precision.INT8] to load the pre-quantized `<name>-int8_ncnn` sibling.
 * [activeBackend] always reports what actually resolved; [backendNote] says why a request
 * was downgraded.
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

    /**
     * The precision that actually resolved, reported by native code (never the requested flag):
     * "ncnn-CPU-fp32", "ncnn-CPU-fp16", "ncnn-CPU-int8+fp32", "ncnn-CPU-int8+fp16",
     * "ncnn-Vulkan" (GPU fp16), "ncnn-Vulkan-fp32" (GPU, fp16 declined by the model), or "none".
     */
    val activeBackend: String get() = if (handle != 0L) nativeActiveBackend(handle) else "none"

    /** Why a requested precision was downgraded (e.g. "fp32 pinned: emulated-router ..."); "" if it was honoured. */
    val backendNote: String get() = if (handle != 0L) nativeBackendNote(handle) else ""

    /** Message from the last failed native call ("" if none). */
    val lastError: String get() = nativeLastError()

    /**
     * Load an ncnn model directory containing `model.ncnn.param`, `model.ncnn.bin`, and
     * `metadata.yaml`. If [useVulkan] is true but no usable GPU exists, it transparently
     * falls back to the CPU choice (check [activeBackend]). [precision] selects the CPU
     * numeric policy (see [Precision]); [Precision.INT8] loads the `<name>-int8_ncnn` sibling
     * and returns false (with [lastError]) if it does not exist. Returns true on success.
     */
    fun init(
        modelDir: String,
        useVulkan: Boolean = false,
        threads: Int = 0,
        precision: Precision = Precision.AUTO,
    ): Boolean {
        close()
        handle = nativeInit(modelDir, useVulkan, threads, precision.native)
        if (handle != 0L) names = nativeMetaNames(handle)
        return handle != 0L
    }

    /**
     * Robust policy for one model. Loads the fp32 CPU reference first and uses its detection
     * count on [probe] as the yardstick; then loads the requested [precision] and keeps it only if
     * it still detects and agrees with the reference (+/-1 for fp16; for int8 a garbage band of
     * non-zero and within 35% of the count - a single image's int8 count legitimately jitters by
     * 20% (seg on the COCO probe: 15 vs 12) while the model's accuracy is certified on the full val
     * set, so this check only catches a dead int8 graph); then tries Vulkan the same way (skipped
     * for INT8, which is CPU-only). fp16 underflow (empty result) and int8 garbage are caught here
     * on unknown SoCs.
     * Ends loaded either way; INT8 with no sibling returns false.
     */
    fun initBest(
        modelDir: String,
        probe: Bitmap,
        threads: Int = 0,
        precision: Precision = Precision.AUTO,
    ): Boolean {
        if (!init(modelDir, useVulkan = false, threads = threads, precision = Precision.FP32)) return false
        val ref = infer(probe).size
        if (precision != Precision.FP32) {
            if (!init(modelDir, useVulkan = false, threads = threads, precision = precision)) {
                if (precision == Precision.INT8) return false   // no int8 sibling: hard failure
                init(modelDir, useVulkan = false, threads = threads, precision = Precision.FP32)
            } else if (activeBackend.endsWith("fp16") || activeBackend.contains("int8")) {
                val n = infer(probe).size
                val tol = if (activeBackend.contains("int8")) maxOf(2, (ref * 35) / 100) else 1
                if (n == 0 || abs(n - ref) > tol) {
                    init(modelDir, useVulkan = false, threads = threads, precision = Precision.FP32)
                }
            }
        }
        if (precision == Precision.INT8) return isLoaded
        if (!init(modelDir, useVulkan = true, threads = threads, precision = precision)) {
            return init(modelDir, useVulkan = false, threads = threads, precision = precision)
        }
        if (!activeBackend.startsWith("ncnn-Vulkan")) return true // no GPU: already on the CPU choice
        val gpu = infer(probe)
        val agrees = gpu.isNotEmpty() && abs(gpu.size - ref) <= 1
        if (!agrees) init(modelDir, useVulkan = false, threads = threads, precision = precision)
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

    private external fun nativeInit(modelDir: String, useVulkan: Boolean, threads: Int, precision: Int): Long
    private external fun nativeSetConfig(handle: Long, conf: Float, iou: Float, maxDet: Int)
    private external fun nativeInfer(handle: Long, bitmap: Bitmap): FloatArray?
    private external fun nativeSegOverlay(handle: Long, dimsOut: IntArray): ByteArray?
    private external fun nativeActiveBackend(handle: Long): String
    private external fun nativeBackendNote(handle: Long): String
    private external fun nativeMetaNames(handle: Long): Array<String>
    private external fun nativeLastError(): String
    private external fun nativeRelease(handle: Long)

    companion object {
        init { System.loadLibrary("yolomaster_ncnn") }
    }
}
