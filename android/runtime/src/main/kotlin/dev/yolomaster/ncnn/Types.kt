package dev.yolomaster.ncnn

/**
 * Numeric precision policy for the ncnn CPU path. Mirrors `Precision` in the C++ core; the
 * integer codes are the JNI ABI and must stay stable.
 *
 *  - [AUTO]: fp16 when the model is fp16-safe (the dense models) on an armv8.2 CPU; fp32 pinned
 *    for the emulated-router mixture models, whose export constants (1e-9 / 1e30) are
 *    unrepresentable in fp16 and would zero the routing. The decision is read from the model
 *    itself (its `.param` fingerprint), not from the caller.
 *  - [FP32] / [FP16]: explicit. A request the model cannot honour is downgraded and explained in
 *    [YoloMasterNcnn.backendNote]; it is never run as a silent zero-detection model.
 *  - [INT8]: load the pre-quantized sibling directory `<name>-int8_ncnn` (mixed per-layer int8
 *    from scripts/quantize_ncnn_int8.py; the sensitive layers stay float). CPU only. Loading fails
 *    if the sibling is absent - an int8 number must never silently come from a float model.
 */
enum class Precision(val native: Int) { AUTO(0), FP32(1), FP16(2), INT8(3) }

/** One detection in original-image pixel coordinates. */
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val classId: Int,
    val label: String,
)

/**
 * Segmentation result: the detections plus a composited RGBA overlay the size of the
 * original image ([maskWidth] x [maskHeight], 4 bytes/pixel, RGBA order). [maskRgba] is
 * empty for non-segmentation models.
 */
data class SegResult(
    val detections: List<Detection>,
    val maskRgba: ByteArray,
    val maskWidth: Int,
    val maskHeight: Int,
) {
    override fun equals(other: Any?): Boolean =
        this === other || (other is SegResult &&
            detections == other.detections && maskWidth == other.maskWidth &&
            maskHeight == other.maskHeight && maskRgba.contentEquals(other.maskRgba))

    override fun hashCode(): Int =
        (detections.hashCode() * 31 + maskWidth) * 31 + maskHeight
}
