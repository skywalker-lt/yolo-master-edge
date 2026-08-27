package dev.yolomaster.ncnn

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
