package dev.yolomaster.ncnn

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * On-device runtime harness. Proves the ncnn backend before any app UI exists.
 *
 * Prerequisites (run `android/scripts/stage_models.sh` first, then rebuild):
 *   - assets/models/v0.1-seg-n_ncnn/{model.ncnn.param,model.ncnn.bin,metadata.yaml}
 *   - assets/models/moa-n_ncnn/{...}
 *   - assets/probe.jpg   (any scene with detectable objects)
 *
 * Run on a connected arm64 device:  ./gradlew :runtime:connectedAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class ParityTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val tag = "ParityTest"

    private fun stage(model: String): String {
        val out = File(ctx.filesDir, "models/$model")
        if (File(out, "model.ncnn.param").exists()) return out.absolutePath
        out.mkdirs()
        for (f in listOf("model.ncnn.param", "model.ncnn.bin", "metadata.yaml")) {
            ctx.assets.open("models/$model/$f").use { input ->
                File(out, f).outputStream().use { input.copyTo(it) }
            }
        }
        return out.absolutePath
    }

    private fun probe(): Bitmap =
        ctx.assets.open("probe.jpg").use { BitmapFactory.decodeStream(it) }
            .copy(Bitmap.Config.ARGB_8888, false)

    /** Default seg-N model loads, reports CPU-fp32, and produces detections. */
    @Test
    fun default_model_detects_on_cpu_fp32() {
        val img = probe()
        YoloMasterNcnn().use { rt ->
            assertTrue("init CPU: ${rt.lastError}", rt.init(stage("v0.1-seg-n_ncnn"), useVulkan = false))
            assertEquals("ncnn-CPU-fp32", rt.activeBackend)
            rt.setConfig(conf = 0.25f, iou = 0.45f)
            val t0 = System.nanoTime()
            val dets = rt.infer(img)
            val ms = (System.nanoTime() - t0) / 1e6
            Log.i(tag, "v0.1-seg-N cpu-fp32: ${dets.size} dets in %.1f ms".format(ms))
            // fp16-underflow regression guard: the bug this runtime inherits the fix for
            // returns zero detections on ARM.
            assertTrue("CPU-fp32 must produce detections (fp16 regression?)", dets.isNotEmpty())
        }
    }

    /** An emulated-router MoE model runs on ARM (proves both hazards are handled on-device). */
    @Test
    fun mixture_router_runs_on_arm() {
        val img = probe()
        YoloMasterNcnn().use { rt ->
            assertTrue("init moa-n: ${rt.lastError}", rt.init(stage("moa-n_ncnn"), useVulkan = false))
            val dets = rt.infer(img)
            Log.i(tag, "moa-n cpu-fp32: ${dets.size} dets")
            assertTrue("emulated MoE router must detect on ARM", dets.isNotEmpty())
        }
    }

    /** Vulkan either agrees with CPU (within +/-1 detection) or cleanly falls back to CPU-fp32. */
    @Test
    fun vulkan_matches_cpu_or_falls_back() {
        val img = probe()
        val dir = stage("v0.1-seg-n_ncnn")
        val cpuCount = YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = false)); rt.infer(img).size
        }
        YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = true))
            if (rt.activeBackend == "ncnn-Vulkan") {
                val gpu = rt.infer(img)
                Log.i(tag, "vulkan: ${gpu.size} vs cpu: $cpuCount")
                assertTrue("Vulkan diverged from CPU", gpu.isNotEmpty() && abs(gpu.size - cpuCount) <= 1)
            } else {
                assertEquals("no GPU must fall back to CPU-fp32", "ncnn-CPU-fp32", rt.activeBackend)
            }
        }
    }

    /** A missing model directory fails cleanly with an error, never a native crash. */
    @Test
    fun missing_model_fails_cleanly() {
        YoloMasterNcnn().use { rt ->
            assertFalse(rt.init(File(ctx.filesDir, "does-not-exist").absolutePath, useVulkan = false))
            assertTrue("expected a load error message", rt.lastError.isNotEmpty())
        }
    }
}
