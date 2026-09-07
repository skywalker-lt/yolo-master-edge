package dev.yolomaster.app.model

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.util.zip.ZipInputStream

/** Which compute unit a model may run on; the app shows this in the compute picker. */
enum class ComputeChoice(val label: String) {
    GPU("GPU"), CPU("CPU");

    companion object {
        /** iOS hides CPU behind a Settings toggle; INT8 models are CPU-only regardless. */
        fun available(allowCPU: Boolean, model: BundledModel?): List<ComputeChoice> = when {
            model?.cpuOnly == true -> listOf(CPU)
            allowCPU -> listOf(GPU, CPU)
            else -> listOf(GPU)
        }
    }
}

/**
 * One ncnn model directory (`model.ncnn.param` + `model.ncnn.bin` + `metadata.yaml`).
 * `id` is the directory name, which is also the runtime's identity for INT8 siblings.
 */
data class BundledModel(
    val id: String,
    val dir: File,
    val isSeg: Boolean,
    val isCustom: Boolean,
) {
    val isInt8: Boolean get() = id.endsWith("-int8_ncnn")
    /** ncnn has no Vulkan int8 kernels: INT8 entries force CPU (the runtime would too). */
    val cpuOnly: Boolean get() = isInt8
    val fullName: String get() = Naming.fullName(id)
    val shortID: String get() = Naming.shortID(id)

    companion object {
        /** iOS default: the first model whose stem mentions "seg", else the first. */
        fun preferred(models: List<BundledModel>): BundledModel? =
            models.firstOrNull { it.id.contains("seg", ignoreCase = true) } ?: models.firstOrNull()
    }
}

/**
 * Discovery of bundled (APK assets, copied once to `filesDir/models`) and custom
 * (`filesDir/CustomModels`) models. Mirrors `BundledModel.discover()` in `Models.swift`: the
 * list is enumeration only; nothing is loaded until a tab selects a model.
 */
class ModelCatalog(private val ctx: Context) {
    private val bundledRoot = File(ctx.filesDir, "models")
    private val customRoot = File(ctx.filesDir, "CustomModels")
    private val stamp = File(bundledRoot, ".assets-version")

    /** Bump when the bundled asset set changes so the copy is refreshed. */
    private val assetsVersion = "1"

    @Volatile private var cached: List<BundledModel>? = null

    /** Blocking: first call copies ~70 MB of assets. Call off the main thread. */
    @Synchronized
    fun discover(force: Boolean = false): List<BundledModel> {
        cached?.takeIf { !force }?.let { return it }
        ensureAssetsCopied()
        val out = ArrayList<BundledModel>()
        listDirs(bundledRoot).forEach { out += describe(it, custom = false) }
        listDirs(customRoot).forEach { out += describe(it, custom = true) }
        val list = out.distinctBy { it.id }.sortedBy { it.id.lowercase() }
        cached = list
        return list
    }

    fun invalidate() { cached = null }

    private fun listDirs(root: File): List<File> =
        root.listFiles()?.filter { File(it, "model.ncnn.param").isFile && File(it, "model.ncnn.bin").isFile }
            ?.sortedBy { it.name } ?: emptyList()

    private fun describe(dir: File, custom: Boolean): BundledModel =
        BundledModel(id = dir.name, dir = dir, isSeg = readTask(dir) == "segment", isCustom = custom)

    private fun readTask(dir: File): String? {
        val yaml = File(dir, "metadata.yaml")
        if (!yaml.isFile) return null
        yaml.useLines { lines ->
            for (l in lines) {
                val t = l.trim()
                if (t.startsWith("task:")) return t.removePrefix("task:").trim().trim('"', '\'')
            }
        }
        return null
    }

    private fun ensureAssetsCopied() {
        if (stamp.isFile && stamp.readText() == assetsVersion) return
        val am = ctx.assets
        val names = am.list("models") ?: emptyArray()
        bundledRoot.mkdirs()
        for (name in names) {
            val files = am.list("models/$name") ?: continue
            if ("model.ncnn.param" !in files) continue
            val dst = File(bundledRoot, name)
            dst.mkdirs()
            for (f in files) {
                am.open("models/$name/$f").use { i -> File(dst, f).outputStream().use { o -> i.copyTo(o) } }
            }
        }
        stamp.writeText(assetsVersion)
    }

    // ---- custom models (Settings > Custom models) ------------------------------------------

    /** Import a picked directory tree (SAF). Returns the new model or throws with a reason. */
    fun importTree(uri: Uri): BundledModel {
        val doc = DocumentFile.fromTreeUri(ctx, uri) ?: throw IllegalArgumentException("Cannot open folder")
        val name = sanitize(doc.name ?: "custom")
        val dst = File(customRoot, if (name.endsWith("_ncnn")) name else "${name}_ncnn")
        dst.mkdirs()
        var got = 0
        for (f in doc.listFiles()) {
            val n = f.name ?: continue
            if (n in REQUIRED || n == "metadata.yaml") {
                ctx.contentResolver.openInputStream(f.uri)?.use { i -> File(dst, n).outputStream().use { i.copyTo(it) } }
                if (n in REQUIRED) got++
            }
        }
        if (got < REQUIRED.size) { dst.deleteRecursively(); throw IllegalArgumentException("Folder must contain model.ncnn.param and model.ncnn.bin") }
        invalidate()
        return describe(dst, custom = true)
    }

    /** Import a .zip whose entries (any top folder) contain the model files. */
    fun importZip(uri: Uri, displayName: String?): BundledModel {
        val name = sanitize((displayName ?: "custom").removeSuffix(".zip"))
        val dst = File(customRoot, if (name.endsWith("_ncnn")) name else "${name}_ncnn")
        dst.mkdirs()
        var got = 0
        ctx.contentResolver.openInputStream(uri)?.use { raw ->
            ZipInputStream(raw).use { zip ->
                var e = zip.nextEntry
                while (e != null) {
                    val base = e.name.substringAfterLast('/')
                    if (!e.isDirectory && (base in REQUIRED || base == "metadata.yaml")) {
                        File(dst, base).outputStream().use { zip.copyTo(it) }
                        if (base in REQUIRED) got++
                    }
                    zip.closeEntry(); e = zip.nextEntry
                }
            }
        } ?: throw IllegalArgumentException("Cannot open zip")
        if (got < REQUIRED.size) { dst.deleteRecursively(); throw IllegalArgumentException("Zip must contain model.ncnn.param and model.ncnn.bin") }
        invalidate()
        return describe(dst, custom = true)
    }

    fun deleteCustom(model: BundledModel) {
        if (model.isCustom) { model.dir.deleteRecursively(); invalidate() }
    }

    private fun sanitize(s: String) = s.replace(Regex("[^A-Za-z0-9._-]"), "_").ifEmpty { "custom" }

    private companion object { val REQUIRED = setOf("model.ncnn.param", "model.ncnn.bin") }
}
