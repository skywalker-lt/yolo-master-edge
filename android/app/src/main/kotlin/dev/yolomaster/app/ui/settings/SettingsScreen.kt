package dev.yolomaster.app.ui.settings

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddCircleOutline
import androidx.compose.material.icons.filled.Article
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Inventory2
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.yolomaster.app.R
import dev.yolomaster.app.YoloMasterApp
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.system.Prefs
import dev.yolomaster.app.system.rememberBoolPref
import dev.yolomaster.app.ui.theme.IosOrange
import dev.yolomaster.app.ui.theme.IosRed
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/*
 * The Settings tab (`ContentView.swift:104-425`): About, Licenses & Acknowledgements, Privacy &
 * Security, Compute, Custom models, Erase. Strings are the iOS ones with the Core ML wording
 * replaced by the ncnn facts.
 */

private data class Ack(val name: String, val logo: Int?, val blurb: String, val repo: String)

private val baseProjects = listOf(
    Ack("YOLO-Master @ Tencent", R.drawable.ack_tencent, "The MoE YOLO detector family this app packages. Copyright 2026 Tencent, AGPL-3.0.", "https://github.com/Tencent/YOLO-Master"),
    Ack("YOLO-Master Edge (this app)", R.drawable.ack_skywalker_lt, "The Android app and the ncnn export toolchain. Copyright 2026 Thomas (Ruiheng Li), HKUST, AGPL-3.0.", "https://github.com/skywalker-lt/yolo-master-edge"),
)
private val acknowledgements = listOf(
    Ack("Ultralytics", R.drawable.ack_ultralytics, "The YOLO training and inference framework YOLO-Master builds on. Copyright 2025 Ultralytics, AGPL-3.0.", "https://github.com/ultralytics/ultralytics"),
    Ack("ncnn @ Tencent", null, "The on-device inference runtime. Copyright 2017-2026 THL A29 Limited, BSD-3-Clause.", "https://github.com/Tencent/ncnn"),
)

private val licenseNotice = """
================================================================
YOLO-Master for Android  -  license notice
================================================================

This application is distributed under the GNU Affero General
Public License, version 3 (AGPL-3.0). Source:
  https://github.com/skywalker-lt/yolo-master-edge

It bundles, links or derives from:

  YOLO-Master            Copyright 2026 Tencent
                         AGPL-3.0
                         https://github.com/Tencent/YOLO-Master

  ultralytics            Copyright 2025 Ultralytics
                         AGPL-3.0
                         https://github.com/ultralytics/ultralytics

  ncnn                   Copyright 2017-2026 THL A29 Limited
                         BSD-3-Clause
                         https://github.com/Tencent/ncnn

  opencv-mobile          Copyright nihui and OpenCV contributors
                         Apache-2.0
                         https://github.com/nihui/opencv-mobile

The ncnn model weights bundled with this app are derived from
YOLO-Master checkpoints and are distributed under AGPL-3.0.

USE NOTICE (not a license term)
  This is a research beta. The authors release it for research
  and personal experience only, and neither intend nor support
  any commercial use. This statement expresses the authors'
  intent; it does not add restrictions to, or otherwise limit,
  the rights granted under the AGPL-3.0.

NO WARRANTY
  This program is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied
  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the GNU Affero General Public License for more
  details.
================================================================
""".trimIndent()

@Composable
fun SettingsScreen() {
    val ios = LocalIosColors.current
    val ctx = LocalContext.current
    val app = remember { YoloMasterApp.from(ctx) }
    val scope = rememberCoroutineScope()
    val runs by app.history.runs.collectAsState()
    var allowCPU by rememberBoolPref(Prefs.ALLOW_CPU, true)
    var customModels by remember { mutableStateOf<List<BundledModel>>(emptyList()) }
    var importError by remember { mutableStateOf<String?>(null) }
    var confirmErase by remember { mutableStateOf(false) }
    val n = runs.size

    fun refreshCustom() {
        scope.launch { customModels = withContext(Dispatchers.IO) { app.catalog.discover(force = true).filter { it.isCustom } } }
    }
    LaunchedEffect(Unit) { refreshCustom() }

    val treePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) scope.launch {
            try { withContext(Dispatchers.IO) { app.catalog.importTree(uri) }; refreshCustom() }
            catch (t: Throwable) { importError = t.message ?: "Import failed" }
        }
    }
    val zipPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch {
            try { withContext(Dispatchers.IO) { app.catalog.importZip(uri, uri.lastPathSegment) }; refreshCustom() }
            catch (t: Throwable) { importError = t.message ?: "Import failed" }
        }
    }

    Column(Modifier.fillMaxSize().background(ios.groupedBackground).verticalScroll(rememberScrollState())) {
        Text("Settings", style = IosType.largeTitle, color = ios.label, modifier = Modifier.padding(start = 20.dp, top = 56.dp, bottom = 8.dp))

        // A. About
        FormSection {
            var open by remember { mutableStateOf(false) }
            DisclosureRow(open = open, onToggle = { open = !open }, header = {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp), modifier = Modifier.padding(vertical = 4.dp)) {
                    Image(
                        painterResource(R.drawable.ack_appmark), contentDescription = null,
                        modifier = Modifier.size(56.dp).clip(RoundedCornerShape(12.dp)).border(1.dp, ios.label.copy(alpha = 0.1f), RoundedCornerShape(12.dp)),
                    )
                    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Text("YOLO-Master for Android", style = IosType.title3Bold, color = ios.label)
                        Text("Version 1.1.0 Beta", style = IosType.caption, color = ios.secondaryLabel)
                        Text("On-device MoE object detection with ncnn.", style = IosType.caption, color = ios.secondaryLabel)
                    }
                }
            }) {
                Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("YOLO-Master extends the real-time YOLO detector with Mixture-of-Experts (MoE) routing. Instead of one dense network, lightweight expert branches specialize on different feature patterns, and a learned router activates only the most relevant experts for each image.", style = IosType.footnote, color = ios.secondaryLabel)
                    Text("That adds capacity where it matters, with clear gains on small and crowded objects, while keeping inference light enough to run in real time. On Android the model runs through ncnn on the GPU (Vulkan) or the CPU, in fp16 or mixed-INT8.", style = IosType.footnote, color = ios.secondaryLabel)
                    HorizontalDivider(color = ios.separator)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally)) {
                        LinkChip("Paper", Icons.Filled.Article, "https://arxiv.org/pdf/2512.23273")
                        LinkChip("Model Repo", Icons.Filled.Inventory2, "https://github.com/Tencent/YOLO-Master")
                        LinkChip("App Repo", Icons.Filled.Code, "https://github.com/skywalker-lt/yolo-master-edge")
                    }
                }
            }
        }

        // B. Licenses & Acknowledgements
        FormSection {
            var open by remember { mutableStateOf(false) }
            DisclosureRow(open = open, onToggle = { open = !open }, header = { LabelRow(Icons.Filled.Description, "Licenses & Acknowledgements") }) {
                Column(Modifier.padding(top = 6.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text("YOLO-Master and Ultralytics are licensed under AGPL-3.0; ncnn under BSD-3-Clause and opencv-mobile under Apache-2.0. This app and the ncnn weights it bundles are distributed under AGPL-3.0, and this app complies with those terms.", style = IosType.footnote, color = ios.secondaryLabel)
                    Text("This is a research beta. The authors release it for research and personal experience only, and neither intend nor support any commercial use.", style = IosType.footnoteBold, color = ios.label)
                    Text("This is a statement of the authors' intent. It does not add restrictions to, or otherwise limit, the rights granted under the AGPL-3.0.", style = IosType.caption2, color = ios.tertiaryLabel)
                    GroupLabel("BUILT ON")
                    baseProjects.forEach { AckRow(it) }
                    GroupLabel("ACKNOWLEDGEMENTS")
                    acknowledgements.forEach { AckRow(it) }
                    SelectionContainer {
                        Text(
                            licenseNotice, style = IosType.mono11.copy(fontFamily = FontFamily.Monospace, fontSize = 11.sp), color = ios.label,
                            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(ios.secondarySystemBackground).padding(12.dp),
                        )
                    }
                }
            }
        }

        // C. Privacy & Security
        FormSection {
            var open by remember { mutableStateOf(false) }
            DisclosureRow(open = open, onToggle = { open = !open }, header = { LabelRow(Icons.Filled.Security, "Privacy & Security") }) {
                Column(Modifier.padding(top = 6.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("YOLO-Master for Android runs entirely on your device.", style = IosType.footnoteSemibold, color = ios.label)
                    AccessRow(Icons.Filled.CameraAlt, "Camera", "Live on-device detection. Frames are processed in memory and never stored or transmitted.")
                    AccessRow(Icons.Filled.PhotoLibrary, "Photo Library", "Reads only the images you pick for the Photo tab, and saves detection frames you explicitly export.")
                    Text("All inference runs locally on the GPU or CPU. No images, results, or usage data ever leave your phone. We do not collect, store, or transmit any personal data. The only time we receive anything from you is if you choose to contact us or report an issue yourself.", style = IosType.footnote, color = ios.secondaryLabel)
                }
            }
        }

        // D. Compute
        FormSection(header = "Compute", footer = "CPU runs the fp16 path and is required for the mixed-INT8 models. Turn off to hide CPU from the Live and Photo compute pickers. The Bench tab always measures CPU.") {
            Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("Allow CPU inference", style = IosType.body, color = ios.label, modifier = Modifier.weight(1f))
                Switch(checked = allowCPU, onCheckedChange = { allowCPU = it })
            }
        }

        // E. Custom models
        FormSection(
            headerContent = {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Custom models", style = IosType.footnote, color = ios.secondaryLabel)
                    Text("BETA", style = IosType.caption2Bold, color = IosOrange, modifier = Modifier.clip(CircleShape).background(IosOrange.copy(alpha = 0.25f)).padding(horizontal = 5.dp, vertical = 1.dp))
                }
            },
            footer = "Import a folder or .zip containing model.ncnn.param, model.ncnn.bin and metadata.yaml. Imported models appear in the Live, Photo, and Bench pickers the next time you open that tab. The first load may take a moment.",
        ) {
            Row(Modifier.fillMaxWidth().clickable { treePicker.launch(null) }.padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Icon(Icons.Filled.AddCircleOutline, null, tint = ios.accent, modifier = Modifier.size(20.dp))
                Text("Load custom ncnn model (folder)", style = IosType.body, color = ios.accent)
            }
            HorizontalDivider(color = ios.separator)
            Row(Modifier.fillMaxWidth().clickable { zipPicker.launch(arrayOf("application/zip")) }.padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Icon(Icons.Filled.AddCircleOutline, null, tint = ios.accent, modifier = Modifier.size(20.dp))
                Text("Load custom ncnn model (.zip)", style = IosType.body, color = ios.accent)
            }
            customModels.forEach { m ->
                HorizontalDivider(color = ios.separator)
                Row(Modifier.fillMaxWidth().padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Icon(Icons.Filled.ViewInAr, null, tint = ios.secondaryLabel, modifier = Modifier.size(20.dp))
                    Text(m.fullName, style = IosType.body, color = ios.label, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                    Icon(Icons.Filled.Delete, "Delete", tint = IosRed, modifier = Modifier.size(20.dp).clickable { app.catalog.deleteCustom(m); refreshCustom() })
                }
            }
        }

        // F. Erase
        FormSection(footer = "$n saved ${if (n == 1) "run" else "runs"}.") {
            Row(Modifier.fillMaxWidth().clickable(enabled = n > 0) { confirmErase = true }.padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Icon(Icons.Filled.Delete, null, tint = if (n > 0) IosRed else ios.tertiaryLabel, modifier = Modifier.size(20.dp))
                Text("Erase all benchmark history", style = IosType.body, color = if (n > 0) IosRed else ios.tertiaryLabel)
            }
        }
        Spacer(Modifier.size(24.dp))
    }

    if (confirmErase) AlertDialog(
        onDismissRequest = { confirmErase = false },
        title = { Text("Erase all history?") },
        text = { Text("This permanently deletes all $n saved benchmark runs.") },
        confirmButton = { TextButton(onClick = { app.history.clear(); confirmErase = false }) { Text("Erase", color = IosRed) } },
        dismissButton = { TextButton(onClick = { confirmErase = false }) { Text("Cancel") } },
    )
    importError?.let { msg ->
        AlertDialog(
            onDismissRequest = { importError = null },
            title = { Text("Import failed") },
            text = { Text(msg) },
            confirmButton = { TextButton(onClick = { importError = null }) { Text("OK") } },
        )
    }
}

// ---- form pieces -----------------------------------------------------------------------------

@Composable
private fun FormSection(
    header: String? = null, footer: String? = null, headerContent: (@Composable () -> Unit)? = null,
    content: @Composable () -> Unit,
) {
    val ios = LocalIosColors.current
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
        if (headerContent != null) Box(Modifier.padding(start = 16.dp, bottom = 6.dp)) { headerContent() }
        else if (header != null) Text(header.uppercase(), style = IosType.footnote, color = ios.secondaryLabel, modifier = Modifier.padding(start = 16.dp, bottom = 6.dp))
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ios.systemBackground).padding(horizontal = 16.dp, vertical = 6.dp)) { content() }
        if (footer != null) Text(footer, style = IosType.footnote, color = ios.secondaryLabel, modifier = Modifier.padding(start = 16.dp, top = 6.dp, end = 16.dp))
    }
}

@Composable
private fun DisclosureRow(open: Boolean, onToggle: () -> Unit, header: @Composable () -> Unit, body: @Composable () -> Unit) {
    val ios = LocalIosColors.current
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().clickable { onToggle() }.padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.weight(1f)) { header() }
            Icon(Icons.Filled.ExpandMore, null, tint = ios.accent, modifier = Modifier.size(20.dp).rotate(if (open) 180f else 0f))
        }
        AnimatedVisibility(open) { Box(Modifier.padding(bottom = 8.dp)) { body() } }
    }
}

@Composable
private fun LabelRow(icon: ImageVector, text: String) {
    val ios = LocalIosColors.current
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Icon(icon, null, tint = ios.accent, modifier = Modifier.size(20.dp))
        Text(text, style = IosType.body, color = ios.label)
    }
}

@Composable
private fun GroupLabel(text: String) {
    val ios = LocalIosColors.current
    Text(text.uppercase(), style = IosType.caption2Bold, color = ios.tertiaryLabel, modifier = Modifier.padding(top = 2.dp))
}

@Composable
private fun LinkChip(title: String, icon: ImageVector, url: String) {
    val ios = LocalIosColors.current
    val ctx = LocalContext.current
    Row(
        Modifier.clip(CircleShape).background(ios.accent.copy(alpha = 0.14f))
            .clickable { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
            .padding(horizontal = 12.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Icon(icon, null, tint = ios.accent, modifier = Modifier.size(12.dp))
        Text(title, style = IosType.captionSemibold, color = ios.accent, maxLines = 1)
    }
}

@Composable
private fun AckRow(a: Ack) {
    val ios = LocalIosColors.current
    val ctx = LocalContext.current
    Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        if (a.logo != null) Image(painterResource(a.logo), null, modifier = Modifier.size(40.dp).clip(RoundedCornerShape(8.dp)))
        else Box(Modifier.size(40.dp).clip(RoundedCornerShape(8.dp)).background(ios.quaternaryFill), contentAlignment = Alignment.Center) {
            Text(a.name.take(1), style = IosType.headline, color = ios.secondaryLabel)
        }
        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(a.name, style = IosType.calloutSemibold, color = ios.label)
            Text(a.blurb, style = IosType.caption, color = ios.secondaryLabel)
            Row(
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp),
                modifier = Modifier.clickable { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(a.repo))) },
            ) {
                Icon(Icons.Filled.Link, null, tint = ios.accent, modifier = Modifier.size(12.dp))
                Text(a.repo.removePrefix("https://"), style = IosType.caption, color = ios.accent)
            }
        }
    }
}

@Composable
private fun AccessRow(icon: ImageVector, title: String, detail: String) {
    val ios = LocalIosColors.current
    Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Box(Modifier.width(22.dp), contentAlignment = Alignment.Center) { Icon(icon, null, tint = ios.secondaryLabel, modifier = Modifier.size(16.dp)) }
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(title, style = IosType.captionSemibold, color = ios.label)
            Text(detail, style = IosType.caption, color = ios.secondaryLabel)
        }
    }
}

@Suppress("unused")
private val colorUnused = Color.Unspecified
