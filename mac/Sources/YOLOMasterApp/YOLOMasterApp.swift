// YOLOMasterApp — SwiftUI frontend for the Core ML runner (YOLOMasterKit backend).
//
// Pick a .mlpackage + a source (image / folder / video). A preview frame loads and you tune
// conf/iou/style/label on it IN REAL TIME (cached forward pass — no re-inference). The finder
// picks which frame to tune:
//   image  -> the image; Save the result
//   folder -> file list (thumbnail + name); tune -> Export folder -> <folder>_annotated/
//   video  -> frame scrubber;                tune -> Export video  -> <video>_annotated.mp4
// The finder is hidden while exporting; a progress panel shows instead.
//
// Build & run:  swift run -c release --package-path mac YOLOMasterApp   |   Bundle: mac/make_app.sh
import SwiftUI
import AppKit
import UniformTypeIdentifiers
import CoreGraphics
import ImageIO
import YOLOMasterKit

// Unbundled `swift run` launches as an accessory process (.prohibited) — the window never
// shows. Force a regular, foreground GUI app so both `swift run` and the .app work.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ note: Notification) {
        NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct YOLOMasterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    var body: some Scene {
        WindowGroup("YOLO-Master · Core ML") {
            ContentView().frame(minWidth: 1040, minHeight: 700)
        }
        .windowStyle(.titleBar)
    }
}

/// Fast downsampled thumbnail (doesn't decode the full image).
func makeThumbnail(_ url: URL, max: CGFloat = 96) -> NSImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                 kCGImageSourceThumbnailMaxPixelSize: max,
                                 kCGImageSourceCreateThumbnailWithTransform: true]
    guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
    return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
}

// ---------- inference engine ----------
final class InferenceEngine: ObservableObject {
    @Published var resultImage: NSImage?
    @Published var detCount = 0
    @Published var inferMs = 0.0
    @Published var modelSummary = ""
    @Published var status = "Choose a model (.mlpackage) + a source (image / folder / video)."
    @Published var busy = false
    @Published var exporting = false        // true only during folder/video export
    @Published var progress: Double?        // nil = indeterminate (video export)
    @Published var outputURL: URL?          // folder/video export result

    private var detector: Detector?
    private var key = ""
    private var rawOutput: Detector.RawOutput?    // cached forward pass of the preview frame
    private var sourceCG: CGImage?
    private var lastInferMs = 0.0
    private var lastAnnotated: CGImage?
    private let queue = DispatchQueue(label: "com.yolomaster.inference")

    // ===== preview a single frame (image, folder pick, or video scrub) =====
    func previewURL(model: URL, image: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        guard let cg = loadCGImage(image) else { publish(error: "Could not read image."); return }
        preview(model: model, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label)
    }

    func preview(model: URL, cg: CGImage, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        busy = true; progress = nil; outputURL = nil; status = "Preview…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                let raw = try det.forward(cg)
                self.rawOutput = raw; self.sourceCG = cg; self.lastInferMs = raw.inferMs
                let summary = det.summary
                DispatchQueue.main.async { self.modelSummary = summary; self.inferMs = raw.inferMs }
                self.render(conf: conf, iou: iou, style: style, label: label)
            } catch { self.publish(error: "Preview failed: \(error.localizedDescription)") }
        }
    }

    private var pendingRestyle: DispatchWorkItem?
    /// Re-decode (conf/iou) + re-annotate (style/label) from the cached preview frame — NO model call.
    func restyle(conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        pendingRestyle?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.render(conf: conf, iou: iou, style: style, label: label) }
        pendingRestyle = item
        queue.async(execute: item)
    }

    private func render(conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        guard let det = detector, let raw = rawOutput, let cg = sourceCG else {
            DispatchQueue.main.async { self.busy = false }; return
        }
        let dets = det.decode(raw, conf: Float(conf), iou: CGFloat(iou))
        let annotated = annotate(cg, dets, names: det.classNames, style: style, label: label) ?? cg
        self.lastAnnotated = annotated
        let ns = NSImage(cgImage: annotated, size: NSSize(width: cg.width, height: cg.height))
        let ms = self.lastInferMs
        DispatchQueue.main.async {
            self.resultImage = ns; self.detCount = dets.count; self.busy = false
            self.status = "preview: \(dets.count) detections · \(String(format: "%.1f", ms)) ms"
        }
    }

    // ===== export: folder (batch) / video (per frame) with the tuned params =====
    func exportFolder(model: URL, input: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting folder…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.lastPathComponent + "_annotated")
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: model.path + "|" + compute.rawValue)
                let stats = runFolder(det, input: input, output: out, conf: Float(conf), iou: CGFloat(iou), style: style, label: label) { done, total, last in
                    DispatchQueue.main.async {
                        self.progress = total > 0 ? Double(done) / Double(total) : nil
                        if let last { self.resultImage = NSImage(cgImage: last, size: NSSize(width: last.width, height: last.height)) }
                        self.status = "Exporting \(done)/\(total)…"
                    }
                }
                DispatchQueue.main.async {
                    self.outputURL = out; self.busy = false; self.exporting = false; self.progress = nil
                    self.status = "Exported \(stats.processed)/\(stats.total) · mean \(String(format: "%.1f", stats.meanMs)) ms"
                }
            } catch { self.publish(error: "Export failed: \(error.localizedDescription)") }
        }
    }

    func exportVideo(model: URL, input: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        busy = true; exporting = true; progress = nil; outputURL = nil; status = "Exporting video…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.deletingPathExtension().lastPathComponent + "_annotated.mp4")
        Task { [weak self] in
            guard let self else { return }
            do {
                let det = try Detector(modelURL: model, compute: compute)
                let stats = try await runVideo(det, input: input, output: out, conf: Float(conf), iou: CGFloat(iou), style: style, label: label) { frames, last in
                    DispatchQueue.main.async {
                        if let last { self.resultImage = NSImage(cgImage: last, size: NSSize(width: last.width, height: last.height)) }
                        self.status = "Exporting \(frames) frames…"
                    }
                }
                DispatchQueue.main.async {
                    self.outputURL = out; self.busy = false; self.exporting = false
                    self.status = "Exported \(stats.frames) frames @\(stats.fps)fps · mean \(String(format: "%.1f", stats.meanMs)) ms"
                }
            } catch { self.publish(error: "Export failed: \(error.localizedDescription)") }
        }
    }

    private func reuseDetector(model: URL, compute: ComputeMode, key k: String) throws -> Detector {
        if let d = detector, key == k { return d }
        let d = try Detector(modelURL: model, compute: compute); detector = d; key = k; return d
    }

    private func publish(error: String) {
        DispatchQueue.main.async { self.status = error; self.busy = false; self.exporting = false; self.progress = nil }
    }

    func save() {
        guard let cg = lastAnnotated else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.jpeg, .png]; panel.nameFieldStringValue = "annotated.jpg"
        if panel.runModal() == .OK, let url = panel.url { saveCGImage(cg, to: url) }
    }
    func reveal() { if let u = outputURL { NSWorkspace.shared.activateFileViewerSelecting([u]) } }
}

// ---------- UI ----------
struct ContentView: View {
    @StateObject private var engine = InferenceEngine()

    @State private var modelURL: URL?
    @State private var sourceURL: URL?
    @State private var conf = 0.25
    @State private var iou = 0.50
    @State private var style: BoxStyle = .hud
    @State private var label: LabelMode = .full
    @State private var compute: ComputeMode = .cpuAndGPU
    @State private var showPicker = false
    @State private var pickTarget: PickTarget = .model
    @State private var folderImages: [URL] = []
    @State private var selectedIndex = 0
    @State private var videoDur = 0.0
    @State private var scrubTime = 0.0
    @State private var scrubWork: DispatchWorkItem?

    private enum PickTarget { case model, source }
    private var sourceKind: SourceKind { sourceURL.map(classifySource) ?? .unknown }
    private var kindLabel: String {
        switch sourceKind { case .image: "image"; case .folder: "folder"; case .video: "video"; case .unknown: "unsupported" }
    }
    private var canExport: Bool { modelURL != nil && sourceURL != nil && !engine.busy }
    private var pickerTypes: [UTType] {
        if pickTarget == .source { return [.image, .movie, .mpeg4Movie, .folder] }
        let byId = ["com.apple.coreml.mlpackage", "com.apple.coreml.mlmodelc", "com.apple.coreml.model"].compactMap { UTType($0) }
        let byExt = ["mlpackage", "mlmodelc", "mlmodel"].compactMap { UTType(filenameExtension: $0) }
        let all = byId + byExt + [.package]
        return all.isEmpty ? [.item] : all
    }

    var body: some View {
        HStack(spacing: 0) {
            controls.frame(width: 300).padding(16)
            Divider()
            if sourceKind == .folder && !engine.exporting {   // file browser (hidden during export)
                fileList.frame(width: 250)
                Divider()
            }
            VStack(spacing: 0) {
                preview.frame(maxWidth: .infinity, maxHeight: .infinity)
                if engine.exporting { exportPanel }
                else if sourceKind == .video { scrubberBar }
            }
        }
        .fileImporter(isPresented: $showPicker, allowedContentTypes: pickerTypes) { result in
            if case .success(let url) = result { assign(url) }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers {
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    DispatchQueue.main.async { assign(url) }
                }
            }
            return true
        }
        .onChange(of: conf) { rerender() }
        .onChange(of: iou) { rerender() }
        .onChange(of: style) { rerender() }
        .onChange(of: label) { rerender() }
        .onChange(of: modelURL) { setupSource() }
        .onChange(of: sourceURL) { setupSource() }
    }

    private func assign(_ url: URL) {
        switch url.pathExtension.lowercased() {
        case "mlpackage", "mlmodelc", "mlmodel": modelURL = url
        default: sourceURL = url
        }
    }

    private func setupSource() {
        guard let m = modelURL, let s = sourceURL else { return }
        switch sourceKind {
        case .image:
            engine.previewURL(model: m, image: s, compute: compute, conf: conf, iou: iou, style: style, label: label)
        case .folder:
            folderImages = listImages(s); selectedIndex = 0
            if let first = folderImages.first {
                engine.previewURL(model: m, image: first, compute: compute, conf: conf, iou: iou, style: style, label: label)
            }
        case .video:
            Task {
                let dur = await videoDuration(s)
                await MainActor.run { videoDur = dur; scrubTime = 0 }
                if let cg = await extractFrame(s, atSeconds: 0) {
                    await MainActor.run { engine.preview(model: m, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label) }
                }
            }
        case .unknown: break
        }
    }

    private func rerender() { engine.restyle(conf: conf, iou: iou, style: style, label: label) }

    private func scrubFrame() {
        scrubWork?.cancel()
        let t = scrubTime
        let work = DispatchWorkItem {
            guard let m = modelURL, let s = sourceURL else { return }
            Task {
                if let cg = await extractFrame(s, atSeconds: t) {
                    await MainActor.run { engine.preview(model: m, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label) }
                }
            }
        }
        scrubWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1, execute: work)
    }

    private func export() {
        guard let m = modelURL, let s = sourceURL else { return }
        switch sourceKind {
        case .folder: engine.exportFolder(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label)
        case .video:  engine.exportVideo(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label)
        default: break
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("YOLO-Master · Core ML").font(.title3).bold()

            picker(title: "Model", value: modelURL?.lastPathComponent ?? "none",
                   button: "Choose .mlpackage…") { pickTarget = .model; DispatchQueue.main.async { showPicker = true } }
            picker(title: "Source", value: sourceURL.map { "\($0.lastPathComponent)  (\(kindLabel))" } ?? "none",
                   button: "Choose image / folder / video…") { pickTarget = .source; DispatchQueue.main.async { showPicker = true } }

            slider(title: "conf", value: $conf, range: 0.01...0.95)
            slider(title: "iou",  value: $iou,  range: 0.10...0.90)
            labeled("Box style") {
                Picker("", selection: $style) { ForEach(BoxStyle.allCases, id: \.self) { Text($0.rawValue).tag($0) } }
                    .pickerStyle(.segmented).labelsHidden()
            }
            labeled("Label") {
                Picker("", selection: $label) { ForEach(LabelMode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }
                    .pickerStyle(.segmented).labelsHidden()
            }
            labeled("Compute") {
                Picker("", selection: $compute) { ForEach(ComputeMode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }
                    .pickerStyle(.menu).labelsHidden()
            }

            actionRow
            if engine.busy && !engine.exporting {
                HStack(spacing: 6) { ProgressView().controlSize(.small); Text("inferring…").font(.caption).foregroundStyle(.secondary) }
            }

            Spacer()
            Text(engine.status).font(.callout).foregroundStyle(.secondary)
            if !engine.modelSummary.isEmpty {
                Text(engine.modelSummary).font(.caption2).foregroundStyle(.tertiary).textSelection(.enabled)
            }
        }
    }

    @ViewBuilder private var actionRow: some View {
        HStack {
            switch sourceKind {
            case .image:
                Button("Save…") { engine.save() }
                    .buttonStyle(.borderedProminent).disabled(engine.resultImage == nil)
            case .folder, .video:
                Button(sourceKind == .video ? "Export video →" : "Export folder →") { export() }
                    .buttonStyle(.borderedProminent).disabled(!canExport)
                if engine.outputURL != nil { Button("Reveal") { engine.reveal() } }
            case .unknown:
                Button("Run") {}.disabled(true)
            }
        }
    }

    private var preview: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if let img = engine.resultImage {
                Image(nsImage: img).resizable().scaledToFit().padding(12)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: sourceKind == .video ? "film" : sourceKind == .folder ? "folder" : "photo")
                        .font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text(sourceURL == nil ? "Choose a model + source" : "Loading preview…").foregroundStyle(.secondary)
                }
            }
        }
    }

    // export progress panel (replaces the finder while exporting)
    private var exportPanel: some View {
        VStack(spacing: 6) {
            if let p = engine.progress { ProgressView(value: p) { Text(engine.status).font(.caption) } }
            else { ProgressView { Text(engine.status).font(.caption) } }
        }
        .padding(14)
        .frame(maxWidth: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // folder finder: scrollable file list (thumbnail + filename), lazy for large folders
    private var fileList: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("\(folderImages.count) images — click to preview & tune")
                .font(.caption).foregroundStyle(.secondary).padding(8)
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(folderImages.enumerated()), id: \.offset) { i, url in
                        fileRow(i, url)
                    }
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func fileRow(_ i: Int, _ url: URL) -> some View {
        HStack(spacing: 8) {
            if let ns = makeThumbnail(url, max: 64) {
                Image(nsImage: ns).resizable().scaledToFill().frame(width: 46, height: 32).clipped().cornerRadius(3)
            } else {
                Color.gray.opacity(0.2).frame(width: 46, height: 32).cornerRadius(3)
            }
            Text(url.lastPathComponent).font(.caption).lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(i == selectedIndex ? Color.accentColor.opacity(0.25) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture {
            selectedIndex = i
            if let m = modelURL {
                engine.previewURL(model: m, image: url, compute: compute, conf: conf, iou: iou, style: style, label: label)
            }
        }
    }

    // video finder: frame scrubber
    private var scrubberBar: some View {
        VStack(spacing: 2) {
            Slider(value: $scrubTime, in: 0...max(videoDur, 0.01)) { editing in if !editing { scrubFrame() } }
            Text("preview frame @ \(String(format: "%.2f", scrubTime))s / \(String(format: "%.1f", videoDur))s")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // ---- small view builders ----
    private func picker(title: String, value: String, button: String, action: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Button(button, action: action).frame(maxWidth: .infinity)
            Text(value).font(.caption2).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
        }
    }
    private func slider(title: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack { Text(title).font(.caption).foregroundStyle(.secondary); Spacer()
                     Text(String(format: "%.2f", value.wrappedValue)).font(.caption).monospacedDigit() }
            Slider(value: value, in: range)
        }
    }
    private func labeled<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }
}
