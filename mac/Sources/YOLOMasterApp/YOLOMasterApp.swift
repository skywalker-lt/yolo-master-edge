// YOLOMasterApp — SwiftUI frontend for the Core ML runner.
//
// Same backend as the CLI (YOLOMasterKit). Pick a .mlpackage + a source (image, folder,
// or video), tune conf/iou/style/label/compute, run on-device Core ML.
//   image  -> annotated preview; conf/iou/style/label update in real time (cached forward pass)
//   folder -> batch-annotate all images -> <folder>_annotated/  (progress + live preview)
//   video  -> annotate every frame -> <video>_annotated.mp4     (progress + live preview)
//
// Build & run:  swift run -c release --package-path mac YOLOMasterApp
// Bundle .app:  mac/make_app.sh
import SwiftUI
import AppKit
import UniformTypeIdentifiers
import CoreGraphics
import YOLOMasterKit

// Unbundled `swift run` launches as an accessory process (.prohibited) — the SwiftUI window
// never shows. Force a regular, foreground GUI app so both `swift run` and the .app work.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ note: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct YOLOMasterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    var body: some Scene {
        WindowGroup("YOLO-Master · Core ML") {
            ContentView().frame(minWidth: 940, minHeight: 620)
        }
        .windowStyle(.titleBar)
    }
}

// ---------- inference engine: shared backend, off the main thread ----------
final class InferenceEngine: ObservableObject {
    @Published var resultImage: NSImage?
    @Published var detCount = 0
    @Published var inferMs = 0.0
    @Published var modelSummary = ""
    @Published var status = "Choose or drag a model (.mlpackage) + a source (image / folder / video)."
    @Published var busy = false
    @Published var progress: Double?        // nil while indeterminate (video)
    @Published var outputURL: URL?          // folder/video result on disk

    // touched only on `queue` (serial) -> no data race
    private var detector: Detector?
    private var key = ""
    private var rawOutput: Detector.RawOutput?   // cached forward pass (image mode)
    private var sourceCG: CGImage?
    private var lastInferMs = 0.0
    private var lastAnnotated: CGImage?
    private let queue = DispatchQueue(label: "com.yolomaster.inference")

    // ===== image: forward once, then re-decode/annotate live =====
    func run(model: URL, image: URL, conf: Double, iou: Double,
             style: BoxStyle, label: LabelMode, compute: ComputeMode) {
        busy = true; progress = nil; outputURL = nil; status = "Running…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                guard let cg = loadCGImage(image) else { self.publish(error: "Could not read image."); return }
                let raw = try det.forward(cg)
                self.rawOutput = raw; self.sourceCG = cg; self.lastInferMs = raw.inferMs
                let summary = det.summary
                DispatchQueue.main.async { self.modelSummary = summary; self.inferMs = raw.inferMs }
                self.render(conf: conf, iou: iou, style: style, label: label)
            } catch { self.publish(error: "Failed: \(error.localizedDescription)") }
        }
    }

    private var pendingRestyle: DispatchWorkItem?
    /// Re-decode (conf/iou) + re-annotate (style/label) from the cached forward pass — NO model
    /// call. Drives real-time control changes; no-op until a forward has run. Coalesces drags.
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
            self.status = "\(dets.count) detections · \(String(format: "%.1f", ms)) ms"
        }
    }

    // ===== folder: batch through the shared pipeline =====
    func processFolder(model: URL, input: URL, compute: ComputeMode,
                       conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        busy = true; progress = 0; outputURL = nil; status = "Processing folder…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.lastPathComponent + "_annotated")
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: model.path + "|" + compute.rawValue)
                let stats = runFolder(det, input: input, output: out, conf: Float(conf), iou: CGFloat(iou),
                                      style: style, label: label) { done, total, last in
                    DispatchQueue.main.async {
                        self.progress = total > 0 ? Double(done) / Double(total) : nil
                        if let last { self.resultImage = NSImage(cgImage: last, size: NSSize(width: last.width, height: last.height)) }
                        self.status = "Folder \(done)/\(total)…"
                    }
                }
                DispatchQueue.main.async {
                    self.outputURL = out; self.busy = false; self.progress = nil
                    self.status = "Folder done: \(stats.processed)/\(stats.total) · mean \(String(format: "%.1f", stats.meanMs)) ms"
                }
            } catch { self.publish(error: "Folder failed: \(error.localizedDescription)") }
        }
    }

    // ===== video: annotate every frame through the shared pipeline =====
    func processVideo(model: URL, input: URL, compute: ComputeMode,
                      conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        busy = true; progress = nil; outputURL = nil; status = "Processing video…"
        let out = input.deletingLastPathComponent()
            .appendingPathComponent(input.deletingPathExtension().lastPathComponent + "_annotated.mp4")
        Task { [weak self] in
            guard let self else { return }
            do {
                let det = try Detector(modelURL: model, compute: compute)
                let stats = try await runVideo(det, input: input, output: out, conf: Float(conf), iou: CGFloat(iou),
                                               style: style, label: label) { frames, last in
                    DispatchQueue.main.async {
                        if let last { self.resultImage = NSImage(cgImage: last, size: NSSize(width: last.width, height: last.height)) }
                        self.status = "Video \(frames) frames…"
                    }
                }
                DispatchQueue.main.async {
                    self.outputURL = out; self.busy = false
                    self.status = "Video done: \(stats.frames) frames @\(stats.fps)fps · mean \(String(format: "%.1f", stats.meanMs)) ms"
                }
            } catch { DispatchQueue.main.async { self.status = "Video failed: \(error.localizedDescription)"; self.busy = false } }
        }
    }

    /// Build or reuse the Detector for (model, compute). Call on `queue`.
    private func reuseDetector(model: URL, compute: ComputeMode, key k: String) throws -> Detector {
        if let d = detector, key == k { return d }
        let d = try Detector(modelURL: model, compute: compute)
        detector = d; key = k
        return d
    }

    func clearCache() {
        queue.async { self.rawOutput = nil; self.sourceCG = nil }
        resultImage = nil; outputURL = nil; detCount = 0; progress = nil
        status = "Ready — press Run."
    }

    private func publish(error: String) {
        DispatchQueue.main.async { self.status = error; self.busy = false; self.progress = nil }
    }

    func save() {
        guard let cg = lastAnnotated else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.jpeg, .png]
        panel.nameFieldStringValue = "annotated.jpg"
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

    private enum PickTarget { case model, source }
    private var sourceKind: SourceKind { sourceURL.map(classifySource) ?? .unknown }
    private var kindLabel: String {
        switch sourceKind { case .image: "image"; case .folder: "folder"; case .video: "video"; case .unknown: "unsupported" }
    }
    private var canRun: Bool { modelURL != nil && sourceURL != nil && sourceKind != .unknown && !engine.busy }
    private var pickerTypes: [UTType] {
        if pickTarget == .source { return [.image, .movie, .mpeg4Movie, .folder] }
        // .mlpackage/.mlmodelc are package bundles — need their concrete Core ML UTTypes.
        let byId = ["com.apple.coreml.mlpackage", "com.apple.coreml.mlmodelc", "com.apple.coreml.model"].compactMap { UTType($0) }
        let byExt = ["mlpackage", "mlmodelc", "mlmodel"].compactMap { UTType(filenameExtension: $0) }
        let all = byId + byExt + [.package]
        return all.isEmpty ? [.item] : all
    }

    var body: some View {
        HStack(spacing: 0) {
            controls.frame(width: 300).padding(16)
            Divider()
            preview.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .fileImporter(isPresented: $showPicker, allowedContentTypes: pickerTypes) { result in
            guard case .success(let url) = result else { return }
            assign(url)
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in       // drag a model or source onto the window
            for p in providers {
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    DispatchQueue.main.async { assign(url) }
                }
            }
            return true
        }
        // conf/iou/style/label are frontend post-processing -> live update (image mode only).
        .onChange(of: conf) { rerender() }
        .onChange(of: iou) { rerender() }
        .onChange(of: style) { rerender() }
        .onChange(of: label) { rerender() }
    }

    /// Route a picked/dropped URL to model vs source by extension; a new source clears the cache.
    private func assign(_ url: URL) {
        switch url.pathExtension.lowercased() {
        case "mlpackage", "mlmodelc", "mlmodel": modelURL = url
        default: sourceURL = url; engine.clearCache()
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

            HStack {
                Button(action: runInference) {
                    HStack { if engine.busy { ProgressView().controlSize(.small) }; Text(runTitle) }
                        .frame(maxWidth: .infinity)
                }
                .keyboardShortcut(.return, modifiers: [])
                .disabled(!canRun)
                .buttonStyle(.borderedProminent)

                if engine.outputURL != nil {
                    Button("Reveal") { engine.reveal() }
                } else {
                    Button("Save…") { engine.save() }.disabled(engine.resultImage == nil || sourceKind != .image)
                }
            }
            if engine.busy {
                if let p = engine.progress { ProgressView(value: p) } else { ProgressView() }
            }

            Spacer()
            Text(engine.status).font(.callout).foregroundStyle(.secondary)
            if !engine.modelSummary.isEmpty {
                Text(engine.modelSummary).font(.caption2).foregroundStyle(.tertiary).textSelection(.enabled)
            }
        }
    }

    private var runTitle: String {
        switch sourceKind { case .folder: "Run folder"; case .video: "Run video"; default: "Run" }
    }

    private var preview: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if let img = engine.resultImage {
                Image(nsImage: img).resizable().scaledToFit().padding(12)
            } else if sourceKind == .image, let url = sourceURL, let ns = NSImage(contentsOf: url) {
                Image(nsImage: ns).resizable().scaledToFit().padding(12).opacity(0.55)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: sourceKind == .video ? "film" : sourceKind == .folder ? "folder" : "photo")
                        .font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text(sourceURL == nil ? "Choose a model + source, then Run"
                                          : "Press Run to process the \(kindLabel)").foregroundStyle(.secondary)
                }
            }
        }
    }

    private func runInference() {
        guard let m = modelURL, let s = sourceURL else { return }
        switch sourceKind {
        case .image:  engine.run(model: m, image: s, conf: conf, iou: iou, style: style, label: label, compute: compute)
        case .folder: engine.processFolder(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label)
        case .video:  engine.processVideo(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label)
        case .unknown: engine.status = "Unsupported source."
        }
    }
    /// Live post-processing only makes sense for a single cached image.
    private func rerender() {
        if sourceKind == .image { engine.restyle(conf: conf, iou: iou, style: style, label: label) }
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
