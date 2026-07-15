// YOLOMasterApp — SwiftUI frontend for the Core ML runner.
//
// Uses the SAME backend as the CLI (YOLOMasterKit): pick a .mlpackage + an image, tune
// conf/iou/style/compute, run on-device Core ML, and view/save the annotated result.
//
// Build & run:  swift run -c release --package-path mac YOLOMasterApp
// Bundle .app:  mac/make_app.sh   (double-clickable, redistributable)
import SwiftUI
import AppKit
import UniformTypeIdentifiers
import CoreGraphics
import YOLOMasterKit

// An unbundled `swift run` executable launches as an accessory process (activation
// policy .prohibited) — SwiftUI makes the window but macOS never shows/focuses it.
// Force a regular, foreground GUI app so both `swift run` and the bundled .app work.
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
            ContentView()
                .frame(minWidth: 940, minHeight: 620)
        }
        .windowStyle(.titleBar)
    }
}

// ---------- inference engine: caches the Detector, runs off the main thread ----------
final class InferenceEngine: ObservableObject {
    @Published var resultImage: NSImage?
    @Published var detCount = 0
    @Published var inferMs = 0.0
    @Published var modelSummary = ""
    @Published var status = "Choose or drag a model (.mlpackage) + an image."
    @Published var busy = false

    // touched only on `queue` (serial) -> no data race
    private var detector: Detector?
    private var key = ""
    private var rawOutput: Detector.RawOutput?     // cached forward pass
    private var sourceCG: CGImage?
    private var lastInferMs = 0.0
    private var lastAnnotated: CGImage?
    private let queue = DispatchQueue(label: "com.yolomaster.inference")

    /// Expensive path: (re)load the model if needed, run the forward pass ONCE, cache the raw
    /// output + source image, then render with the current params. Call on model/image/compute change.
    func run(model: URL, image: URL, conf: Double, iou: Double,
             style: BoxStyle, label: LabelMode, compute: ComputeMode) {
        busy = true
        status = "Running…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det: Detector
                if let d = self.detector, self.key == k {
                    det = d
                } else {
                    det = try Detector(modelURL: model, compute: compute)
                    self.detector = det; self.key = k
                }
                guard let cg = loadCGImage(image) else {
                    self.publish(error: "Could not read image."); return
                }
                let raw = try det.forward(cg)
                self.rawOutput = raw; self.sourceCG = cg; self.lastInferMs = raw.inferMs
                let summary = det.summary
                DispatchQueue.main.async { self.modelSummary = summary; self.inferMs = raw.inferMs }
                self.render(conf: conf, iou: iou, style: style, label: label)
            } catch {
                self.publish(error: "Failed: \(error.localizedDescription)")
            }
        }
    }

    private var pendingRestyle: DispatchWorkItem?
    /// Cheap path: re-decode (conf/iou) + re-annotate (style/label) from the cached forward
    /// pass — NO model call. Drives real-time control changes; no-op until a forward has run.
    /// Coalesces rapid slider drags (cancels the superseded queued render).
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
            self.resultImage = ns
            self.detCount = dets.count
            self.status = "\(dets.count) detections · \(String(format: "%.1f", ms)) ms"
            self.busy = false
        }
    }

    private func publish(error: String) {
        DispatchQueue.main.async { self.status = error; self.busy = false }
    }

    func save() {
        guard let cg = lastAnnotated else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.jpeg, .png]
        panel.nameFieldStringValue = "annotated.jpg"
        if panel.runModal() == .OK, let url = panel.url {
            saveCGImage(cg, to: url)
        }
    }
}

// ---------- UI ----------
struct ContentView: View {
    @StateObject private var engine = InferenceEngine()

    @State private var modelURL: URL?
    @State private var imageURL: URL?
    @State private var conf = 0.25
    @State private var iou = 0.50
    @State private var style: BoxStyle = .hud
    @State private var label: LabelMode = .full
    @State private var compute: ComputeMode = .cpuAndGPU
    @State private var showPicker = false
    @State private var pickTarget: PickTarget = .model

    private enum PickTarget { case model, image }
    private var canRun: Bool { modelURL != nil && imageURL != nil && !engine.busy }
    private var pickerTypes: [UTType] {
        if pickTarget == .image { return [.image] }
        // .mlpackage/.mlmodelc are package bundles — need their concrete Core ML UTTypes
        // to be selectable (the generic .package / .folder don't enable them).
        let byId = ["com.apple.coreml.mlpackage", "com.apple.coreml.mlmodelc", "com.apple.coreml.model"]
            .compactMap { UTType($0) }
        let byExt = ["mlpackage", "mlmodelc", "mlmodel"].compactMap { UTType(filenameExtension: $0) }
        let all = byId + byExt + [.package]
        return all.isEmpty ? [.item] : all
    }

    var body: some View {
        HStack(spacing: 0) {
            controls
                .frame(width: 300)
                .padding(16)
            Divider()
            preview
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .fileImporter(isPresented: $showPicker, allowedContentTypes: pickerTypes) { result in
            guard case .success(let url) = result else { return }
            assign(url)
        }
        // Bulletproof fallback: drag a .mlpackage or image onto the window.
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers {
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    DispatchQueue.main.async { assign(url) }
                }
            }
            return true
        }
        // conf/iou/style/label are frontend post-processing -> live update, no re-inference.
        .onChange(of: conf) { rerender() }
        .onChange(of: iou) { rerender() }
        .onChange(of: style) { rerender() }
        .onChange(of: label) { rerender() }
    }

    /// Route a picked/dropped URL to model vs image by extension.
    private func assign(_ url: URL) {
        switch url.pathExtension.lowercased() {
        case "mlpackage", "mlmodelc", "mlmodel": modelURL = url
        default: imageURL = url
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("YOLO-Master · Core ML").font(.title3).bold()

            picker(title: "Model", value: modelURL?.lastPathComponent ?? "none",
                   button: "Choose .mlpackage…") { pickTarget = .model; DispatchQueue.main.async { showPicker = true } }
            picker(title: "Image", value: imageURL?.lastPathComponent ?? "none",
                   button: "Choose image…") { pickTarget = .image; DispatchQueue.main.async { showPicker = true } }

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
                    HStack { if engine.busy { ProgressView().controlSize(.small) }; Text("Run") }
                        .frame(maxWidth: .infinity)
                }
                .keyboardShortcut(.return, modifiers: [])
                .disabled(!canRun)
                .buttonStyle(.borderedProminent)

                Button("Save…") { engine.save() }
                    .disabled(engine.resultImage == nil)
            }

            Spacer()
            Text(engine.status).font(.callout).foregroundStyle(.secondary)
            if !engine.modelSummary.isEmpty {
                Text(engine.modelSummary).font(.caption2).foregroundStyle(.tertiary).textSelection(.enabled)
            }
        }
    }

    private var preview: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if let img = engine.resultImage {
                Image(nsImage: img).resizable().scaledToFit().padding(12)
            } else if let url = imageURL, let ns = NSImage(contentsOf: url) {
                Image(nsImage: ns).resizable().scaledToFit().padding(12).opacity(0.55)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "photo").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text("Annotated result appears here").foregroundStyle(.secondary)
                }
            }
        }
    }

    private func runInference() {
        guard let m = modelURL, let i = imageURL else { return }
        engine.run(model: m, image: i, conf: conf, iou: iou, style: style, label: label, compute: compute)
    }
    /// conf/iou/style/label are post-processing — re-render from the cached forward pass, no inference.
    private func rerender() {
        engine.restyle(conf: conf, iou: iou, style: style, label: label)
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
