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
    @Published var status = "Choose a model and an image."
    @Published var busy = false

    // touched only on `queue` (serial) -> no data race
    private var detector: Detector?
    private var key = ""
    private var lastAnnotated: CGImage?
    private let queue = DispatchQueue(label: "com.yolomaster.inference")

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
                let res = try det.detect(cg, conf: Float(conf), iou: CGFloat(iou))
                let annotated = annotate(cg, res.detections, names: det.classNames, style: style, label: label) ?? cg
                self.lastAnnotated = annotated
                let ns = NSImage(cgImage: annotated, size: NSSize(width: cg.width, height: cg.height))
                DispatchQueue.main.async {
                    self.resultImage = ns
                    self.detCount = res.detections.count
                    self.inferMs = res.inferMs
                    self.modelSummary = det.summary
                    self.status = "\(res.detections.count) detections · \(String(format: "%.1f", res.inferMs)) ms"
                    self.busy = false
                }
            } catch {
                self.publish(error: "Failed: \(error.localizedDescription)")
            }
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
        pickTarget == .model
            ? [UTType(filenameExtension: "mlpackage") ?? .package, UTType(filenameExtension: "mlmodelc") ?? .package, .folder]
            : [.image]
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
            switch pickTarget {
            case .model: modelURL = url
            case .image: imageURL = url
            }
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
