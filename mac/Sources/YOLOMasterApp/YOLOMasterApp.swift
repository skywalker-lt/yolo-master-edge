// YOLOMasterApp — SwiftUI frontend for the Core ML runner (YOLOMasterKit backend).
//
// Pipeline:  choose model + source  ->  RUN (infer the whole set once, progress bar)  ->
//            browse the Finder + tune conf/iou/style/label in real time (cheap NMS/redraw
//            from cached candidates, NO re-inference)  ->  Export writes with the tuned params.
//   image  -> Run infers 1 -> tune -> Save
//   folder -> Run infers all (cache) -> Finder (Icons/List/Gallery) + arrows to browse -> Export folder
//   video  -> scrub a frame (infers it) -> tune -> Export video
//
// Build & run:  swift run -c release --package-path mac YOLOMasterApp   |   Bundle: mac/make_app.sh
import SwiftUI
import AppKit
import UniformTypeIdentifiers
import CoreGraphics
import ImageIO
import YOLOMasterKit

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
        WindowGroup("YOLO-Master · Core ML") { ContentView().frame(minWidth: 1120, minHeight: 720) }
            .windowStyle(.titleBar)
    }
}

// ---- async, cached thumbnails ----
func makeThumbnail(_ url: URL, max: CGFloat) -> NSImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                 kCGImageSourceThumbnailMaxPixelSize: max,
                                 kCGImageSourceCreateThumbnailWithTransform: true]
    guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
    return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
}
final class ThumbCache {
    static let shared = ThumbCache()
    private let cache = NSCache<NSString, NSImage>()
    private let queue = DispatchQueue(label: "com.yolomaster.thumb", qos: .userInitiated, attributes: .concurrent)
    init() { cache.countLimit = 800 }
    func thumb(_ url: URL, max: CGFloat, _ done: @escaping (NSImage?) -> Void) {
        let key = "\(Int(max))|\(url.path)" as NSString
        if let img = cache.object(forKey: key) { done(img); return }
        queue.async { [weak self] in
            let img = makeThumbnail(url, max: max)
            if let img { self?.cache.setObject(img, forKey: key) }
            DispatchQueue.main.async { done(img) }
        }
    }
}
struct AsyncThumb: View {
    let url: URL; var max: CGFloat = 128; var fit: Bool = false
    @State private var image: NSImage?
    var body: some View {
        Group {
            if let image {
                if fit { Image(nsImage: image).resizable().scaledToFit() }
                else { Image(nsImage: image).resizable().scaledToFill() }
            } else {
                Rectangle().fill(Color.gray.opacity(0.15))
            }
        }
        .onAppear { if image == nil { ThumbCache.shared.thumb(url, max: max) { image = $0 } } }
    }
}

// ---------- inference engine (two-phase: forward-once + cheap tuning) ----------
final class InferenceEngine: ObservableObject {
    @Published var resultImage: NSImage?
    @Published var detCount = 0
    @Published var modelSummary = ""
    @Published var status = "Choose a model (.mlpackage) + a source, then Run."
    @Published var busy = false
    @Published var exporting = false
    @Published var hasResults = false          // folder: inference cache ready
    @Published var progress: Double?
    @Published var outputURL: URL?

    private var detector: Detector?
    private var key = ""
    private var detNames: [String] = []
    private var currentCG: CGImage?
    private var currentCands: [Detection] = []
    private var currentMs = 0.0
    private var lastAnnotated: CGImage?
    private var folderCache: [FolderItem] = []
    private var folderInput: URL?
    private let queue = DispatchQueue(label: "com.yolomaster.inference")

    func resetResults() {
        hasResults = false; folderCache = []; folderInput = nil; outputURL = nil
        resultImage = nil; detCount = 0; currentCG = nil; currentCands = []
        status = "Ready — press Run."
    }

    // ---- image / video-frame: forward one, cache candidates, render ----
    func previewURL(model: URL, image: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        guard let cg = loadCGImage(image) else { publish(error: "Could not read image."); return }
        preview(model: model, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label)
    }
    func preview(model: URL, cg: CGImage, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        busy = true; progress = nil; status = "Inferring…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                let raw = try det.forward(cg)
                self.currentCG = cg; self.currentCands = det.candidates(raw); self.currentMs = raw.inferMs
                self.detNames = det.classNames
                let summary = det.summary
                DispatchQueue.main.async { self.modelSummary = summary }
                self.render(conf: conf, iou: iou, style: style, label: label)
            } catch { self.publish(error: "Inference failed: \(error.localizedDescription)") }
        }
    }

    // ---- folder: infer ALL once (progress), cache candidates ----
    func runFolder(model: URL, input: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        busy = true; exporting = false; hasResults = false; progress = 0; outputURL = nil; status = "Inferring folder…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                self.detNames = det.classNames
                let items = inferFolder(det, input: input, confFloor: 0.05) { done, total in
                    DispatchQueue.main.async {
                        self.progress = total > 0 ? Double(done) / Double(total) : nil
                        self.status = "Inferring \(done)/\(total)…"
                    }
                }
                self.folderCache = items; self.folderInput = input
                let summary = det.summary
                if let first = items.first, let cg = loadCGImage(first.url) {
                    self.currentCG = cg; self.currentCands = first.candidates; self.currentMs = 0
                }
                DispatchQueue.main.async {
                    self.modelSummary = summary; self.hasResults = !items.isEmpty
                    self.busy = false; self.progress = nil
                    self.status = "Inferred \(items.count) images — browse & tune, then Export"
                }
                self.render(conf: conf, iou: iou, style: style, label: label)
            } catch { self.publish(error: "Inference failed: \(error.localizedDescription)") }
        }
    }

    // ---- show a cached folder item (instant; no inference) ----
    func showFolder(index i: Int, url: URL, conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        queue.async { [weak self] in
            guard let self, let cg = loadCGImage(url) else { return }
            self.currentCG = cg
            self.currentCands = self.folderCache.indices.contains(i) ? self.folderCache[i].candidates : []
            self.currentMs = 0
            self.render(conf: conf, iou: iou, style: style, label: label)
        }
    }

    // ---- tuning: cheap re-NMS + redraw of the current frame ----
    private var pendingRestyle: DispatchWorkItem?
    func restyle(conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        pendingRestyle?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.render(conf: conf, iou: iou, style: style, label: label) }
        pendingRestyle = item
        queue.async(execute: item)
    }
    private func render(conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        guard let cg = currentCG, !detNames.isEmpty else { DispatchQueue.main.async { self.busy = false }; return }
        let dets = Detector.nms(currentCands, conf: Float(conf), iou: CGFloat(iou))
        let annotated = annotate(cg, dets, names: detNames, style: style, label: label) ?? cg
        self.lastAnnotated = annotated
        let ns = NSImage(cgImage: annotated, size: NSSize(width: cg.width, height: cg.height))
        let ms = self.currentMs
        DispatchQueue.main.async {
            self.resultImage = ns; self.detCount = dets.count; self.busy = false
            self.status = ms > 0 ? "\(dets.count) detections · \(String(format: "%.1f", ms)) ms" : "\(dets.count) detections"
        }
    }

    // ---- export ----
    func exportFolder(conf: Double, iou: Double, style: BoxStyle, label: LabelMode) {
        guard let input = folderInput, !folderCache.isEmpty else { return }
        busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting folder…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.lastPathComponent + "_annotated")
        let cache = folderCache, names = detNames
        queue.async { [weak self] in
            guard let self else { return }
            let n = exportFolderCached(cache, output: out, names: names, conf: Float(conf), iou: CGFloat(iou), style: style, label: label) { done, total in
                DispatchQueue.main.async { self.progress = total > 0 ? Double(done)/Double(total) : nil; self.status = "Exporting \(done)/\(total)…" }
            }
            DispatchQueue.main.async {
                self.outputURL = out; self.busy = false; self.exporting = false; self.progress = nil
                self.status = "Exported \(n) images"
            }
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
                    self.status = "Exported \(stats.frames) frames @\(stats.fps)fps"
                }
            } catch { DispatchQueue.main.async { self.status = "Export failed: \(error.localizedDescription)"; self.busy = false; self.exporting = false } }
        }
    }

    private func reuseDetector(model: URL, compute: ComputeMode, key k: String) throws -> Detector {
        if let d = detector, key == k { return d }
        let d = try Detector(modelURL: model, compute: compute); detector = d; key = k; return d
    }
    private func publish(error: String) { DispatchQueue.main.async { self.status = error; self.busy = false; self.exporting = false; self.progress = nil } }
    func save() {
        guard let cg = lastAnnotated else { return }
        let panel = NSSavePanel(); panel.allowedContentTypes = [.jpeg, .png]; panel.nameFieldStringValue = "annotated.jpg"
        if panel.runModal() == .OK, let url = panel.url { saveCGImage(cg, to: url) }
    }
    func reveal() { if let u = outputURL { NSWorkspace.shared.activateFileViewerSelecting([u]) } }
}

// ---------- Finder (Icons / List / Gallery) ----------
enum FinderMode: String, CaseIterable { case icons, list, gallery }

struct FinderView: View {
    let images: [URL]
    @Binding var selected: Int
    @Binding var mode: FinderMode
    @Binding var iconSize: Double
    let onSelect: (Int) -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Picker("", selection: $mode) {
                    Image(systemName: "square.grid.2x2").tag(FinderMode.icons)
                    Image(systemName: "list.bullet").tag(FinderMode.list)
                    Image(systemName: "rectangle.grid.1x2").tag(FinderMode.gallery)
                }.pickerStyle(.segmented).labelsHidden().fixedSize()
                Spacer()
                Text("\(images.count) images").font(.caption).foregroundStyle(.secondary)
                if mode == .icons { Slider(value: $iconSize, in: 64...200).frame(width: 90) }
            }.padding(8)
            Divider()
            switch mode { case .icons: icons; case .list: list; case .gallery: gallery }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
    private var icons: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: iconSize), spacing: 8)], spacing: 8) {
                ForEach(images.indices, id: \.self) { i in
                    VStack(spacing: 3) {
                        AsyncThumb(url: images[i], max: 220)
                            .frame(width: iconSize, height: iconSize * 0.72).clipped().cornerRadius(5)
                            .overlay(RoundedRectangle(cornerRadius: 5).stroke(i == selected ? Color.accentColor : .clear, lineWidth: 3))
                        Text(images[i].lastPathComponent).font(.caption2).lineLimit(1).truncationMode(.middle).frame(width: iconSize)
                    }.contentShape(Rectangle()).onTapGesture { onSelect(i) }
                }
            }.padding(10)
        }
    }
    private var list: some View {
        ScrollView {
            LazyVStack(spacing: 1) {
                ForEach(images.indices, id: \.self) { i in
                    HStack(spacing: 8) {
                        AsyncThumb(url: images[i], max: 90).frame(width: 54, height: 38).clipped().cornerRadius(3)
                        Text(images[i].lastPathComponent).font(.callout).lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(i == selected ? Color.accentColor.opacity(0.25) : .clear)
                    .contentShape(Rectangle()).onTapGesture { onSelect(i) }
                }
            }
        }
    }
    private var gallery: some View {
        VStack(spacing: 6) {
            if images.indices.contains(selected) {
                AsyncThumb(url: images[selected], max: 800, fit: true).frame(maxWidth: .infinity, maxHeight: .infinity)
                Text(images[selected].lastPathComponent).font(.caption).lineLimit(1)
            }
            Divider()
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 4) {
                    ForEach(images.indices, id: \.self) { i in
                        AsyncThumb(url: images[i], max: 130).frame(width: 82, height: 58).clipped().cornerRadius(3)
                            .overlay(RoundedRectangle(cornerRadius: 3).stroke(i == selected ? Color.accentColor : .clear, lineWidth: 2))
                            .onTapGesture { onSelect(i) }
                    }
                }.padding(6)
            }.frame(height: 76)
        }.padding(8)
    }
}

// ---------- main UI ----------
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
    @State private var finderMode: FinderMode = .icons
    @State private var iconSize: Double = 108
    @State private var videoDur = 0.0
    @State private var scrubTime = 0.0
    @State private var scrubWork: DispatchWorkItem?
    @FocusState private var kbFocused: Bool

    private enum PickTarget { case model, source }
    private var sourceKind: SourceKind { sourceURL.map(classifySource) ?? .unknown }
    private var kindLabel: String {
        switch sourceKind { case .image: "image"; case .folder: "folder"; case .video: "video"; case .unknown: "unsupported" }
    }
    private var pickerTypes: [UTType] {
        if pickTarget == .source { return [.image, .movie, .mpeg4Movie, .folder] }
        let byId = ["com.apple.coreml.mlpackage", "com.apple.coreml.mlmodelc", "com.apple.coreml.model"].compactMap { UTType($0) }
        let byExt = ["mlpackage", "mlmodelc", "mlmodel"].compactMap { UTType(filenameExtension: $0) }
        let all = byId + byExt + [.package]; return all.isEmpty ? [.item] : all
    }

    var body: some View {
        HStack(spacing: 0) {
            controls.frame(width: 300).padding(16)
            Divider()
            if sourceKind == .folder && engine.hasResults && !engine.exporting {
                FinderView(images: folderImages, selected: $selectedIndex, mode: $finderMode, iconSize: $iconSize) { selectAndShow($0) }
                    .frame(width: 380)
                Divider()
            }
            VStack(spacing: 0) {
                preview.frame(maxWidth: .infinity, maxHeight: .infinity)
                    .overlay(alignment: .bottom) { if engine.busy { progressBar } }
                if sourceKind == .video && !engine.exporting { scrubberBar }
            }
        }
        .fileImporter(isPresented: $showPicker, allowedContentTypes: pickerTypes) { if case .success(let u) = $0 { assign(u) } }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers { _ = p.loadObject(ofClass: URL.self) { url, _ in guard let url else { return }; DispatchQueue.main.async { assign(url) } } }
            return true
        }
        .onChange(of: conf) { rerender() }
        .onChange(of: iou) { rerender() }
        .onChange(of: style) { rerender() }
        .onChange(of: label) { rerender() }
        .onChange(of: modelURL) { setupSource() }
        .onChange(of: sourceURL) { setupSource() }
        .focusable().focused($kbFocused).onAppear { DispatchQueue.main.async { kbFocused = true } }
        .onKeyPress(.leftArrow)  { step(-1, vertical: false); return .handled }
        .onKeyPress(.rightArrow) { step(1,  vertical: false); return .handled }
        .onKeyPress(.upArrow)    { step(-1, vertical: true);  return .handled }
        .onKeyPress(.downArrow)  { step(1,  vertical: true);  return .handled }
    }

    private func assign(_ url: URL) {
        switch url.pathExtension.lowercased() {
        case "mlpackage", "mlmodelc", "mlmodel": modelURL = url
        default: sourceURL = url
        }
    }
    private func setupSource() {
        engine.resetResults()
        guard let s = sourceURL else { return }
        switch classifySource(s) {
        case .folder: folderImages = listImages(s); selectedIndex = 0
        case .video:
            Task {
                let dur = await videoDuration(s)
                await MainActor.run { videoDur = dur; scrubTime = 0 }
                if let m = modelURL, let cg = await extractFrame(s, atSeconds: 0) {
                    await MainActor.run { engine.preview(model: m, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label) }
                }
            }
        default: break
        }
    }
    private func runInfer() {
        guard let m = modelURL, let s = sourceURL else { return }
        switch sourceKind {
        case .image:  engine.previewURL(model: m, image: s, compute: compute, conf: conf, iou: iou, style: style, label: label)
        case .folder: engine.runFolder(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label)
        default: break
        }
    }
    private func selectAndShow(_ i: Int) {
        guard folderImages.indices.contains(i) else { return }
        selectedIndex = i
        engine.showFolder(index: i, url: folderImages[i], conf: conf, iou: iou, style: style, label: label)
    }
    private func rerender() { engine.restyle(conf: conf, iou: iou, style: style, label: label) }
    private func scrubFrame() {
        scrubWork?.cancel()
        let t = scrubTime
        let work = DispatchWorkItem {
            guard let m = modelURL, let s = sourceURL else { return }
            Task { if let cg = await extractFrame(s, atSeconds: t) {
                await MainActor.run { engine.preview(model: m, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label) } } }
        }
        scrubWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1, execute: work)
    }
    private var gridColumns: Int { max(1, Int((380.0 - 24) / (iconSize + 8))) }
    private func step(_ dir: Int, vertical: Bool) {
        switch sourceKind {
        case .folder where engine.hasResults && !folderImages.isEmpty:
            let stride = (vertical && finderMode == .icons) ? gridColumns : 1
            selectAndShow(min(max(0, selectedIndex + dir * stride), folderImages.count - 1))
        case .video:
            scrubTime = min(max(0, scrubTime + Double(dir) * (vertical ? 1.0 : 0.2)), max(videoDur, 0.0)); scrubFrame()
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
            slider(title: "conf", value: $conf, range: 0.05...0.95)
            slider(title: "iou",  value: $iou,  range: 0.10...0.90)
            labeled("Box style") {
                Picker("", selection: $style) { ForEach(BoxStyle.allCases, id: \.self) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).labelsHidden()
            }
            labeled("Label") {
                Picker("", selection: $label) { ForEach(LabelMode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).labelsHidden()
            }
            labeled("Compute") {
                Picker("", selection: $compute) { ForEach(ComputeMode.allCases, id: \.self) { Text($0.rawValue).tag($0) } }.pickerStyle(.menu).labelsHidden()
            }
            actionRow
            Spacer()
            Text(engine.status).font(.callout).foregroundStyle(.secondary)
            if !engine.modelSummary.isEmpty { Text(engine.modelSummary).font(.caption2).foregroundStyle(.tertiary).textSelection(.enabled) }
        }
    }

    @ViewBuilder private var actionRow: some View {
        HStack {
            switch sourceKind {
            case .image:
                Button("Run") { runInfer() }.buttonStyle(.borderedProminent).disabled(sourceURL == nil || engine.busy)
                Button("Save…") { engine.save() }.disabled(engine.resultImage == nil)
            case .folder:
                Button(engine.hasResults ? "Re-run" : "Run") { runInfer() }.buttonStyle(.borderedProminent).disabled(sourceURL == nil || engine.busy)
                Button("Export →") { engine.exportFolder(conf: conf, iou: iou, style: style, label: label) }.disabled(!engine.hasResults || engine.busy)
                if engine.outputURL != nil { Button("Reveal") { engine.reveal() } }
            case .video:
                Button("Export video →") {
                    if let m = modelURL, let s = sourceURL { engine.exportVideo(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label) }
                }.buttonStyle(.borderedProminent).disabled(sourceURL == nil || engine.busy)
                if engine.outputURL != nil { Button("Reveal") { engine.reveal() } }
            case .unknown: Button("Run") {}.disabled(true)
            }
        }
    }

    private var progressBar: some View {
        VStack(spacing: 4) {
            if let p = engine.progress { ProgressView(value: p) } else { ProgressView().progressViewStyle(.linear) }
            Text(engine.status).font(.caption)
        }.padding(10).frame(maxWidth: .infinity).background(.ultraThinMaterial)
    }

    private var preview: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if let img = engine.resultImage {
                Image(nsImage: img).resizable().scaledToFit().padding(12)
            } else if sourceKind == .folder && !engine.hasResults && !engine.busy {
                VStack(spacing: 8) {
                    Image(systemName: "folder").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text("\(folderImages.count) images — press Run to infer").foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: sourceKind == .video ? "film" : "photo").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text(sourceURL == nil ? "Choose a model + source" : "Press Run").foregroundStyle(.secondary)
                }
            }
        }
    }

    private var scrubberBar: some View {
        VStack(spacing: 2) {
            Slider(value: $scrubTime, in: 0...max(videoDur, 0.01)) { editing in if !editing { scrubFrame() } }
            Text("preview frame @ \(String(format: "%.2f", scrubTime))s / \(String(format: "%.1f", videoDur))s")
                .font(.caption2).foregroundStyle(.secondary)
        }.padding(.horizontal, 12).padding(.vertical, 8).background(Color(nsColor: .windowBackgroundColor))
    }

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
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption).foregroundStyle(.secondary); content() }
    }
}
