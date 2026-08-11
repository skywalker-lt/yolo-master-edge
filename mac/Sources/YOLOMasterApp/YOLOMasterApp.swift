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
import AVFoundation
@preconcurrency import YOLOMasterKit   // Detector/RawOutput aren't Sendable; we hop them to main safely

let brandColor = Color.accentColor   // default system accent (blue)

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
        WindowGroup("YOLO-Master CoreML Runner") { ContentView().frame(minWidth: 1120, minHeight: 720) }
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
    // The full modifier chain (~25 modifiers) exceeds the SwiftUI type-checker budget as ONE
    // expression ("unable to type-check this expression in reasonable time", pointing at
    // whichever modifier the budget dies on). Staging it through explicitly-typed computed
    // properties gives each stage its own budget. Behavior is identical.
    var body: some View { keyboardLayer }

    /// Stage 1: layout + panels (sheet / importer / drop).
    private var mainStage: some View {
        HStack(spacing: 0) {
            controls.frame(width: 300).padding(16)
            Divider()
            if !cameraOn && sourceKind == .folder && engine.hasResults && !engine.exporting {
                FinderView(images: folderImages, selected: $selectedIndex, mode: $finderMode, iconSize: $iconSize) { selectAndShow($0) }
                    .frame(width: 380)
                Divider()
            }
            VStack(spacing: 0) {
                preview.frame(maxWidth: .infinity, maxHeight: .infinity)
                    .overlay(alignment: .bottom) { if engine.busy && !cameraOn { progressBar } }
                if !cameraOn && sourceKind == .video && engine.hasResults && !engine.exporting { scrubberBar }
            }
        }
        .sheet(isPresented: $showInfo) { InfoView() }
        .fileImporter(isPresented: $showPicker, allowedContentTypes: pickerTypes) { if case .success(let u) = $0 { assign(u) } }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers { _ = p.loadObject(ofClass: URL.self) { url, _ in guard let url else { return }; DispatchQueue.main.async { assign(url) } } }
            return true
        }
    }

    /// Stage 2: cheap re-render observers (no re-inference).
    private var tuningObservers: some View {
        mainStage
            .onChange(of: conf) { rerender() }
            .onChange(of: iou) { rerender() }
            .onChange(of: nmsMode) { rerender() }
            .onChange(of: sigma) { if nmsMode == .clusterWeighted { rerender() } }
            .onChange(of: style) { rerender() }
            .onChange(of: label) { rerender() }
            .onChange(of: overlay) { rerender() }
    }

    /// Stage 3: observers that re-infer or re-target the source / follow playback.
    private var sourceObservers: some View {
        tuningObservers
            .onChange(of: preprocess) {   // preprocessing changes the forward pass -> re-infer (not a cheap re-render)
                if cameraOn { return }    // LiveCameraView hot-swaps the detector itself
                guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
                runInfer()
            }
            .onChange(of: tiling) {       // tiling changes the forward pass(es) -> re-infer, images/folders only
                guard !cameraOn, sourceKind == .image || sourceKind == .folder else { return }
                guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
                runInfer()
            }
            .onChange(of: modelURL) { setupSource() }
            .onChange(of: sourceURL) { setupSource() }
            .onChange(of: scrubTime) { if scrubbing { pc.seek(scrubTime) } }   // seek while dragging
            .onChange(of: pc.currentTime) { if pc.isPlaying && !scrubbing { scrubTime = pc.currentTime } }   // slider follows playback
            .onChange(of: pc.displayTime) { refreshVideoOverlays() }
    }

    /// Stage 4: keyboard focus + shortcuts + tint.
    private var keyboardLayer: some View {
        sourceObservers
            .focusable().focused($kbFocused).focusEffectDisabled().onAppear { DispatchQueue.main.async { kbFocused = true } }
            .onKeyPress(.leftArrow)  { step(-1, vertical: false); return .handled }
            .onKeyPress(.rightArrow) { step(1,  vertical: false); return .handled }
            .onKeyPress(.upArrow)    { step(-1, vertical: true);  return .handled }
            .onKeyPress(.downArrow)  { step(1,  vertical: true);  return .handled }
            .onKeyPress(.space) { toggleVideoPlayback() }   // space toggles play/pause of the inferred video
            .tint(brandColor)   // teal accent for buttons/controls (Live Camera keeps its own pink tint)
    }

    private func assign(_ url: URL) {
        switch url.pathExtension.lowercased() {
        case "mlpackage", "mlmodelc", "mlmodel": modelURL = url
        default: sourceURL = url
        }
    }
    private func setupSource() {
        pc.pause()
        engine.resetResults()
        sourceError = nil; folderImages = []
        guard let s = sourceURL else { return }
        switch classifySource(s) {
        case .folder:
            let others = folderNonImages(s)
            if !others.isEmpty {
                let sample = others.prefix(3).map { $0.lastPathComponent }.joined(separator: ", ")
                let more = others.count > 3 ? " (+\(others.count - 3) more)" : ""
                sourceError = "This folder isn’t images-only — it contains: \(sample)\(more). Pick a folder that holds only image files."
            } else {
                let imgs = listImages(s)
                if imgs.isEmpty { sourceError = "This folder has no images." }
                else { folderImages = imgs; selectedIndex = 0 }
            }
        case .video:
            pc.load(s)
            Task { let dur = await videoDuration(s); await MainActor.run { videoDur = dur; scrubTime = 0 } }
        case .unknown:
            sourceError = "Unsupported source. Choose an image, a video, or a folder of images."
        default: break
        }
    }
    private func runInfer() {
        guard let m = modelURL, let s = sourceURL, sourceError == nil else { return }
        switch sourceKind {
        case .image:  engine.previewURL(model: m, image: s, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess,
                                        tiling: TilingConfig(mode: tiling), nmsMode: nmsMode, sigma: sigma)
        case .folder: engine.runFolder(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess,
                                       tiling: TilingConfig(mode: tiling), nmsMode: nmsMode, sigma: sigma)
        case .video:  engine.runVideo(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label, preprocess: preprocess, overlay: overlay,
                                      nmsMode: nmsMode, sigma: sigma)
        default: break
        }
    }
    // ---- live camera (session lifecycle + detector build handled inside LiveCameraView) ----
    private func toggleVideoPlayback() -> KeyPress.Result {   // extracted so the view body type-checks
        guard sourceKind == .video, engine.hasResults, !cameraOn else { return .ignored }
        pc.togglePlay()
        return .handled
    }
    private func startCamera() { guard modelURL != nil, !engine.busy else { return }; pc.pause(); cameraOn = true }
    private func stopCamera() { cameraOn = false }

    private func selectAndShow(_ i: Int) {
        guard folderImages.indices.contains(i) else { return }
        selectedIndex = i
        engine.showFolder(index: i, url: folderImages[i], conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma)
    }
    /// Re-derive the shown video frame's stats + seg-mask overlay from the cache. Extracted from
    /// the body's onChange chain: inline, these two many-argument calls blow the SwiftUI
    /// type-checker budget ("unable to type-check this expression in reasonable time").
    private func refreshVideoOverlays() {
        guard sourceKind == .video, engine.hasResults else { return }
        let t: Double = pc.displayTime
        engine.setVideoFrameStats(time: t, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma)
        engine.updateVideoMask(time: t, conf: conf, iou: iou, overlay: overlay, nmsMode: nmsMode, sigma: sigma)
    }
    private func rerender() {
        if cameraOn { return }   // camera overlay reads conf/iou/style/label live — no engine re-render
        if sourceKind == .video {
            refreshVideoOverlays()   // overlay redraws on conf/iou/label automatically
        } else {
            engine.restyle(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma)
        }
    }
    private var gridColumns: Int { max(1, Int((380.0 - 24) / (iconSize + 8))) }
    private func step(_ dir: Int, vertical: Bool) {
        switch sourceKind {
        case .folder where engine.hasResults && !folderImages.isEmpty:
            let stride = (vertical && finderMode == .icons) ? gridColumns : 1
            selectAndShow(min(max(0, selectedIndex + dir * stride), folderImages.count - 1))
        case .video where engine.hasResults:
            scrubTime = min(max(0, scrubTime + Double(dir) * (vertical ? 1.0 : 0.2)), max(videoDur, 0.0))
            pc.seek(scrubTime)
        default: break
        }
    }

    private var controls: some View {
        VStack(spacing: 14) {
            HStack(spacing: 10) {
                Image(nsImage: NSImage(named: "NSApplicationIcon") ?? NSImage())
                    .resizable().frame(width: 30, height: 30)
                VStack(alignment: .leading, spacing: 0) {
                    Text("YOLO-Master").font(.headline)
                    Text("Core ML runner").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button { showInfo = true } label: { Image(systemName: "info.circle").font(.system(size: 16)) }
                    .buttonStyle(.borderless).help("About & Licenses")
            }

            ScrollView {
                VStack(spacing: 14) {
                    sectionBox("Files", "folder") {
                        fileRow(icon: "cube.box.fill", title: "Model",
                                value: modelURL?.lastPathComponent ?? "Choose .mlpackage…", set: modelURL != nil) {
                            pickTarget = .model; DispatchQueue.main.async { showPicker = true }
                        }
                        Divider()
                        fileRow(icon: "photo.on.rectangle.angled", title: "Source",
                                value: sourceURL.map { "\($0.lastPathComponent) · \(kindLabel)" } ?? "Choose image / folder / video…",
                                set: sourceURL != nil) {
                            pickTarget = .source; DispatchQueue.main.async { showPicker = true }
                        }
                    }
                    sectionBox("Preprocess", "aspectratio") {
                        segRow("Input fit") {
                            Picker("", selection: $preprocess) {
                                Text("Letterbox").tag(Detector.PreprocessMode.letterbox)
                                Text("Stretch").tag(Detector.PreprocessMode.stretch)
                            }.pickerStyle(.segmented).labelsHidden().disabled(cameraOn)
                        }
                        if cameraOn {
                            Text("Stop the camera to change the input fit.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    sectionBox("Tiling", "square.grid.3x3") {
                        segRow("Mode") {
                            Picker("", selection: $tiling) {
                                ForEach(TilingMode.allCases, id: \.self) { Text($0.label).tag($0) }
                            }.pickerStyle(.segmented).labelsHidden()
                                .disabled(cameraOn || sourceKind == .video)
                        }
                        if cameraOn || sourceKind == .video {
                            Text("Tiling applies to images and folders only.")
                                .font(.caption2).foregroundStyle(.secondary)
                        } else if tiling == .sparse {
                            Text("Global pass + \(modelInfoImgsz) tiles where the global pass found objects; runs every tile if it found none.")
                                .font(.caption2).foregroundStyle(.secondary)
                        } else if tiling == .dense {
                            Text("Global pass + every \(modelInfoImgsz) tile — slowest, best small-object recall.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    sectionBox("Detection", "slider.horizontal.3") {
                        sliderRow("Confidence", $conf, 0.05...0.95)
                        sliderRow("IoU (NMS)", $iou, 0.10...0.90)
                        segRow("NMS") {
                            Picker("", selection: $nmsMode) {
                                ForEach(NMSMode.allCases, id: \.self) { Text($0.label).tag($0) }
                            }.pickerStyle(.segmented).labelsHidden()
                        }
                        if nmsMode == .clusterWeighted {
                            sliderRow("Sigma", $sigma, 0.01...0.5)
                            Text("Survivor boxes are refined by score-weighted averaging over overlapping candidates.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    sectionBox("Appearance", "paintbrush.fill") {
                        if isSegModel && tiledActive {
                            Text("Masks are disabled in tiled modes (boxes only).")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        if isSegModel && !tiledActive {
                            segRow("Overlay") {
                                Picker("", selection: $overlay) { ForEach(SegOverlay.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) } }
                                    .pickerStyle(.segmented).labelsHidden()
                            }
                        }
                        if !(isSegModel && overlay == .masks) {   // box style is irrelevant with boxes hidden
                            segRow("Box style") {
                                Picker("", selection: $style) { ForEach(BoxStyle.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) } }
                                    .pickerStyle(.segmented).labelsHidden()
                            }
                        }
                        segRow("Label") {   // labels stay adjustable even in masks-only mode
                            Picker("", selection: $label) { ForEach(LabelMode.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) } }
                                .pickerStyle(.segmented).labelsHidden()
                        }
                    }
                    sectionBox("Device", "cpu") {
                        Picker("", selection: $compute) { ForEach(ComputeMode.allCases, id: \.self) { Text($0.label).tag($0) } }
                            .pickerStyle(.menu).labelsHidden().frame(maxWidth: .infinity, alignment: .leading).disabled(cameraOn)
                        if cameraOn {
                            Text("Stop the camera to change the compute backend.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    sectionBox("Inference", "chart.bar.doc.horizontal") { summaryContent }
                }
            }
            .disabled(engine.busy)   // lock every control during image/folder/video inference; tune after it finishes (camera isn't engine.busy)

            actionRow
            Text("© 2026 Thomas Li").font(.system(size: 9)).foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder private var actionRow: some View {
        VStack(spacing: 8) {
            if cameraOn {
                primaryButton("Stop Camera", "stop.fill") { stopCamera() }
            } else {
                sourceActions
                VStack(spacing: 4) {
                    Button { startCamera() } label: {
                        Label("Live Camera", systemImage: "camera.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent).tint(.pink).controlSize(.large)
                    .disabled(modelURL == nil || engine.busy)
                    if modelURL == nil {
                        Text("Load a model to enable the live camera").font(.caption2).foregroundStyle(.secondary)
                    } else if engine.busy {
                        Text("Finish or wait for the current inference before starting the camera.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    @ViewBuilder private var sourceActions: some View {
        VStack(spacing: 8) {
            switch sourceKind {
            case .image:
                primaryButton("Run", "play.fill") { runInfer() }.disabled(sourceURL == nil || engine.busy || sourceError != nil)
                secondaryButton("Save…", "square.and.arrow.down") { engine.save() }.disabled(engine.resultImage == nil)
            case .folder:
                primaryButton(engine.hasResults ? "Re-run inference" : "Run inference", "play.fill") { runInfer() }
                    .disabled(sourceURL == nil || engine.busy || sourceError != nil)
                HStack(spacing: 8) {
                    secondaryButton("Save image", "square.and.arrow.down") { engine.save() }
                        .disabled(!engine.hasResults || engine.busy)
                    secondaryButton("Export all", "square.and.arrow.up") { engine.exportFolder(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma) }
                        .disabled(!engine.hasResults || engine.busy)
                    if engine.outputURL != nil { revealButton }
                }
            case .video:
                primaryButton(engine.hasResults ? "Re-run inference" : "Run inference", "play.fill") { runInfer() }
                    .disabled(sourceURL == nil || engine.busy || sourceError != nil)
                HStack(spacing: 8) {
                    secondaryButton("Save frame", "square.and.arrow.down") { engine.saveVideoFrame(time: pc.displayTime, conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma) }
                        .disabled(!engine.hasResults || engine.busy)
                    secondaryButton("Export video", "square.and.arrow.up") { engine.exportVideo(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma) }
                        .disabled(!engine.hasResults || engine.busy)
                    if engine.outputURL != nil { revealButton }
                }
            case .unknown:
                EmptyView()
            }
        }
    }

    private func primaryButton(_ title: String, _ icon: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) { Label(title, systemImage: icon).frame(maxWidth: .infinity) }
            .buttonStyle(.borderedProminent).controlSize(.large)
    }
    private func secondaryButton(_ title: String, _ icon: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) { Label(title, systemImage: icon).frame(maxWidth: .infinity) }.controlSize(.large)
    }
    private var revealButton: some View {
        Button { engine.reveal() } label: { Image(systemName: "magnifyingglass") }.controlSize(.large)
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
            if cameraOn {
                LiveCameraView(modelURL: modelURL, compute: compute, preprocess: preprocess,
                               conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma,
                               overlay: overlay, style: style, label: label,
                               isSegment: $cameraIsSegment, mirror: $cameraMirror).padding(12)
            } else if let err = sourceError {
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 44)).foregroundStyle(.orange)
                    Text(err).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 380)
                }.padding(24)
            } else if sourceKind == .video && engine.hasResults {
                VideoStage(engine: engine, pc: pc, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, overlay: overlay, style: style, label: label).padding(12)
            } else if let img = engine.resultImage {
                Image(nsImage: img).resizable().scaledToFit().padding(12)
            } else if (sourceKind == .folder || sourceKind == .video) && !engine.hasResults && !engine.busy {
                VStack(spacing: 8) {
                    Image(systemName: sourceKind == .video ? "film" : "folder").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text(sourceKind == .video ? "Press Run to infer the video"
                                              : "\(folderImages.count) images — press Run to infer").foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: sourceKind == .video ? "film" : "photo").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text(sourceURL != nil ? "Press Run"
                         : modelURL == nil ? "Choose a model + source"
                         : "Choose an image / folder / video — or start Live Camera").foregroundStyle(.secondary)
                }
            }
        }
    }

    private var scrubberBar: some View {
        HStack(spacing: 12) {
            Button { pc.togglePlay() } label: {
                Image(systemName: pc.isPlaying ? "pause.fill" : "play.fill").font(.title3).frame(width: 22)
            }.buttonStyle(.borderless)
            VStack(spacing: 2) {
                Slider(value: $scrubTime, in: 0...max(videoDur, 0.01)) { editing in
                    scrubbing = editing
                    if editing { wasPlaying = pc.isPlaying; pc.pause() }
                    else { pc.seek(scrubTime); if wasPlaying { pc.togglePlay() } }
                }
                Text("\(String(format: "%.2f", scrubTime))s / \(String(format: "%.1f", videoDur))s")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }.padding(.horizontal, 12).padding(.vertical, 8).background(Color(nsColor: .windowBackgroundColor))
    }

    private func sectionBox<C: View>(_ title: String, _ icon: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)                       // title: left edge aligns with the card below
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                .padding(.leading, 2)
            VStack(alignment: .leading, spacing: 14) { content() }   // roomier gap between option rows
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)                                          // breathing room inside the card
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
        }
    }
    private func fileRow(icon: String, title: String, value: String, set: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 15)).foregroundStyle(set ? brandColor : .secondary).frame(width: 20)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.caption).foregroundStyle(.secondary)
                    Text(value).font(.callout).lineLimit(1).truncationMode(.middle).foregroundStyle(set ? Color.primary : .secondary)
                }
                Spacer(minLength: 4)
                Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())   // whole row is the hit target, not just the text
        }.buttonStyle(.plain)
    }
    private func sliderRow(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title).font(.callout)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue)).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
            }
            Slider(value: value, in: range)
        }
    }
    private func segRow<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.callout); content() }
    }

    // ---- inference summary panel ----
    @ViewBuilder private var summaryContent: some View {
        if let s = engine.infer {
            if let info = engine.modelInfo {
                statRow("Model", info.name)
                statRow("Input", "\(info.imgsz) × \(info.imgsz) px")
                statRow("Classes", "\(info.nc)")
                statRow("Compute", info.compute)
            }
            Divider()
            statRow(isVideoSource ? "Frames" : (s.count > 1 ? "Images" : "Frame"), "\(s.count)")
            statRow("Model-only", speedText(s.meanMs, s.fps))
            statRow("Overall", speedText(s.wallMeanMs, s.wallFps))
            if s.count > 1 {
                statRow("Model min/max", String(format: "%.1f / %.1f ms", s.minMs, s.maxMs))
                statRow("Total time", String(format: "%.2fs wall · %.2fs model", s.wallMs / 1000, s.totalMs / 1000))
            }
            if let t = engine.tileStats {
                Divider()
                statRow("Tiling", tiling.label)
                statRow("Tiles", "\(t.tilesRun) run / \(t.tilesTotal) grid" + (t.capped > 0 ? " (capped)" : ""))
                if t.fallbacks > 0 { statRow("Fallback", "\(t.fallbacks) image(s) ran all tiles") }
            }
            if nmsMode == .clusterWeighted { statRow("NMS", nmsMode.label) }
            Divider()
            statRow("Detections", "\(engine.detCount)  (this frame)")
            if !engine.classCounts.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(engine.classCounts.prefix(12)) { c in
                        HStack {
                            Text(c.name).font(.caption)
                            Spacer(minLength: 8)
                            Text("\(c.count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                }.padding(.top, 2)
            }
        } else {
            HStack { Image(systemName: "info.circle").foregroundStyle(.tertiary)
                     Text("Run inference to see stats.").font(.caption).foregroundStyle(.secondary) }
        }
    }

    private var isVideoSource: Bool { sourceKind == .video }
    /// Tiled modes affect only image/folder sources; video/camera keep single-pass masks.
    private var tiledActive: Bool { tiling != .off && !cameraOn && (sourceKind == .image || sourceKind == .folder) }
    private func speedText(_ ms: Double, _ fps: Double) -> String {
        String(format: "%.1f", ms) + (isVideoSource ? " ms/frame · " : " ms/img · ")
            + String(format: "%.1f", fps) + (isVideoSource ? " fps" : " img/s")
    }
    private func statRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value).font(.caption.monospacedDigit()).lineLimit(1).truncationMode(.middle)
        }
    }
}
