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

// ---------- stats models ----------
struct StatModelInfo: Equatable { let name: String; let imgsz: Int; let nc: Int; let compute: String }
struct ClassCount: Identifiable, Equatable { var id: String { name }; let name: String; let count: Int }

// ---------- inference engine (two-phase: forward-once + cheap tuning) ----------
final class InferenceEngine: ObservableObject, @unchecked Sendable {   // state is guarded by `queue` + main-hops
    @Published var resultImage: NSImage?
    @Published var detCount = 0
    @Published var modelSummary = ""
    @Published var status = "Choose a model (.mlpackage) + a source, then Run."
    @Published var busy = false
    @Published var exporting = false
    @Published var hasResults = false          // folder: inference cache ready
    @Published var progress: Double?
    @Published var outputURL: URL?
    @Published var modelInfo: StatModelInfo?   // model name / imgsz / classes / compute
    @Published var infer: InferSummary?        // count / avg / min / max / total / fps
    @Published var classCounts: [ClassCount] = []   // per-class breakdown of the current frame
    @Published var modelIsSegment = false      // drives the Masks/Boxes/Both overlay control

    private var detector: Detector?
    private var currentRaw: Detector.RawOutput?    // cached forward pass for the shown image (seg masks need protos)
    private var key = ""
    private var detNames: [String] = []
    private var currentCG: CGImage?
    private var currentCands: [Detection] = []
    private var currentMs = 0.0
    private var lastAnnotated: CGImage?
    private var folderCache: [FolderItem] = []
    private var folderInput: URL?
    private var videoCache: [[Detection]] = []
    private var videoRaws: [Detector.RawOutput?] = []   // per-frame proto (seg only) for on-demand masks
    private var videoDet: Detector?                       // the seg detector, to compute mask overlays
    @Published var videoMaskImg: CGImage?                 // mask overlay for the shown video frame
    @Published private(set) var videoFps: Double = 30
    @Published private(set) var videoURL: URL?
    @Published private(set) var videoSize: CGSize = .zero
    private var videoInput: URL?
    private let queue = DispatchQueue(label: "com.yolomaster.inference")

    func resetResults() {
        hasResults = false; folderCache = []; folderInput = nil; videoCache = []; videoInput = nil; videoURL = nil; videoSize = .zero; outputURL = nil
        resultImage = nil; detCount = 0; currentCG = nil; currentCands = []; currentRaw = nil
        videoRaws = []; videoDet = nil; videoMaskImg = nil
        infer = nil; classCounts = []
        status = "Ready — press Run."
    }

    // ---- image / video-frame: forward one, cache candidates, render ----
    func previewURL(model: URL, image: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay, preprocess: Detector.PreprocessMode) {
        guard let cg = loadCGImage(image) else { publish(error: "Could not read image."); return }
        preview(model: model, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess)
    }
    func preview(model: URL, cg: CGImage, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay, preprocess: Detector.PreprocessMode) {
        busy = true; progress = nil; status = "Inferring…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                det.preprocess = preprocess
                let raw = try det.forward(cg)
                self.currentCG = cg; self.currentCands = det.candidates(raw); self.currentMs = raw.inferMs
                self.currentRaw = det.isSegment ? raw : nil
                self.detNames = det.classNames
                let info = StatModelInfo(name: model.lastPathComponent, imgsz: det.imgsz, nc: det.nc, compute: compute.label)
                let s = InferSummary([raw.inferMs], wallMs: raw.inferMs)
                DispatchQueue.main.async { self.modelInfo = info; self.infer = s; self.modelIsSegment = det.isSegment }
                self.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay)
            } catch { self.publish(error: "Inference failed: \(error.localizedDescription)") }
        }
    }

    // ---- folder: infer ALL once (progress), cache candidates ----
    func runFolder(model: URL, input: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay, preprocess: Detector.PreprocessMode) {
        busy = true; exporting = false; hasResults = false; progress = 0; outputURL = nil; status = "Inferring folder…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                det.preprocess = preprocess
                self.detNames = det.classNames
                let (items, summary) = inferFolder(det, input: input, confFloor: 0.05) { done, total in
                    DispatchQueue.main.async {
                        self.progress = total > 0 ? Double(done) / Double(total) : nil
                        self.status = "Inferring \(done)/\(total)…"
                    }
                }
                self.folderCache = items; self.folderInput = input
                let info = StatModelInfo(name: model.lastPathComponent, imgsz: det.imgsz, nc: det.nc, compute: compute.label)
                if let first = items.first, let cg = loadCGImage(first.url) {
                    self.currentCG = cg; self.currentCands = first.candidates; self.currentMs = 0
                    self.currentRaw = det.isSegment ? try? det.forward(cg) : nil
                }
                DispatchQueue.main.async {
                    self.modelInfo = info; self.infer = summary; self.hasResults = !items.isEmpty
                    self.busy = false; self.progress = nil; self.modelIsSegment = det.isSegment
                    self.status = "Inferred \(items.count) images — browse & tune, then Export"
                }
                self.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay)
            } catch { self.publish(error: "Inference failed: \(error.localizedDescription)") }
        }
    }

    // ---- show a cached folder item (instant; re-forwards only for seg masks) ----
    func showFolder(index i: Int, url: URL, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay) {
        queue.async { [weak self] in
            guard let self, let cg = loadCGImage(url) else { return }
            self.currentCG = cg
            self.currentCands = self.folderCache.indices.contains(i) ? self.folderCache[i].candidates : []
            self.currentMs = 0
            self.currentRaw = (self.detector?.isSegment == true) ? try? self.detector?.forward(cg) : nil
            self.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay)
        }
    }

    // ---- tuning: cheap re-NMS + redraw of the current frame ----
    private var pendingRestyle: DispatchWorkItem?
    func restyle(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay) {
        pendingRestyle?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay) }
        pendingRestyle = item
        queue.async(execute: item)
    }
    private func render(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay) {
        guard let cg = currentCG, !detNames.isEmpty else { DispatchQueue.main.async { self.busy = false }; return }
        let dets = Detector.nms(currentCands, conf: Float(conf), iou: CGFloat(iou))
        var masks: [MaskBitmap] = [], drawBoxes = true
        if let det = detector, det.isSegment, let raw = currentRaw, overlay != .boxes {
            masks = dets.compactMap { det.maskImage($0, raw) }
            drawBoxes = overlay != .masks
        }
        let annotated = annotate(cg, dets, names: detNames, style: style, label: label, masks: masks, drawBoxes: drawBoxes) ?? cg
        self.lastAnnotated = annotated
        let ns = NSImage(cgImage: annotated, size: NSSize(width: cg.width, height: cg.height))
        let ms = self.currentMs
        var byClass: [Int: Int] = [:]
        for d in dets { byClass[d.cls, default: 0] += 1 }
        let breakdown = byClass.sorted { $0.value > $1.value }.map {
            ClassCount(name: self.detNames.indices.contains($0.key) ? self.detNames[$0.key] : "class\($0.key)", count: $0.value)
        }
        DispatchQueue.main.async {
            self.resultImage = ns; self.detCount = dets.count; self.busy = false; self.classCounts = breakdown
            self.status = ms > 0 ? "\(dets.count) detections · \(String(format: "%.1f", ms)) ms" : "\(dets.count) detections"
        }
    }

    // ---- export ----
    func exportFolder(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay) {
        guard let input = folderInput, !folderCache.isEmpty else { return }
        busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting folder…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.lastPathComponent + "_annotated")
        let cache = folderCache, names = detNames, det = detector
        queue.async { [weak self] in
            guard let self else { return }
            let n = exportFolderCached(cache, output: out, names: names, conf: Float(conf), iou: CGFloat(iou), style: style, label: label, detector: det, overlay: overlay) { done, total in
                DispatchQueue.main.async { self.progress = total > 0 ? Double(done)/Double(total) : nil; self.status = "Exporting \(done)/\(total)…" }
            }
            DispatchQueue.main.async {
                self.outputURL = out; self.busy = false; self.exporting = false; self.progress = nil
                self.status = "Exported \(n) images"
            }
        }
    }
    // ---- video: infer ALL frames once (progress), cache candidates ----
    func runVideo(model: URL, input: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, preprocess: Detector.PreprocessMode, overlay: SegOverlay) {
        busy = true; exporting = false; hasResults = false; progress = 0; outputURL = nil; status = "Inferring video…"
        Task { [weak self] in
            guard let self else { return }
            do {
                let det = try Detector(modelURL: model, compute: compute)
                det.preprocess = preprocess
                self.detNames = det.classNames
                let (frames, raws, summary, fps, size) = try await inferVideo(det, input: input, confFloor: 0.05) { done, est in
                    DispatchQueue.main.async {
                        self.progress = est > 0 ? min(1, Double(done) / Double(est)) : nil
                        self.status = "Inferring frame \(done)…"
                    }
                }
                let info = StatModelInfo(name: model.lastPathComponent, imgsz: det.imgsz, nc: det.nc, compute: compute.label)
                DispatchQueue.main.async {
                    self.videoCache = frames; self.videoRaws = raws; self.videoDet = det.isSegment ? det : nil
                    self.modelIsSegment = det.isSegment
                    self.videoFps = fps; self.videoInput = input; self.videoURL = input; self.videoSize = size; self.videoMaskImg = nil
                    self.modelInfo = info; self.infer = summary; self.hasResults = !frames.isEmpty
                    self.busy = false; self.progress = nil
                    self.status = "Inferred \(frames.count) frames — play / scrub & tune, then Export"
                    self.setVideoFrameStats(time: 0, conf: conf, iou: iou)
                    self.updateVideoMask(time: 0, conf: conf, iou: iou, overlay: overlay)
                }
            } catch { DispatchQueue.main.async { self.status = "Inference failed: \(error.localizedDescription)"; self.busy = false; self.progress = nil } }
        }
    }

    // ---- video overlay data: AVPlayer displays frames; we draw boxes from cached candidates at time ----
    var names: [String] { detNames }
    var videoIsSegment: Bool { videoDet != nil }
    private func videoFrameIndex(_ time: Double) -> Int {
        min(max(0, Int((time * videoFps).rounded())), max(0, videoCache.count - 1))
    }
    func detsAt(time: Double, conf: Double, iou: Double) -> [Detection] {
        guard !videoCache.isEmpty else { return [] }
        return Detector.nms(videoCache[videoFrameIndex(time)], conf: Float(conf), iou: CGFloat(iou))
    }
    /// Compute the seg mask overlay for the frame at `time` (background) and publish it. No-op for
    /// detection models or masks-off. Recomputed as the shown frame / conf / iou / overlay change.
    func updateVideoMask(time: Double, conf: Double, iou: Double, overlay: SegOverlay) {
        guard let det = videoDet, overlay != .boxes, !videoCache.isEmpty else {
            if videoMaskImg != nil { videoMaskImg = nil }
            return
        }
        let idx = videoFrameIndex(time)
        guard videoRaws.indices.contains(idx), let raw = videoRaws[idx] else { return }
        let cands = videoCache[idx]
        queue.async { [weak self] in
            let dets = Detector.nms(cands, conf: Float(conf), iou: CGFloat(iou))
            let img = det.maskOverlay(dets, raw)
            DispatchQueue.main.async { self?.videoMaskImg = img }
        }
    }
    /// Update the 'this frame' summary stats for the video frame at `time`.
    func setVideoFrameStats(time: Double, conf: Double, iou: Double) {
        let dets = detsAt(time: time, conf: conf, iou: iou)
        detCount = dets.count
        var byClass: [Int: Int] = [:]; for d in dets { byClass[d.cls, default: 0] += 1 }
        classCounts = byClass.sorted { $0.value > $1.value }.map {
            ClassCount(name: detNames.indices.contains($0.key) ? detNames[$0.key] : "class\($0.key)", count: $0.value)
        }
    }

    // ---- export video from cached candidates (NO inference) ----
    func exportVideo(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay) {
        guard let input = videoInput, !videoCache.isEmpty else { return }
        busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting video…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.deletingPathExtension().lastPathComponent + "_annotated.mp4")
        let frames = videoCache, names = detNames, rw = videoRaws, det = videoDet
        Task { [weak self] in
            guard let self else { return }
            do {
                let stats = try await exportVideoCached(input: input, output: out, framesCands: frames, names: names, conf: Float(conf), iou: CGFloat(iou), style: style, label: label, raws: rw, detector: det, overlay: overlay) { done, total in
                    DispatchQueue.main.async { self.progress = total > 0 ? Double(done) / Double(total) : nil; self.status = "Exporting \(done)/\(total)…" }
                }
                DispatchQueue.main.async {
                    self.outputURL = out; self.busy = false; self.exporting = false; self.progress = nil
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
enum FinderMode: String, CaseIterable { case icons, list }

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
                }.pickerStyle(.segmented).labelsHidden().fixedSize()
                Spacer()
                Text("\(images.count) images").font(.caption).foregroundStyle(.secondary)
                if mode == .icons { Slider(value: $iconSize, in: 64...200).frame(width: 90) }
            }.padding(8)
            Divider()
            switch mode { case .icons: icons; case .list: list }
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
}

// ---------- AVPlayer video stage (real-time playback) + live detection overlay ----------
final class PlayerController: ObservableObject {
    let player = AVPlayer()
    @Published var currentTime: Double = 0     // playhead (drives the slider during playback)
    @Published var displayTime: Double = 0     // time of the frame actually ON SCREEN (drives the overlay)
    @Published var isPlaying = false
    private var loaded: URL?
    private var timeObs: Any?
    private var endObs: NSObjectProtocol?
    func load(_ url: URL) {
        guard loaded != url else { return }
        loaded = url
        player.replaceCurrentItem(with: AVPlayerItem(url: url))
        currentTime = 0; displayTime = 0; isPlaying = false
        if timeObs == nil {
            timeObs = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 30), queue: .main) { [weak self] t in
                guard let self else { return }
                let s = t.seconds.isFinite ? t.seconds : 0
                self.currentTime = s
                if self.isPlaying { self.displayTime = s }   // during playback, boxes track the shown frame
            }
        }
        if endObs == nil {
            endObs = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: nil, queue: .main) { [weak self] _ in
                self?.player.seek(to: .zero); if self?.isPlaying == true { self?.player.play() }
            }
        }
    }
    func togglePlay() { isPlaying.toggle(); isPlaying ? player.play() : player.pause() }
    func pause() { isPlaying = false; player.pause() }
    func seek(_ t: Double) {
        player.seek(to: CMTime(seconds: max(0, t), preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] done in
            if done { DispatchQueue.main.async { self?.displayTime = t } }   // draw boxes only after the frame is on screen
        }
    }
    deinit { if let o = timeObs { player.removeTimeObserver(o) }; if let e = endObs { NotificationCenter.default.removeObserver(e) } }
}

final class PlayerContainer: NSView {
    let playerLayer = AVPlayerLayer()
    override func makeBackingLayer() -> CALayer { playerLayer.videoGravity = .resizeAspect; return playerLayer }
}
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer
    func makeNSView(context: Context) -> PlayerContainer { let v = PlayerContainer(); v.wantsLayer = true; v.playerLayer.player = player; return v }
    func updateNSView(_ v: PlayerContainer, context: Context) { v.playerLayer.player = player }
}

let overlayPalette: [Color] = [
    Color(red: 0.98, green: 0.26, blue: 0.30), Color(red: 0.20, green: 0.71, blue: 0.98), Color(red: 0.16, green: 0.85, blue: 0.52),
    Color(red: 0.99, green: 0.79, blue: 0.12), Color(red: 0.72, green: 0.40, blue: 0.98), Color(red: 0.99, green: 0.55, blue: 0.18),
    Color(red: 0.10, green: 0.83, blue: 0.80), Color(red: 0.98, green: 0.36, blue: 0.66), Color(red: 0.55, green: 0.82, blue: 0.28),
    Color(red: 0.40, green: 0.52, blue: 0.98)]

/// AVPlayer plays the raw video (hardware-decoded, real-time); a Canvas overlays boxes for the
/// current play-head time from cached candidates, so playback is smooth AND boxes stay live-tunable.
struct VideoStage: View {
    @ObservedObject var engine: InferenceEngine
    @ObservedObject var pc: PlayerController
    let conf: Double, iou: Double
    let overlay: SegOverlay
    let style: BoxStyle, label: LabelMode
    var body: some View {
        ZStack {
            PlayerView(player: pc.player)
            Canvas { ctx, size in
                let vid = engine.videoSize
                guard vid.width > 0, vid.height > 0 else { return }
                let scale = Swift.min(size.width / vid.width, size.height / vid.height)
                let dw = vid.width * scale, dh = vid.height * scale
                let ox = (size.width - dw) / 2, oy = (size.height - dh) / 2
                let lw = Swift.max(1.5, dw / 640 * 1.5)
                if let mask = engine.videoMaskImg {   // segmentation overlay, scaled to the video rect
                    ctx.draw(Image(decorative: mask, scale: 1), in: CGRect(x: ox, y: oy, width: dw, height: dh))
                }
                let masksOnly = engine.videoIsSegment && overlay == .masks   // hide boxes, keep labels/masks
                if masksOnly && label == .off { return }
                for d in engine.detsAt(time: pc.displayTime, conf: conf, iou: iou) {   // displayTime -> boxes match the shown frame
                    let color = overlayPalette[d.cls % overlayPalette.count]
                    let r = CGRect(x: ox + d.rect.minX * scale, y: oy + d.rect.minY * scale, width: d.rect.width * scale, height: d.rect.height * scale)
                    let rp = Path(roundedRect: r, cornerRadius: 3)
                    if !masksOnly {
                    switch style {
                    case .solid:
                        ctx.stroke(rp, with: .color(color), lineWidth: lw * 1.2)
                    case .neon:
                        var g = ctx
                        g.addFilter(.shadow(color: color, radius: lw * 4))
                        g.stroke(rp, with: .color(color), lineWidth: lw * 1.3)
                    case .hud:
                        ctx.fill(Path(r), with: .color(color.opacity(0.08)))
                        ctx.stroke(Path(r), with: .color(color.opacity(0.35)), lineWidth: lw * 0.6)
                        let arm = Swift.min(Swift.min(r.width, r.height) * 0.28, lw * 22)
                        var br = Path()
                        br.move(to: CGPoint(x: r.minX + arm, y: r.minY)); br.addLine(to: CGPoint(x: r.minX, y: r.minY)); br.addLine(to: CGPoint(x: r.minX, y: r.minY + arm))
                        br.move(to: CGPoint(x: r.maxX - arm, y: r.minY)); br.addLine(to: CGPoint(x: r.maxX, y: r.minY)); br.addLine(to: CGPoint(x: r.maxX, y: r.minY + arm))
                        br.move(to: CGPoint(x: r.minX, y: r.maxY - arm)); br.addLine(to: CGPoint(x: r.minX, y: r.maxY)); br.addLine(to: CGPoint(x: r.minX + arm, y: r.maxY))
                        br.move(to: CGPoint(x: r.maxX, y: r.maxY - arm)); br.addLine(to: CGPoint(x: r.maxX, y: r.maxY)); br.addLine(to: CGPoint(x: r.maxX - arm, y: r.maxY))
                        ctx.stroke(br, with: .color(color), lineWidth: lw * 1.4)
                    }
                    }   // if !masksOnly
                    if label != .off {
                        let name = d.cls < engine.names.count ? engine.names[d.cls] : "class\(d.cls)"
                        let txt = label == .min ? name : "\(name) \(String(format: "%.2f", d.score))"
                        let resolved = ctx.resolve(Text(txt).font(.system(size: Swift.max(9, dw / 95))).bold().foregroundColor(.white))
                        let ts = resolved.measure(in: size)
                        let chip = CGRect(x: r.minX, y: Swift.max(oy, r.minY - ts.height - 4), width: ts.width + 6, height: ts.height + 4)
                        ctx.fill(Path(roundedRect: chip, cornerRadius: 3), with: .color(color.opacity(0.85)))
                        ctx.draw(resolved, at: CGPoint(x: chip.minX + 3, y: chip.minY + 2), anchor: .topLeading)
                    }
                }
            }
            .allowsHitTesting(false)
        }
    }
}

// ---------- main UI ----------
struct ContentView: View {
    @StateObject private var engine = InferenceEngine()
    // Default to the model bundled in the app (Resources); user can still pick another. nil under `swift run`.
    @State private var modelURL: URL? = Bundle.main.url(forResource: "v0.1-seg-N", withExtension: "mlpackage")
    @State private var sourceURL: URL?
    @State private var conf = 0.25
    @State private var iou = 0.50
    @State private var style: BoxStyle = .hud
    @State private var label: LabelMode = .full
    @State private var overlay: SegOverlay = .both     // segmentation: masks / boxes / both
    @State private var preprocess: Detector.PreprocessMode = .letterbox   // input fit: letterbox vs force-resize to imgsz
    @State private var compute: ComputeMode = .cpuAndGPU
    @State private var showPicker = false
    @State private var pickTarget: PickTarget = .model
    @State private var folderImages: [URL] = []
    @State private var selectedIndex = 0
    @State private var finderMode: FinderMode = .icons
    @State private var iconSize: Double = 108
    @State private var videoDur = 0.0
    @State private var scrubTime = 0.0
    @State private var scrubbing = false
    @State private var wasPlaying = false
    @StateObject private var pc = PlayerController()
    @State private var cameraOn = false   // live-camera mode; the session lives in LiveCameraView (isolated observation)
    @State private var cameraIsSegment = false   // set by LiveCameraView once its detector is built
    @FocusState private var kbFocused: Bool

    private enum PickTarget { case model, source }
    private var sourceKind: SourceKind { sourceURL.map(classifySource) ?? .unknown }
    private var modelInfoImgsz: String { engine.modelInfo.map { "\($0.imgsz)×\($0.imgsz)" } ?? "the model's imgsz" }
    private var isSegModel: Bool { cameraOn ? cameraIsSegment : engine.modelIsSegment }   // drives the Overlay control in both modes
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
        .fileImporter(isPresented: $showPicker, allowedContentTypes: pickerTypes) { if case .success(let u) = $0 { assign(u) } }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers { _ = p.loadObject(ofClass: URL.self) { url, _ in guard let url else { return }; DispatchQueue.main.async { assign(url) } } }
            return true
        }
        .onChange(of: conf) { rerender() }
        .onChange(of: iou) { rerender() }
        .onChange(of: style) { rerender() }
        .onChange(of: label) { rerender() }
        .onChange(of: overlay) { rerender() }
        .onChange(of: preprocess) {   // preprocessing changes the forward pass -> re-infer (not a cheap re-render)
            if cameraOn { return }    // LiveCameraView hot-swaps the detector itself
            guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
            runInfer()
        }
        .onChange(of: modelURL) { setupSource() }
        .onChange(of: sourceURL) { setupSource() }
        .onChange(of: scrubTime) { if scrubbing { pc.seek(scrubTime) } }   // seek while dragging
        .onChange(of: pc.currentTime) { if pc.isPlaying && !scrubbing { scrubTime = pc.currentTime } }   // slider follows playback
        .onChange(of: pc.displayTime) {
            if sourceKind == .video && engine.hasResults {
                engine.setVideoFrameStats(time: pc.displayTime, conf: conf, iou: iou)
                engine.updateVideoMask(time: pc.displayTime, conf: conf, iou: iou, overlay: overlay)
            }
        }
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
        pc.pause()
        engine.resetResults()
        guard let s = sourceURL else { return }
        switch classifySource(s) {
        case .folder: folderImages = listImages(s); selectedIndex = 0
        case .video:
            pc.load(s)
            Task { let dur = await videoDuration(s); await MainActor.run { videoDur = dur; scrubTime = 0 } }
        default: break
        }
    }
    private func runInfer() {
        guard let m = modelURL, let s = sourceURL else { return }
        switch sourceKind {
        case .image:  engine.previewURL(model: m, image: s, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess)
        case .folder: engine.runFolder(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess)
        case .video:  engine.runVideo(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label, preprocess: preprocess, overlay: overlay)
        default: break
        }
    }
    // ---- live camera (session lifecycle + detector build handled inside LiveCameraView) ----
    private func startCamera() { guard modelURL != nil, !engine.busy else { return }; pc.pause(); cameraOn = true }
    private func stopCamera() { cameraOn = false }

    private func selectAndShow(_ i: Int) {
        guard folderImages.indices.contains(i) else { return }
        selectedIndex = i
        engine.showFolder(index: i, url: folderImages[i], conf: conf, iou: iou, style: style, label: label, overlay: overlay)
    }
    private func rerender() {
        if cameraOn { return }   // camera overlay reads conf/iou/style/label live — no engine re-render
        if sourceKind == .video {
            if engine.hasResults {
                engine.setVideoFrameStats(time: pc.displayTime, conf: conf, iou: iou)   // overlay redraws on conf/iou/label automatically
                engine.updateVideoMask(time: pc.displayTime, conf: conf, iou: iou, overlay: overlay)
            }
        } else {
            engine.restyle(conf: conf, iou: iou, style: style, label: label, overlay: overlay)
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
                Image(systemName: "viewfinder.circle.fill").font(.system(size: 27)).foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 0) {
                    Text("YOLO-Master").font(.headline)
                    Text("Core ML runner").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
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
                    sectionBox("Detection", "slider.horizontal.3") {
                        sliderRow("Confidence", $conf, 0.05...0.95)
                        sliderRow("IoU (NMS)", $iou, 0.10...0.90)
                    }
                    sectionBox("Appearance", "paintbrush.fill") {
                        if isSegModel {
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

            actionRow
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
                primaryButton("Run", "play.fill") { runInfer() }.disabled(sourceURL == nil || engine.busy)
                secondaryButton("Save…", "square.and.arrow.down") { engine.save() }.disabled(engine.resultImage == nil)
            case .folder:
                primaryButton(engine.hasResults ? "Re-run inference" : "Run inference", "play.fill") { runInfer() }
                    .disabled(sourceURL == nil || engine.busy)
                HStack(spacing: 8) {
                    secondaryButton("Export", "square.and.arrow.up") { engine.exportFolder(conf: conf, iou: iou, style: style, label: label, overlay: overlay) }
                        .disabled(!engine.hasResults || engine.busy)
                    if engine.outputURL != nil { revealButton }
                }
            case .video:
                primaryButton(engine.hasResults ? "Re-run inference" : "Run inference", "play.fill") { runInfer() }
                    .disabled(sourceURL == nil || engine.busy)
                HStack(spacing: 8) {
                    secondaryButton("Export", "square.and.arrow.up") { engine.exportVideo(conf: conf, iou: iou, style: style, label: label, overlay: overlay) }
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
                               conf: conf, iou: iou, overlay: overlay, style: style, label: label,
                               isSegment: $cameraIsSegment).padding(12)
            } else if sourceKind == .video && engine.hasResults {
                VideoStage(engine: engine, pc: pc, conf: conf, iou: iou, overlay: overlay, style: style, label: label).padding(12)
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
        GroupBox {
            VStack(alignment: .leading, spacing: 12) { content() }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 2)
        } label: {
            Label(title, systemImage: icon).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
        }
    }
    private func fileRow(icon: String, title: String, value: String, set: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 15)).foregroundStyle(set ? Color.accentColor : .secondary).frame(width: 20)
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
