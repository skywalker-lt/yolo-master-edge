// Batch photo detection: up to 100 images, gallery (3-up grid) or pager view,
// macOS-style phase progress (Loading/iCloud -> Inference), per-image stage
// averages + img/s dial. Slider retunes re-decode cached passes - no re-runs.
import CoreML
import PhotosUI
import SwiftUI
import YOLOMasterKit

struct PhotoTestView: View {
    enum ViewMode { case gallery, pager }
    enum Phase: Equatable { case idle, loading, inferring }

    @State private var models = BundledModel.discover()
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = .ane
    @State private var style: BoxStyle = .chip
    @State private var picks: [PhotosPickerItem] = []
    @State private var images: [CGImage] = []
    @State private var fileNames: [String] = []
    @State private var fileTypes: [String] = []
    @State private var results: [[Detection]] = []
    @State private var page = 0
    @State private var viewMode: ViewMode = .gallery
    @State private var conf = 0.25
    @State private var iou = 0.5
    @State private var showTuning = false
    @State private var showHUD = true
    @State private var isRunning = false
    @State private var phase: Phase = .idle
    @State private var progress = 0
    @State private var progressTotal = 0
    @State private var statPre: Double = 0
    @State private var statInf: Double = 0
    @State private var statDec: Double = 0
    @State private var throughput: Double = 0
    @State private var wallSeconds: Double = 0
    @State private var perPre: [Double] = []
    @State private var perInf: [Double] = []
    @State private var perDec: [Double] = []
    @State private var errorText = ""
    @State private var lastDet: Detector?
    @State private var lastRaws: [Detector.RawOutput] = []

    private let gridCols = [GridItem(.flexible(), spacing: 4),
                            GridItem(.flexible(), spacing: 4),
                            GridItem(.flexible(), spacing: 4)]

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                content
                if phase != .idle { progressBar }
                if showTuning {
                    TuningPanel(conf: $conf, iou: $iou, style: $style,
                                hudVisible: $showHUD).padding(.horizontal)
                }
                if showHUD { hud }
                if !errorText.isEmpty {
                    Text(errorText).font(.caption2).foregroundStyle(.red)
                        .lineLimit(3).padding(.horizontal, 12)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .safeAreaInset(edge: .top) { controls }
            .onAppear { if selectedModel == nil { selectedModel = models.first } }
            .onChange(of: picks) { _, items in load(items) }
            .onChange(of: selectedModel) { _, _ in run() }
            .onChange(of: compute) { _, _ in run() }
            .onChange(of: conf) { _, _ in retune() }
            .onChange(of: iou) { _, _ in retune() }
        }
    }

    /// Pager: THIS image's stage times + its own capability FPS.
    /// Gallery: batch averages + overall FPS + detail rows.
    private var hud: some View {
        let perImage = viewMode == .pager && perInf.indices.contains(page)
        let p = perImage ? perPre[page] : statPre
        let i = perImage ? perInf[page] : statInf
        let d = perImage ? perDec[page] : statDec
        let e2e = p + i + d
        var extras: [(String, String)] = []
        if statInf > 0 {
            if perImage {
                let img = images[page]
                extras = [("image", "\(page + 1)/\(images.count)"),
                          ("size", "\(img.width)x\(img.height)"),
                          ("file", fileNames.indices.contains(page) ? fileNames[page] : "-"),
                          ("type", fileTypes.indices.contains(page) ? fileTypes[page] : "-"),
                          ("e2e", String(format: "%.1f ms", e2e)),
                          ("dets", "\(results.indices.contains(page) ? results[page].count : 0)")]
            } else {
                let totalDets = results.reduce(0) { $0 + $1.count }
                extras = [("images", "\(images.count)"),
                          ("wall", String(format: "%.2f s", wallSeconds)),
                          ("total dets", "\(totalDets)"),
                          ("avg e2e", String(format: "%.1f ms", statPre + statInf + statDec))]
            }
        }
        // pager dial = this image's END-TO-END ms (0-30 scale, band colors);
        // gallery dial = batch FPS
        return StatsHUD(fps: perImage ? e2e : throughput, pre: p, inf: i, dec: d,
                        dets: results.indices.contains(page) ? results[page].count : 0,
                        fpsLabel: perImage ? "ms" : "FPS", active: statInf > 0,
                        extras: extras, mode: perImage ? .ms : .fps, fullWidth: true)
            .frame(maxWidth: .infinity)
            .padding(.horizontal)
    }

    // MARK: pieces

    @ViewBuilder private var content: some View {
        if images.isEmpty {
            ContentUnavailableView("No photos", systemImage: "photo.on.rectangle.angled",
                                   description: Text("Pick up to 100 images"))
                .frame(maxHeight: .infinity)
        } else if viewMode == .gallery {
            ScrollView {
                LazyVGrid(columns: gridCols, spacing: 4) {
                    ForEach(images.indices, id: \.self) { i in
                        galleryCell(i)
                    }
                }
                .padding(.horizontal, 8)
            }
            .transition(.scale(scale: 0.92).combined(with: .opacity))
        } else {
            TabView(selection: $page) {
                ForEach(images.indices, id: \.self) { i in
                    annotated(images[i], results.indices.contains(i) ? results[i] : [])
                        .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .automatic))
            .transition(.scale(scale: 1.08).combined(with: .opacity))
            .gesture(
                MagnifyGesture().onEnded { v in
                    if v.magnification < 0.85 {          // pinch-out -> gallery
                        withAnimation(.spring(duration: 0.35)) { viewMode = .gallery }
                    }
                }
            )
        }
    }

    private func galleryCell(_ i: Int) -> some View {
        GeometryReader { geo in
            let img = images[i]
            let scale = geo.size.width / CGFloat(img.width)
            ZStack(alignment: .topLeading) {
                Image(decorative: img, scale: 1).resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geo.size.width, height: geo.size.width)
                    .clipped()
                Canvas { ctx, _ in
                    if results.indices.contains(i) {
                        DetOverlay.draw(ctx, results[i], scale: scale, ox: 0, oy: 0,
                                        style: style)
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
            .onTapGesture {
                page = i
                withAnimation(.spring(duration: 0.35)) { viewMode = .pager }
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private var progressBar: some View {
        HStack(spacing: 10) {
            Text(phase == .loading ? "Loading / iCloud" : "Inference")
                .font(.caption)
            ProgressView(value: Double(progress), total: Double(max(progressTotal, 1)))
                .progressViewStyle(.linear)
            Text("\(progress)/\(progressTotal)")
                .font(.caption.monospacedDigit())
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private var controls: some View {
        HStack {
            Picker("Model", selection: $selectedModel) {
                ForEach(models) { m in
                    Text(m.shortID).lineLimit(1).fixedSize().tag(Optional(m))
                }
            }
            .fixedSize()
            .disabled(isRunning)
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.allCases) { c in
                    Text(c.rawValue).lineLimit(1).fixedSize().tag(c)
                }
            }
            .fixedSize()
            .disabled(isRunning)
            Spacer(minLength: 0)
            if !images.isEmpty {
                Button {
                    withAnimation { viewMode = viewMode == .gallery ? .pager : .gallery }
                } label: {
                    Image(systemName: viewMode == .gallery
                          ? "rectangle.portrait" : "square.grid.3x3")
                }
                .buttonStyle(.bordered)
            }
            Button {
                withAnimation { showTuning.toggle() }
            } label: {
                Image(systemName: "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            PhotosPicker(selection: $picks, maxSelectionCount: 100, matching: .images) {
                Image(systemName: "photo.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func annotated(_ img: CGImage, _ dets: [Detection]) -> some View {
        GeometryReader { geo in
            let scale = min(geo.size.width / CGFloat(img.width),
                            geo.size.height / CGFloat(img.height))
            let w = CGFloat(img.width) * scale
            let h = CGFloat(img.height) * scale
            ZStack(alignment: .topLeading) {
                Image(decorative: img, scale: 1).resizable()
                    .frame(width: w, height: h)
                Canvas { ctx, _ in
                    DetOverlay.draw(ctx, dets, scale: scale, ox: 0, oy: 0, style: style)
                }
                .frame(width: w, height: h)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 8)
    }

    // MARK: pipeline

    private func load(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        phase = .loading
        progress = 0
        progressTotal = items.count
        errorText = ""
        Task {
            var loaded: [CGImage] = []
            var names: [String] = []
            var types: [String] = []
            for (idx, item) in items.enumerated() {
                var name = "IMG_\(idx + 1)"
                var type = item.supportedContentTypes.first?
                    .preferredFilenameExtension?.uppercased() ?? "-"
                var data: Data?
                if let picked = try? await item.loadTransferable(type: PickedFile.self) {
                    name = picked.url.lastPathComponent
                    if !picked.url.pathExtension.isEmpty {
                        type = picked.url.pathExtension.uppercased()
                    }
                    data = try? Data(contentsOf: picked.url)
                    try? FileManager.default.removeItem(at: picked.url)
                }
                if data == nil { data = try? await item.loadTransferable(type: Data.self) }
                if let data,
                   let src = CGImageSourceCreateWithData(data as CFData, nil),
                   let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                       kCGImageSourceCreateThumbnailWithTransform: true,   // EXIF
                       kCGImageSourceCreateThumbnailFromImageAlways: true,
                       kCGImageSourceThumbnailMaxPixelSize: 2048,
                   ] as CFDictionary) {
                    loaded.append(img)
                    names.append(name)
                    types.append(type)
                }
                let done = idx + 1
                await MainActor.run { progress = done }
            }
            let imgs = loaded, ns = names, ts = types
            await MainActor.run {
                images = imgs
                fileNames = ns
                fileTypes = ts
                results = Array(repeating: [], count: imgs.count)
                page = 0
                viewMode = imgs.count > 1 ? .gallery : .pager
                phase = .idle
                if imgs.isEmpty { errorText = "no loadable photos" }
            }
            run()
        }
    }

    private func run() {
        guard !images.isEmpty, let m = selectedModel else { return }
        let mode = compute.mode
        let confNow = Float(conf), iouNow = CGFloat(iou)
        let imgs = images
        isRunning = true
        phase = .inferring
        progress = 0
        progressTotal = imgs.count
        errorText = ""
        perPre = []; perInf = []; perDec = []
        Task.detached {
            do {
                let det = try Detector(modelURL: m.url, compute: mode)
                var raws: [Detector.RawOutput] = []
                var allDets: [[Detection]] = []
                var sumPre = 0.0, sumInf = 0.0, sumDec = 0.0
                let t0 = CFAbsoluteTimeGetCurrent()
                for (idx, img) in imgs.enumerated() {
                    let a = CFAbsoluteTimeGetCurrent()
                    let raw = try det.forward(img)
                    let b = CFAbsoluteTimeGetCurrent()
                    let d = det.decode(raw, conf: confNow, iou: iouNow)
                    let c = CFAbsoluteTimeGetCurrent()
                    sumPre += (b - a) * 1000 - raw.inferMs
                    sumInf += raw.inferMs
                    sumDec += (c - b) * 1000
                    raws.append(raw)
                    allDets.append(d)
                    let done = idx + 1
                    let dSnapshot = d
                    let pPre = (b - a) * 1000 - raw.inferMs
                    let pInf = raw.inferMs
                    let pDec = (c - b) * 1000
                    await MainActor.run {
                        progress = done
                        if results.indices.contains(idx) { results[idx] = dSnapshot }
                        perPre.append(pPre); perInf.append(pInf); perDec.append(pDec)
                    }
                }
                let wall = CFAbsoluteTimeGetCurrent() - t0
                let n = Double(imgs.count)
                let rawsF = raws, detsF = allDets
                await MainActor.run {
                    results = detsF
                    lastDet = det
                    lastRaws = rawsF
                    statPre = sumPre / n
                    statInf = sumInf / n
                    statDec = sumDec / n
                    throughput = n / max(wall, 0.001)
                    wallSeconds = wall
                    phase = .idle
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    errorText = "ERROR: \(error)"
                    phase = .idle
                    isRunning = false
                }
            }
        }
    }

    /// Slider retune: decode-only over the cached forward passes.
    private func retune() {
        guard let det = lastDet, !lastRaws.isEmpty else { return }
        let confNow = Float(conf), iouNow = CGFloat(iou)
        results = lastRaws.map { det.decode($0, conf: confNow, iou: iouNow) }
    }
}


/// File-representation transfer that preserves the original photo filename.
struct PickedFile: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .image) { received in
            let dst = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "_" + received.file.lastPathComponent)
            try FileManager.default.copyItem(at: received.file, to: dst)
            return Self(url: dst)
        }
    }
}
