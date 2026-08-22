// Batch photo detection: up to 100 images, gallery (3-up grid) or pager view,
// macOS-style phase progress (Loading/iCloud -> Inference), per-image stage
// averages + img/s dial. Slider retunes re-decode cached passes - no re-runs.
import CoreML
import Photos
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
                                hudVisible: $showHUD).padding(.horizontal, 8)
                }
                if showHUD { hud.padding(.bottom, 20) }
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
            .padding(.horizontal, 8)
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
                    ZoomableView(onPinchExit: {
                        withAnimation(.spring(duration: 0.35)) { viewMode = .gallery }
                    }) {
                        annotated(images[i], results.indices.contains(i) ? results[i] : [])
                    }
                    .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .automatic))
            .transition(.scale(scale: 1.08).combined(with: .opacity))
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
        .padding(.horizontal, 8)
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
            PhotosPicker(selection: $picks, maxSelectionCount: 100, matching: .images,
                         photoLibrary: .shared()) {
                Image(systemName: "photo.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 8)
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
            // album-style names (IMG_xxxx) come from PHAssetResource via the
            // picker's itemIdentifier (photoLibrary: .shared() enables it)
            if PHPhotoLibrary.authorizationStatus(for: .readWrite) == .notDetermined {
                _ = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            }
            for (idx, item) in items.enumerated() {
                var name = "IMG_\(idx + 1)"
                var type = item.supportedContentTypes.first?
                    .preferredFilenameExtension?.uppercased() ?? "-"
                if let id = item.itemIdentifier {
                    let assets = PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil)
                    if let asset = assets.firstObject,
                       let res = PHAssetResource.assetResources(for: asset).first {
                        let full = res.originalFilename          // e.g. IMG_4231.HEIC
                        name = (full as NSString).deletingPathExtension
                        let ext = (full as NSString).pathExtension
                        if !ext.isEmpty { type = ext.uppercased() }
                    }
                }
                let data = try? await item.loadTransferable(type: Data.self)
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
                // cache raw passes for slider retune only on modest batches:
                // 100 tensors x ~3MB is a jetsam invitation
                let cacheRaws = imgs.count <= 40
                for (idx, img) in imgs.enumerated() {
                    let step: (Detector.RawOutput, [Detection], Double, Double, Double) =
                        try autoreleasepool {
                            let a = CFAbsoluteTimeGetCurrent()
                            let raw = try det.forward(img)
                            let b = CFAbsoluteTimeGetCurrent()
                            let d = det.decode(raw, conf: confNow, iou: iouNow)
                            let c = CFAbsoluteTimeGetCurrent()
                            return (raw, d, (b - a) * 1000 - raw.inferMs,
                                    raw.inferMs, (c - b) * 1000)
                        }
                    let (raw, d, pPreV, pInfV, pDecV) = step
                    sumPre += pPreV
                    sumInf += pInfV
                    sumDec += pDecV
                    if cacheRaws { raws.append(raw) }
                    allDets.append(d)
                    let done = idx + 1
                    let dSnapshot = d
                    await MainActor.run {
                        progress = done
                        if results.indices.contains(idx) { results[idx] = dSnapshot }
                        perPre.append(pPreV); perInf.append(pInfV); perDec.append(pDecV)
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



/// Pinch-zoomable page: zoom 1-5x with pan when zoomed, double-tap toggles
/// 1x/2.5x, and pinching IN below 1x (with live shrink feedback) exits to the
/// gallery - far more responsive than a hard end-of-gesture threshold.
struct ZoomableView<Content: View>: View {
    let onPinchExit: () -> Void
    @ViewBuilder let content: Content
    @State private var zoom: CGFloat = 1
    @State private var baseZoom: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var baseOffset: CGSize = .zero

    var body: some View {
        content
            .scaleEffect(zoom)
            .offset(offset)
            .gesture(
                MagnifyGesture()
                    .onChanged { v in
                        // below-1x rubber-band (the exit affordance) exists ONLY
                        // when the gesture started at minimum zoom; zoomed-in
                        // pinches bottom out at 1x
                        let lower: CGFloat = baseZoom <= 1.01 ? 0.6 : 1
                        zoom = min(max(baseZoom * v.magnification, lower), 5)
                    }
                    .onEnded { v in
                        let target = baseZoom * v.magnification
                        if baseZoom <= 1.01 && target < 0.95 {   // sensitive exit
                            reset()
                            onPinchExit()
                            return
                        }
                        zoom = min(max(target, 1), 5)
                        baseZoom = zoom
                        if zoom <= 1.01 { withAnimation(.spring(duration: 0.25)) { offset = .zero }; baseOffset = .zero }
                    }
            )
            .simultaneousGesture(
                DragGesture()
                    .onChanged { g in
                        guard baseZoom > 1.01 else { return }
                        offset = CGSize(width: baseOffset.width + g.translation.width,
                                        height: baseOffset.height + g.translation.height)
                    }
                    .onEnded { _ in baseOffset = offset }
            )
            .onTapGesture(count: 2) {
                withAnimation(.spring(duration: 0.3)) {
                    if zoom > 1.5 { reset() } else { zoom = 2.5; baseZoom = 2.5 }
                }
            }
            .animation(.easeOut(duration: 0.1), value: zoom)
            .onDisappear { reset() }
    }

    private func reset() {
        zoom = 1; baseZoom = 1; offset = .zero; baseOffset = .zero
    }
}
