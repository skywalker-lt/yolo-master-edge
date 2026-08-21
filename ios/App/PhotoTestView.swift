// Batch photo detection: pick up to 24 images, swipe through annotated results.
// Stats are PER-IMAGE stage averages plus an img/s throughput dial. Slider
// retunes re-decode the cached forward passes - zero model re-runs.
import CoreML
import PhotosUI
import SwiftUI
import YOLOMasterKit

struct PhotoTestView: View {
    @State private var models = BundledModel.discover()
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = .ane
    @State private var style: BoxStyle = .chip
    @State private var picks: [PhotosPickerItem] = []
    @State private var images: [CGImage] = []
    @State private var results: [[Detection]] = []
    @State private var page = 0
    @State private var conf = 0.25
    @State private var iou = 0.5
    @State private var showTuning = false
    @State private var statPre: Double = 0
    @State private var statInf: Double = 0
    @State private var statDec: Double = 0
    @State private var throughput: Double = 0     // img/s over the whole batch
    @State private var status = ""
    // cached forward passes for instant slider retunes
    @State private var lastDet: Detector?
    @State private var lastRaws: [Detector.RawOutput] = []

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                if images.isEmpty {
                    ContentUnavailableView("No photos", systemImage: "photo.on.rectangle.angled",
                                           description: Text("Pick up to 24 images"))
                        .frame(maxHeight: .infinity)
                } else {
                    TabView(selection: $page) {
                        ForEach(images.indices, id: \.self) { i in
                            annotated(images[i], results.indices.contains(i) ? results[i] : [])
                                .tag(i)
                        }
                    }
                    .tabViewStyle(.page(indexDisplayMode: .automatic))
                }
                if showTuning {
                    TuningPanel(conf: $conf, iou: $iou).padding(.horizontal, 8)
                }
                StatsHUD(fps: throughput, pre: statPre, inf: statInf, dec: statDec,
                         dets: results.indices.contains(page) ? results[page].count : 0,
                         fpsLabel: "img/s", active: statInf > 0)
                if !status.isEmpty {
                    Text(status)
                        .font(.caption2.monospaced())
                        .lineLimit(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 12)
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

    private var controls: some View {
        HStack {
            Picker("Model", selection: $selectedModel) {
                ForEach(models) { m in
                    Text(m.shortID).lineLimit(1).fixedSize().tag(Optional(m))
                }
            }
            .fixedSize()
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.allCases) { c in
                    Text(c.rawValue).lineLimit(1).fixedSize().tag(c)
                }
            }
            .fixedSize()
            Picker("Style", selection: $style) {
                ForEach(BoxStyle.allCases) { s in
                    Text(s.label).lineLimit(1).fixedSize().tag(s)
                }
            }
            .fixedSize()
            Spacer(minLength: 0)
            Button {
                withAnimation { showTuning.toggle() }
            } label: {
                Image(systemName: "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            PhotosPicker(selection: $picks, maxSelectionCount: 24, matching: .images) {
                Image(systemName: "photo.badge.plus")
            }
            .buttonStyle(.borderedProminent)
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

    private func load(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        status = "loading \(items.count) photos..."
        Task {
            var loaded: [CGImage] = []
            for item in items {
                guard let data = try? await item.loadTransferable(type: Data.self),
                      let src = CGImageSourceCreateWithData(data as CFData, nil),
                      let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                          kCGImageSourceCreateThumbnailWithTransform: true,   // EXIF
                          kCGImageSourceCreateThumbnailFromImageAlways: true,
                          kCGImageSourceThumbnailMaxPixelSize: 2048,
                      ] as CFDictionary) else { continue }
                loaded.append(img)
            }
            let imgs = loaded
            await MainActor.run {
                images = imgs
                results = Array(repeating: [], count: imgs.count)
                page = 0
                status = imgs.isEmpty ? "no loadable photos" : ""
            }
            run()
        }
    }

    private func run() {
        guard !images.isEmpty, let m = selectedModel else { return }
        let mode = compute.mode
        let confNow = Float(conf), iouNow = CGFloat(iou)
        let imgs = images
        status = "running \(imgs.count) images..."
        Task.detached {
            do {
                let det = try Detector(modelURL: m.url, compute: mode)
                var raws: [Detector.RawOutput] = []
                var allDets: [[Detection]] = []
                var sumPre = 0.0, sumInf = 0.0, sumDec = 0.0
                let t0 = CFAbsoluteTimeGetCurrent()
                for img in imgs {
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
                }
                let wall = CFAbsoluteTimeGetCurrent() - t0
                let n = Double(imgs.count)
                let dets = allDets, rawsF = raws
                await MainActor.run {
                    results = dets
                    lastDet = det
                    lastRaws = rawsF
                    statPre = sumPre / n
                    statInf = sumInf / n
                    statDec = sumDec / n
                    throughput = n / max(wall, 0.001)
                    status = ""
                }
            } catch {
                await MainActor.run { status = "ERROR: \(error)" }
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
