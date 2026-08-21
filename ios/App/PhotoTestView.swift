// Camera-free detection on library photos. EXIF orientation is applied at load
// (CGImageSourceCreateThumbnailAtIndex + transform) - portrait photos were
// previously processed sideways, which scrambled every box.
import CoreML
import PhotosUI
import SwiftUI
import YOLOMasterKit

struct PhotoTestView: View {
    @State private var models = BundledModel.discover()
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = .ane
    @State private var pick: PhotosPickerItem?
    @State private var image: CGImage?
    @State private var dets: [Detection] = []
    @State private var status = "pick a photo"
    @State private var conf = 0.25
    @State private var iou = 0.5
    @State private var showTuning = false
    @State private var statInf: Double = 0
    @State private var statDec: Double = 0
    @State private var statPre: Double = 0
    // cached forward pass: sliders re-decode WITHOUT re-running the model
    @State private var lastDet: Detector?
    @State private var lastRaw: Detector.RawOutput?

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                if let img = image {
                    GeometryReader { geo in
                        let scale = min(geo.size.width / CGFloat(img.width),
                                        geo.size.height / CGFloat(img.height))
                        let w = CGFloat(img.width) * scale
                        let h = CGFloat(img.height) * scale
                        ZStack(alignment: .topLeading) {
                            Image(decorative: img, scale: 1).resizable()
                                .frame(width: w, height: h)
                            Canvas { ctx, _ in
                                DetOverlay.draw(ctx, dets, scale: scale, ox: 0, oy: 0)
                            }
                            .frame(width: w, height: h)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .padding(.horizontal, 8)
                } else {
                    ContentUnavailableView("No photo", systemImage: "photo")
                        .frame(maxHeight: .infinity)
                }
                if showTuning {
                    TuningPanel(conf: $conf, iou: $iou).padding(.horizontal, 8)
                }
                HStack {
                    StatsHUD(fps: statInf > 0 ? 1000.0 / (statPre + statInf + statDec) : 0,
                             pre: statPre, inf: statInf, dec: statDec,
                             dets: dets.count, fpsLabel: "eqFPS",
                             active: statInf > 0)
                }
                if status.hasPrefix("ERROR") || status.hasPrefix("PROBE") || status.contains("'") {
                    Text(status)
                        .font(.caption2.monospaced())
                        .lineLimit(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 12)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .safeAreaInset(edge: .top) {
                HStack {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(models) { m in Text(m.shortID).lineLimit(1).fixedSize().tag(Optional(m)) }
                    }
                    Picker("Compute", selection: $compute) {
                        ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).tag(c) }
                    }
                    Button {
                        withAnimation { showTuning.toggle() }
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                    .buttonStyle(.bordered)
                    PhotosPicker(selection: $pick, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderedProminent)
                    Button("Probe") { probe() }
                        .buttonStyle(.bordered)
                        .disabled(image == nil)
                }
                .padding(8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)
            }
            .onAppear { if selectedModel == nil { selectedModel = models.first } }
            .onChange(of: pick) { _, item in load(item) }
            .onChange(of: selectedModel) { _, _ in run() }
            .onChange(of: compute) { _, _ in run() }
            .onChange(of: conf) { _, _ in retune() }
            .onChange(of: iou) { _, _ in retune() }
        }
    }

    private func load(_ item: PhotosPickerItem?) {
        guard let item else { return }
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let src = CGImageSourceCreateWithData(data as CFData, nil),
                  let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                      kCGImageSourceCreateThumbnailWithTransform: true,     // EXIF orientation
                      kCGImageSourceCreateThumbnailFromImageAlways: true,
                      kCGImageSourceThumbnailMaxPixelSize: 2048,
                  ] as CFDictionary) else {
                await MainActor.run { status = "photo load failed" }; return
            }
            await MainActor.run { image = img; dets = [] }
            run()
        }
    }

    private func run() {
        guard let img = image, let m = selectedModel else { return }
        let mode = compute.mode
        let confNow = Float(conf), iouNow = CGFloat(iou)
        status = "running..."
        Task.detached {
            do {
                let det = try Detector(modelURL: m.url, compute: mode)
                let t0 = CFAbsoluteTimeGetCurrent()
                let raw = try det.forward(img)
                let t1 = CFAbsoluteTimeGetCurrent()
                let d = det.decode(raw, conf: confNow, iou: iouNow)
                let t2 = CFAbsoluteTimeGetCurrent()
                let fwd = (t1 - t0) * 1000
                await MainActor.run {
                    dets = d
                    lastDet = det; lastRaw = raw
                    statPre = fwd - raw.inferMs
                    statInf = raw.inferMs
                    statDec = (t2 - t1) * 1000
                    status = "ok"
                }
            } catch {
                await MainActor.run { status = "ERROR: \(error)" }
            }
        }
    }

    /// Slider retune: decode-only on the cached forward pass (instant, no model run).
    private func retune() {
        guard let det = lastDet, let raw = lastRaw else { return }
        let t1 = CFAbsoluteTimeGetCurrent()
        let d = det.decode(raw, conf: Float(conf), iou: CGFloat(iou))
        statDec = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        dets = d
    }

    /// Kit-free raw-output probe (diagnostic; kept from the debugging session).
    private func probe() {
        guard let img = image, let m = selectedModel else { return }
        let mode = compute.mode
        status = "probing..."
        Task.detached {
            do {
                let cfg = MLModelConfiguration(); cfg.computeUnits = mode.mlUnits
                let model = try MLModel(contentsOf: m.url, configuration: cfg)
                let inName = model.modelDescription.inputDescriptionsByName.keys.first ?? "images"
                let sz = 640
                var px = [UInt8](repeating: 0, count: sz * sz * 4)
                px.withUnsafeMutableBytes { rawBuf in
                    if let ctx = CGContext(data: rawBuf.baseAddress, width: sz, height: sz,
                                           bitsPerComponent: 8, bytesPerRow: sz * 4,
                                           space: CGColorSpaceCreateDeviceRGB(),
                                           bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) {
                        ctx.draw(img, in: CGRect(x: 0, y: 0, width: sz, height: sz))
                    }
                }
                let arr = try MLMultiArray(shape: [1, 3, 640, 640], dataType: .float32)
                let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
                let plane = sz * sz
                for i in 0..<plane {
                    p[i] = Float32(px[i * 4]) / 255
                    p[plane + i] = Float32(px[i * 4 + 1]) / 255
                    p[2 * plane + i] = Float32(px[i * 4 + 2]) / 255
                }
                let out = try model.prediction(
                    from: MLDictionaryFeatureProvider(dictionary: [inName: MLFeatureValue(multiArray: arr)]))
                var lines: [String] = []
                for name in out.featureNames {
                    guard let a = out.featureValue(for: name)?.multiArrayValue else { continue }
                    var mn = Float.greatestFiniteMagnitude, mx = -Float.greatestFiniteMagnitude
                    var nan = 0
                    let n = min(a.count, 200_000)
                    for i in 0..<n {
                        let v = a[i].floatValue
                        if v.isNaN { nan += 1; continue }
                        mn = min(mn, v); mx = max(mx, v)
                    }
                    lines.append("'\(name)' \(a.shape) min " + String(format: "%.3f", mn) +
                                 " max " + String(format: "%.3f", mx) + " nan \(nan)")
                }
                let msg = lines.joined(separator: "\n")
                await MainActor.run { status = msg }
            } catch {
                await MainActor.run { status = "PROBE ERROR: \(error)" }
            }
        }
    }
}
