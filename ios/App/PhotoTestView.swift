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
                                for d in dets {
                                    let r = CGRect(x: d.rect.minX * scale,
                                                   y: d.rect.minY * scale,
                                                   width: d.rect.width * scale,
                                                   height: d.rect.height * scale)
                                    ctx.stroke(Path(roundedRect: r, cornerRadius: 2),
                                               with: .color(.green), lineWidth: 2)
                                    let name = d.cls < cocoNames.count ? cocoNames[d.cls] : "\(d.cls)"
                                    ctx.draw(Text("\(name) \(Int(d.score * 100))%")
                                        .font(.caption2.bold()).foregroundStyle(.green),
                                             at: CGPoint(x: r.minX + 2, y: max(r.minY - 9, 4)),
                                             anchor: .leading)
                                }
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
                Text(status)
                    .font(.caption.monospacedDigit())
                    .lineLimit(4)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
            }
            .navigationTitle("Photo test")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItemGroup(placement: .topBarLeading) {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(models) { m in Text(m.id).tag(Optional(m)) }
                    }
                    Picker("Unit", selection: $compute) {
                        ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).tag(c) }
                    }
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button("Probe") { probe() }.disabled(image == nil)
                    PhotosPicker(selection: $pick, matching: .images) {
                        Image(systemName: "photo.badge.plus")
                    }
                }
            }
            .onAppear { if selectedModel == nil { selectedModel = models.first } }
            .onChange(of: pick) { _, item in load(item) }
            .onChange(of: selectedModel) { _, _ in run() }
            .onChange(of: compute) { _, _ in run() }
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
        status = "running..."
        Task.detached {
            do {
                let det = try Detector(modelURL: m.url, compute: mode)
                let t0 = CFAbsoluteTimeGetCurrent()
                let raw = try det.forward(img)
                let d = det.decode(raw, conf: 0.25, iou: 0.5)
                let total = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                let msg = "dets \(d.count)  infer " + String(format: "%.1f", raw.inferMs) +
                          "ms  e2e " + String(format: "%.1f", total) + "ms"
                await MainActor.run { dets = d; status = msg }
            } catch {
                await MainActor.run { status = "ERROR: \(error)" }
            }
        }
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
