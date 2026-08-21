// Camera-free sanity check: run the detector on any photo from the library.
// Isolates the model+decode pipeline from the capture path entirely.
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
            VStack {
                HStack {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(models) { m in Text(m.id).tag(Optional(m)) }
                    }
                    Picker("Unit", selection: $compute) {
                        ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).tag(c) }
                    }
                    PhotosPicker(selection: $pick, matching: .images) { Text("Photo") }
                }.padding(.horizontal)
                if let img = image {
                    GeometryReader { geo in
                        let scale = min(geo.size.width / CGFloat(img.width),
                                        geo.size.height / CGFloat(img.height))
                        ZStack(alignment: .topLeading) {
                            Image(decorative: img, scale: 1).resizable()
                                .frame(width: CGFloat(img.width) * scale,
                                       height: CGFloat(img.height) * scale)
                            Canvas { ctx, _ in
                                for d in dets {
                                    let r = CGRect(x: d.rect.minX * scale, y: d.rect.minY * scale,
                                                   width: d.rect.width * scale, height: d.rect.height * scale)
                                    ctx.stroke(Path(r), with: .color(.red), lineWidth: 2)
                                    ctx.draw(Text("\(d.cls):\(Int(d.score * 100))")
                                        .font(.caption2.bold()).foregroundStyle(.red),
                                             at: CGPoint(x: r.minX, y: max(r.minY - 8, 4)), anchor: .leading)
                                }
                            }
                        }
                    }
                } else {
                    ContentUnavailableView("No photo", systemImage: "photo")
                }
                Text(status).font(.caption.monospacedDigit())
            }
            .navigationTitle("Photo test")
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
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                await MainActor.run { status = "photo load failed" }; return
            }
            await MainActor.run { image = img }
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
                let raw = try det.forward(img)
                let cands = det.candidates(raw, confFloor: 0.01)
                let d = det.decode(raw, conf: 0.25, iou: 0.5)
                let msg = String(format: "dets %d | cands>0.01 %d | top %.3f | infer %.1fms",
                                 d.count, cands.count, cands.first?.score ?? 0, raw.inferMs)
                await MainActor.run { dets = d; status = msg }
            } catch {
                await MainActor.run { status = "ERROR: \(error)" }
            }
        }
    }
}
