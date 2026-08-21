// Live camera detection: preview layer + Canvas overlay + model/compute pickers.
// Detection loop: grab freshest frame -> Kit forward(pixelBuffer) -> decode -> draw.
import AVFoundation
import SwiftUI
import YOLOMasterKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

struct LiveView: View {
    @StateObject private var camera = CameraController()
    @State private var models = BundledModel.discover()
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = .ane
    @State private var detections: [Detection] = []
    @State private var frameSize = CGSize(width: 720, height: 1280)
    @State private var inferMS: Double = 0
    @State private var running = false
    @State private var loopTask: Task<Void, Never>?

    var body: some View {
        ZStack {
            if camera.authorized {
                CameraPreview(session: camera.session).ignoresSafeArea()
                overlay
            } else {
                ContentUnavailableView("Camera access required",
                                       systemImage: "camera.fill")
            }
            VStack {
                controls
                Spacer()
                Text(String(format: "%.1f ms  (%.0f FPS)  %@ @%@",
                            inferMS, inferMS > 0 ? 1000.0 / inferMS : 0.0,
                            selectedModel?.id ?? "-", compute.rawValue))
                    .font(.caption.monospacedDigit())
                    .padding(6)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.bottom, 8)
            }
        }
        .onAppear {
            camera.start()
            if selectedModel == nil { selectedModel = models.first }
        }
        .onDisappear { stopLoop(); camera.stop() }
    }

    private var controls: some View {
        HStack {
            Picker("Model", selection: $selectedModel) {
                ForEach(models) { m in Text(m.id).tag(Optional(m)) }
            }
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).tag(c) }
            }
            Button(running ? "Stop" : "Run") { running ? stopLoop() : startLoop() }
                .buttonStyle(.borderedProminent)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding()
    }

    private var overlay: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                let sx = size.width / frameSize.width
                let sy = size.height / frameSize.height
                for d in detections {
                    let r = CGRect(x: d.rect.minX * sx, y: d.rect.minY * sy,
                                   width: d.rect.width * sx, height: d.rect.height * sy)
                    ctx.stroke(Path(roundedRect: r, cornerRadius: 3),
                               with: .color(.green), lineWidth: 2)
                    let label = Text("\(d.cls) \(Int(d.score * 100))%")
                        .font(.caption2.bold()).foregroundStyle(.white)
                    ctx.draw(label, at: CGPoint(x: r.minX + 4, y: max(r.minY - 8, 8)),
                             anchor: .leading)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func startLoop() {
        guard let model = selectedModel else { return }
        running = true
        let mode = compute.mode
        loopTask = Task.detached(priority: .userInitiated) {
            guard let det = try? Detector(modelURL: model.url, compute: mode) else {
                await MainActor.run { running = false }
                return
            }
            while !Task.isCancelled {
                guard let pb = camera.grabLatest() else {
                    try? await Task.sleep(nanoseconds: 20_000_000); continue
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                guard let raw = try? det.forward(pb) else { continue }
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                let dets = det.decode(raw, conf: 0.25, iou: 0.5)
                let w = CGFloat(CVPixelBufferGetWidth(pb))
                let h = CGFloat(CVPixelBufferGetHeight(pb))
                await MainActor.run {
                    detections = dets
                    inferMS = ms
                    frameSize = CGSize(width: w, height: h)
                }
            }
        }
    }

    private func stopLoop() {
        running = false
        loopTask?.cancel()
        loopTask = nil
        detections = []
    }
}
