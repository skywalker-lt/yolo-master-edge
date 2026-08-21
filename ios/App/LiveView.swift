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
    @State private var debugLine = ""
    @State private var running = false
    @State private var loopTask: Task<Void, Never>?

    var body: some View {
        ZStack {
            if models.isEmpty {
                ContentUnavailableView("No models bundled",
                                       systemImage: "shippingbox",
                                       description: Text("Copy .mlpackage files into ios/Models/, re-run xcodegen, rebuild."))
            } else if camera.authorized {
                // preview and overlay MUST share one coordinate space: both
                // full-screen, safe area ignored together, or boxes shear by
                // the notch/home-bar insets
                ZStack {
                    CameraPreview(session: camera.session)
                    overlay
                }.ignoresSafeArea()
            } else {
                ContentUnavailableView("Camera access required",
                                       systemImage: "camera.fill")
            }
            VStack {
                controls
                Spacer()
                VStack(spacing: 2) {
                    Text(String(format: "%.1f ms  (%.0f FPS)  %@ @%@",
                                inferMS, inferMS > 0 ? 1000.0 / inferMS : 0.0,
                                selectedModel?.id ?? "-", compute.rawValue))
                    Text(debugLine)
                }
                .font(.caption.monospacedDigit())
                .padding(6)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
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
                .disabled(selectedModel == nil)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding()
    }

    private var overlay: some View {
        GeometryReader { geo in
            Canvas(rendersAsynchronously: true) { ctx, size in
                // the preview layer is resizeAspectFill: one uniform scale (the
                // larger axis ratio), surplus cropped equally. Mirror exactly
                // that mapping or boxes shear toward the center.
                let scale = max(size.width / frameSize.width,
                                size.height / frameSize.height)
                let ox = (size.width - frameSize.width * scale) / 2
                let oy = (size.height - frameSize.height * scale) / 2
                DetOverlay.draw(ctx, detections, scale: scale, ox: ox, oy: oy)
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
            var uiTicks = 0
            var uiWindowStart = CFAbsoluteTimeGetCurrent()
            var uiHz = 0.0
            var lastFrameID: UInt64 = 0
            while !Task.isCancelled {
                guard let (pb, fid) = camera.grabLatest() else {
                    try? await Task.sleep(nanoseconds: 20_000_000); continue
                }
                if fid == lastFrameID {                    // stale frame - don't reprocess
                    try? await Task.sleep(nanoseconds: 5_000_000); continue
                }
                lastFrameID = fid
                let t0 = CFAbsoluteTimeGetCurrent()
                guard let raw = try? det.forward(pb) else { continue }
                let t1 = CFAbsoluteTimeGetCurrent()
                let dets = det.decode(raw, conf: 0.25, iou: 0.5)
                let t2 = CFAbsoluteTimeGetCurrent()
                let fwdMS = (t1 - t0) * 1000          // wrap+letterbox+fill+predict
                let preMS = fwdMS - raw.inferMs       // everything before predict
                let decMS = (t2 - t1) * 1000
                uiTicks += 1
                let nowT = CFAbsoluteTimeGetCurrent()
                if nowT - uiWindowStart >= 1.0 {
                    uiHz = Double(uiTicks) / (nowT - uiWindowStart)
                    uiTicks = 0; uiWindowStart = nowT
                }
                let f1 = { (v: Double) in String(format: "%.1f", v) }
                let dbg = "pre \(f1(preMS)) | inf \(f1(raw.inferMs)) | dec \(f1(decMS)) " +
                          "| loop \(f1(uiHz))Hz | dets \(dets.count)"
                let w = CGFloat(CVPixelBufferGetWidth(pb))
                let h = CGFloat(CVPixelBufferGetHeight(pb))
                // fire-and-forget: NEVER block the inference loop on the main
                // thread - a busy render loop otherwise throttles detection to
                // its leftover scheduling gaps
                Task { @MainActor in
                    detections = dets
                    inferMS = fwdMS + decMS
                    debugLine = dbg
                    frameSize = CGSize(width: w, height: h)
                }
                // pace to ~30FPS: cools the phone, nothing visible is faster anyway
                let spent = (CFAbsoluteTimeGetCurrent() - t0)   // full iteration cost
                if spent < 0.033 {
                    try? await Task.sleep(nanoseconds: UInt64((0.033 - spent) * 1e9))
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


let cocoNames = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
