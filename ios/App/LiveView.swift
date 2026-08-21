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
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var camera = CameraController()
    @State private var models = BundledModel.discover()
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = .ane
    @State private var detections: [Detection] = []
    @State private var frameSize = CGSize(width: 720, height: 1280)
    @State private var statPre: Double = 0
    @State private var statInf: Double = 0
    @State private var statDec: Double = 0
    @State private var statHz: Double = 0
    @State private var statDets = 0
    @State private var showTuning = false
    @State private var showHUD = true
    @State private var style: BoxStyle = .chip
    @StateObject private var tuning = Tuning()
    @State private var running = false        // loop actually alive
    @State private var wantRun = false        // the user's intent - survives tab
                                              // switches and app backgrounding
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
                if showTuning {
                    TuningPanel(conf: $tuning.conf, iou: $tuning.iou, style: $style,
                                hudVisible: $showHUD)
                        .padding(.horizontal)
                }
                Spacer()
                if showHUD {
                    StatsHUD(fps: statHz, pre: statPre, inf: statInf,
                             dec: statDec, dets: statDets, active: running)
                        .padding(.bottom, 8)
                }
            }
        }
        .onAppear {
            camera.start()
            if selectedModel == nil { selectedModel = models.first }
            if wantRun { startLoop() }                 // auto-resume on tab return
        }
        .onDisappear { suspendLoop(); camera.stop() }  // pause, intent preserved
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                camera.start()
                if wantRun { startLoop() }             // auto-resume from background
            } else {
                suspendLoop()
            }
        }
    }

    private var controls: some View {
        HStack {
            Picker("Model", selection: $selectedModel) {
                ForEach(models) { m in Text(m.shortID).lineLimit(1).fixedSize().tag(Optional(m)) }
            }
            .fixedSize()
            .disabled(wantRun)          // pause inference before switching (mac GUI rule)
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).lineLimit(1).fixedSize().tag(c) }
            }
            .fixedSize()
            .disabled(wantRun)
            Button {
                withAnimation { showTuning.toggle() }
            } label: {
                Image(systemName: "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            Button {
                if wantRun { wantRun = false; stopLoop() }
                else { wantRun = true; startLoop() }
            } label: {
                Image(systemName: wantRun ? "pause.fill" : "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .tint(wantRun ? .red : .accentColor)
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
                DetOverlay.draw(ctx, detections, scale: scale, ox: ox, oy: oy, style: style)
            }
        }
        .allowsHitTesting(false)
    }

    private func startLoop() {
        guard let model = selectedModel, !running else { return }
        running = true
        let mode = compute.mode
        let tuningRef = tuning      // plain class ref: detached-safe, no wrapper
        let cameraRef = camera
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
                guard let (pb, fid) = cameraRef.grabLatest() else {
                    try? await Task.sleep(nanoseconds: 20_000_000); continue
                }
                if fid == lastFrameID {                    // stale frame - don't reprocess
                    try? await Task.sleep(nanoseconds: 5_000_000); continue
                }
                lastFrameID = fid
                let confNow = Float(tuningRef.conf)
                let iouNow = CGFloat(tuningRef.iou)
                let t0 = CFAbsoluteTimeGetCurrent()
                guard let raw = try? det.forward(pb) else { continue }
                let t1 = CFAbsoluteTimeGetCurrent()
                let dets = det.decode(raw, conf: confNow, iou: iouNow)
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
                let infMS = raw.inferMs
                let w = CGFloat(CVPixelBufferGetWidth(pb))
                let h = CGFloat(CVPixelBufferGetHeight(pb))
                // fire-and-forget: NEVER block the inference loop on the main
                // thread - a busy render loop otherwise throttles detection to
                // its leftover scheduling gaps
                // model end-to-end capability, NOT the paced loop rate: this
                // can exceed the camera's 30 and peg the dial purple
                let e2eFPS = 1000.0 / max(preMS + infMS + decMS, 0.1)
                Task { @MainActor in
                    guard running else { return }   // zombie frame after pause: drop
                    detections = dets
                    statPre = preMS
                    statInf = infMS
                    statDec = decMS
                    statHz = e2eFPS
                    statDets = dets.count
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

    /// Stop the loop and WIPE the overlay; keeps `wantRun` untouched so
    /// lifecycle pauses auto-resume.
    private func suspendLoop() {
        running = false
        loopTask?.cancel()
        loopTask = nil
        detections = []
        statDets = 0
    }

    /// User-initiated stop (play/pause button).
    private func stopLoop() {
        suspendLoop()
    }
}


let cocoNames = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
