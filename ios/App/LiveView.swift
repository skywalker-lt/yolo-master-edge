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
    @State private var statMask: Double = 0
    @State private var statHz: Double = 0
    @State private var statDets = 0
    @State private var maskImage: CGImage?          // seg: per-frame composite overlay
    @State private var isSegModel = false           // set once the running Detector loads
    @State private var classNames: [String] = cocoNames
    @State private var showTuning = false
    @State private var showHUD = true
    @State private var style: BoxStyle = .chip
    @StateObject private var tuning = Tuning()
    @State private var zoomBase: CGFloat = 1  // camera zoom factor at gesture start
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
                }
                .ignoresSafeArea()
                .gesture(
                    MagnifyGesture()
                        .onChanged { v in camera.setZoom(zoomBase * v.magnification) }
                        .onEnded { v in zoomBase = camera.setZoom(zoomBase * v.magnification) }
                )
            } else {
                ContentUnavailableView("Camera access required",
                                       systemImage: "camera.fill")
            }
            VStack {
                Spacer()
                if showHUD {
                    StatsHUD(fps: statHz, pre: statPre, inf: statInf,
                             dec: statDec, dets: statDets, active: running,
                             mask: isSegModel ? statMask : nil)
                        .padding(.bottom, 20)
                }
            }
        }
        .safeAreaInset(edge: .top) {
            // same vertical anchor as the photo tab's top bar
            VStack(spacing: 8) {
                controls
                if showTuning {
                    TuningPanel(conf: $tuning.conf, iou: $tuning.iou, style: $style,
                                hudVisible: $showHUD,
                                segOverlay: isSegModel ? $tuning.segOverlay : nil)
                        .padding(.horizontal)
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
        .padding(.horizontal)   // no top padding: aligns with the photo tab's bar
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
                // mask composite is camera-frame-sized: same rect as the frame
                // itself under aspect-fill, drawn beneath the boxes/labels
                if let mask = maskImage {
                    ctx.draw(Image(decorative: mask, scale: 1),
                             in: CGRect(x: ox, y: oy, width: frameSize.width * scale,
                                        height: frameSize.height * scale))
                }
                DetOverlay.draw(ctx, detections, scale: scale, ox: ox, oy: oy, style: style,
                                names: classNames,
                                boxes: !(isSegModel && tuning.segOverlay == .masks))
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
            let seg = det.isSegment
            let names = det.classNames
            await MainActor.run { isSegModel = seg; classNames = names }
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
                // autoreleasepool per frame: without it, autoreleased CoreML/CG
                // transients accumulate on the cooperative thread until jetsam
                // kills the app (~40s in CPU mode, whose transients are largest)
                // masks only when wanted this frame: boxes-only mode skips the
                // whole SGEMM+composite, and the stage bar drops to ~0
                let segNow = seg && tuningRef.segOverlay != .boxes
                let frame: ([Detection], CGImage?, Double, Double, Double, Double)? = autoreleasepool {
                    guard let raw = try? det.forward(pb) else { return nil }
                    let t1 = CFAbsoluteTimeGetCurrent()
                    let d = det.decode(raw, conf: confNow, iou: iouNow)
                    let t2 = CFAbsoluteTimeGetCurrent()
                    let mask: CGImage? = segNow && !d.isEmpty
                        ? det.maskOverlay(Array(d.prefix(100)), raw) : nil
                    let t3 = CFAbsoluteTimeGetCurrent()
                    let fwd = (t1 - t0) * 1000
                    return (d, mask, fwd - raw.inferMs, raw.inferMs,
                            (t2 - t1) * 1000, (t3 - t2) * 1000)
                }
                guard let (dets, mask, preMS, infMS, decMS, maskMS) = frame else { continue }
                uiTicks += 1
                let nowT = CFAbsoluteTimeGetCurrent()
                if nowT - uiWindowStart >= 1.0 {
                    uiHz = Double(uiTicks) / (nowT - uiWindowStart)
                    uiTicks = 0; uiWindowStart = nowT
                }

                let w = CGFloat(CVPixelBufferGetWidth(pb))
                let h = CGFloat(CVPixelBufferGetHeight(pb))
                // fire-and-forget: NEVER block the inference loop on the main
                // thread - a busy render loop otherwise throttles detection to
                // its leftover scheduling gaps
                // model end-to-end capability, NOT the paced loop rate: this
                // can exceed the camera's 30 and peg the dial purple
                let e2eFPS = 1000.0 / max(preMS + infMS + decMS + maskMS, 0.1)
                Task { @MainActor in
                    guard running else { return }   // zombie frame after pause: drop
                    detections = dets
                    maskImage = mask
                    statPre = preMS
                    statInf = infMS
                    statDec = decMS
                    statMask = maskMS
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
        maskImage = nil          // same autowipe rule as the boxes
        statMask = 0
        statDets = 0
    }

    /// User-initiated stop (play/pause button).
    private func stopLoop() {
        suspendLoop()
    }
}
