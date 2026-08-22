// Live camera detection: preview layer + Canvas overlay + model/compute pickers.
// Detection loop: grab freshest frame -> Kit forward(pixelBuffer) -> decode -> draw.
import AVFoundation
import SwiftUI
import YOLOMasterKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// Tap callback with BOTH spaces: capture-device point (for focus) and
    /// layer point (for drawing the focus indicator).
    var onTap: ((CGPoint, CGPoint) -> Void)? = nil
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.tapped(_:)))
        v.addGestureRecognizer(tap)
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {
        context.coordinator.parent = self
    }
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    final class Coordinator: NSObject {
        var parent: CameraPreview
        init(_ p: CameraPreview) { parent = p }
        @objc func tapped(_ gr: UITapGestureRecognizer) {
            guard let v = gr.view as? PreviewView else { return }
            let lp = gr.location(in: v)
            let dp = v.videoPreviewLayer.captureDevicePointConverted(fromLayerPoint: lp)
            parent.onTap?(dp, lp)
        }
    }
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

struct LiveView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var camera = CameraController()
    @State private var models: [BundledModel] = []
    @State private var initializing = true    // first-launch discovery/compile in flight
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
    @State private var zoomBase: CGFloat = 1  // display zoom factor at gesture start
    @State private var focusPoint: CGPoint?   // last tap, preview-layer coords
    @State private var focusVisible = false
    @State private var loadingModel = false   // Detector init in flight (compile + unit load)
    @State private var running = false        // loop actually alive
    @State private var wantRun = true         // the user's intent - survives tab
                                              // switches and app backgrounding;
                                              // true at launch = cold-start auto-run
    @State private var loopTask: Task<Void, Never>?

    var body: some View {
        ZStack {
            if models.isEmpty {
                if initializing {
                    // first run after install: .mlpackage -> .mlmodelc compile
                    HStack(spacing: 10) {
                        ProgressView()
                        Text("Initializing...")
                    }
                    .font(.caption)
                    .padding(14)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                } else {
                    ContentUnavailableView("No models bundled",
                                           systemImage: "shippingbox",
                                           description: Text("Copy .mlpackage files into ios/Models/, re-run xcodegen, rebuild."))
                }
            } else if camera.authorized {
                // preview and overlay MUST share one coordinate space: both
                // full-screen, safe area ignored together, or boxes shear by
                // the notch/home-bar insets
                ZStack {
                    CameraPreview(session: camera.session) { dp, lp in
                        camera.focus(atDevicePoint: dp)
                        focusPoint = lp
                        withAnimation(.easeOut(duration: 0.15)) { focusVisible = true }
                        Task {
                            try? await Task.sleep(nanoseconds: 900_000_000)
                            withAnimation(.easeOut(duration: 0.3)) { focusVisible = false }
                        }
                    }
                    overlay
                    if let fp = focusPoint, focusVisible {
                        RoundedRectangle(cornerRadius: 3)
                            .stroke(.yellow, lineWidth: 1.5)
                            .frame(width: 72, height: 72)
                            .position(fp)
                            .transition(.opacity)
                    }
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
            if loadingModel {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Loading the model to \(compute.rawValue)")
                }
                .font(.caption)
                .padding(14)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
            VStack {
                Spacer()
                if camera.authorized, !models.isEmpty {
                    Text(zoomLabel)
                        .font(.caption.monospacedDigit().bold())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.bottom, 6)
                }
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
            if models.isEmpty {
                // discovery compiles bundled packages on first launch - keep it
                // off the main thread, behind the Initializing card
                Task.detached(priority: .userInitiated) {
                    let found = BundledModel.discover()
                    await MainActor.run {
                        models = found
                        initializing = false
                        if selectedModel == nil {
                            selectedModel = BundledModel.preferred(in: found)
                        }
                        if wantRun { startLoop() }     // cold-start auto-run
                    }
                }
            } else if wantRun {
                startLoop()                            // auto-resume on tab return
            }
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

    /// Camera-app style: whole numbers bare (1x, 2x), fractions one decimal (0.5x, 2.4x).
    private var zoomLabel: String {
        let r = (camera.displayZoom * 10).rounded() / 10
        return r == r.rounded() ? "\(Int(r))x" : String(format: "%.1fx", r)
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
                camera.setTorch(!camera.torchOn)
            } label: {
                Image(systemName: camera.torchOn ? "flashlight.on.fill" : "flashlight.off.fill")
            }
            .buttonStyle(.bordered)
            .tint(camera.torchOn ? .yellow : nil)
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
        loadingModel = true
        let mode = compute.mode
        let tuningRef = tuning      // plain class ref: detached-safe, no wrapper
        let cameraRef = camera
        loopTask = Task.detached(priority: .userInitiated) {
            guard let det = try? Detector(modelURL: model.url, compute: mode) else {
                await MainActor.run { running = false; loadingModel = false }
                return
            }
            let seg = det.isSegment
            let names = det.classNames
            await MainActor.run { isSegModel = seg; classNames = names; loadingModel = false }
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
                    // maxSide 640: quarter the composite pixels vs the 720x1280
                    // frame; the proto grid is 160px, so nothing visible is lost
                    let mask: CGImage? = segNow && !d.isEmpty
                        ? det.maskOverlay(Array(d.prefix(100)), raw, maxSide: 640) : nil
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
        loadingModel = false
        loopTask?.cancel()
        loopTask = nil
        detections = []
        maskImage = nil          // same autowipe rule as the boxes
        // full HUD reset - only the thermal state stays (it is device-ambient,
        // not a property of the stopped run)
        statPre = 0
        statInf = 0
        statDec = 0
        statMask = 0
        statHz = 0
        statDets = 0
    }

    /// User-initiated stop (play/pause button).
    private func stopLoop() {
        suspendLoop()
    }
}
