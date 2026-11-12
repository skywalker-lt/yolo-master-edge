// Live camera detection: preview layer + Canvas overlay + model/compute pickers.
// Detection loop: grab freshest frame -> Kit forward(pixelBuffer) -> decode -> draw.
import AudioToolbox
import AVFoundation
import Photos
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
    @State private var compute: ComputeChoice = ComputeChoice.deviceDefault
    @AppStorage("allowCPU") private var allowCPU = false
    @State private var detections: [Detection] = []
    @State private var frameSize = CGSize(width: 720, height: 1280)
    @State private var statPre: Double = 0
    @State private var statInf: Double = 0
    @State private var statDec: Double = 0
    @State private var statMask: Double = 0
    @State private var statHz: Double = 0
    @State private var statDets = 0
    @State private var statLoopHz: Double = 0   // ACTUAL label refresh rate
    @State private var statCamHz: Double = 0    // sensor->app frame delivery rate
    @State private var statAgeMs: Double = 0    // frame age at delivery (pipeline latency)
    @State private var maskImage: CGImage?          // seg: per-frame composite overlay
    @State private var isSegModel = false           // set once the running Detector loads
    @State private var isNMSFree = false            // ditto: end2end model, IoU lever hidden
    @State private var poseKeypoints = 0            // ditto: keypoints per person of a pose model (17, 133), 0 for any other
    @State private var classNames: [String] = cocoNames
    @State private var showTuning = false
    @State private var showHUD = true
    @State private var style: BoxStyle = .chip
    @StateObject private var tuning = Tuning()
    @State private var zoomBase: CGFloat = 1  // display zoom factor at gesture start
    @State private var focusPoint: CGPoint?   // last tap, preview-layer coords
    @State private var focusVisible = false
    @State private var shutterFlash = false   // brief black blink on capture
    @State private var capturing = false      // capture->compose->save in flight
    @State private var lensExpanded = false   // zoom chip expanded to lens buttons
    @State private var lensCollapse: Task<Void, Never>?
    @State private var lensRect: CGRect = .zero   // global frame of the lens picker
    @State private var frameMirrored = false      // `detections` came from a front-camera frame
    @State private var switchingCamera = false    // a back/front flip is in flight
    @State private var flipAngle: Double = 0      // preview turn about its vertical axis during a flip
    @State private var flipCover: Double = 0      // black cover that hides the camera cut mid-flip
    @State private var flipTurns: Double = 0      // half-turns of the flip button's arrows
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private let haptic = UIImpactFeedbackGenerator(style: .medium)
    private let lightHaptic = UIImpactFeedbackGenerator(style: .light)
    private let heavyHaptic = UIImpactFeedbackGenerator(style: .heavy)
    private let firmHaptic = UIImpactFeedbackGenerator(style: .rigid)   // camera flip
    @State private var longPressFired = false  // swallow the tap that follows a long press
    @State private var loadingModel = false   // Detector init in flight (compile + unit load)
    @State private var loadError: String?     // why the selected model could not be loaded
    @State private var running = false        // loop actually alive
    @State private var wantRun = true         // the user's intent - survives tab
                                              // switches and app backgrounding;
                                              // true at launch = cold-start auto-run
    @State private var loopTask: Task<Void, Never>?
    @State private var visible = false        // Live is the tab on screen

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
                        // dead zone: taps on or near the lens picker are zoom
                        // interaction, never focus selection (the preview is
                        // full-screen, so its coords equal global coords)
                        if lensRect.insetBy(dx: -28, dy: -28).contains(lp) { return }
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
                // camera flip: preview and overlay turn together about the
                // vertical axis; black behind them so the turning edges never
                // show the system background
                .rotation3DEffect(.degrees(flipAngle), axis: (x: 0, y: 1, z: 0), perspective: 0.6)
                .background(Color.black)
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
            Color.black
                .opacity(shutterFlash ? 1 : 0)
                .ignoresSafeArea()
                .allowsHitTesting(false)
            // camera flip cover: fully black at the flip's midpoint, which is
            // when the camera is swapped underneath
            Color.black
                .opacity(flipCover)
                .ignoresSafeArea()
                .allowsHitTesting(false)
            if loadingModel {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Loading the model (\(compute.rawValue))")
                }
                .font(.caption)
                .padding(14)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
            VStack {
                Spacer()
                if camera.authorized, !models.isEmpty {
                    lensControl
                        .padding(.bottom, 6)
                    // shutter centred, camera flip to its right (camera-app layout)
                    ZStack {
                        Button { captureFrame() } label: {
                            ZStack {
                                Circle().stroke(.white.opacity(0.9), lineWidth: 3)
                                    .frame(width: 58, height: 58)
                                if capturing {
                                    ProgressView()
                                        .tint(.white)
                                        .scaleEffect(1.4)
                                } else {
                                    Circle().fill(.white)
                                        .frame(width: 46, height: 46)
                                }
                            }
                        }
                        .buttonStyle(ShutterStyle())
                        .disabled(capturing || switchingCamera)
                        HStack {
                            Spacer()
                            flipButton
                        }
                        .padding(.trailing, 44)
                    }
                    .padding(.bottom, 8)
                }
                if showHUD {
                    StatsHUD(fps: statHz, pre: statPre, inf: statInf,
                             dec: statDec, dets: statDets,
                             active: running && !loadingModel,
                             extras: running && !loadingModel
                                 ? [("camera", String(format: "%.1f fps", statCamHz)),
                                    ("loop", String(format: "%.1f fps", statLoopHz)),
                                    ("frame age", String(format: "%.0f ms", statAgeMs))]
                                 : [],
                             mask: isSegModel ? statMask : nil)
                        .padding(.bottom, 20)
                }
            }
        }
        .safeAreaInset(edge: .top) {
            // same vertical anchor as the photo tab's top bar
            VStack(spacing: 8) {
                controls
                if let loadError {
                    Text(loadError)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                        .padding(.horizontal)
                }
                if showTuning {
                    TuningPanel(conf: $tuning.conf, iou: isNMSFree ? nil : $tuning.iou, style: $style,
                                hudVisible: $showHUD,
                                segOverlay: isSegModel ? $tuning.segOverlay : nil,
                                pose: poseKeypoints > 0 ? $tuning.pose : nil,
                                wholeBody: poseKeypoints == 133)
                        .padding(.horizontal)
                }
            }
        }
        .onChange(of: allowCPU) { _, on in if !on && compute == .cpu { compute = ComputeChoice.deviceDefault } }
        .onChange(of: selectedModel) { _, _ in loadError = nil }   // the message was about the previous model
        .onAppear {
            visible = true
            if !allowCPU && compute == .cpu { compute = ComputeChoice.deviceDefault }
            camera.start()
            haptic.prepare()
            lightHaptic.prepare()
            heavyHaptic.prepare()
            firmHaptic.prepare()
            // NO photo-library warm-up here. There used to be an EMPTY
            // performChanges {} to spin up the Photos connection ahead of the first
            // shutter tap. A change request that creates nothing is not an add-only
            // request: PhotoKit asks for read-write access for it, and this app
            // declares no NSPhotoLibraryUsageDescription (it only ever adds), so iOS
            // aborted the app (termination reason TCC) - on EVERY launch from the
            // moment saving photos had been allowed, on a phone that had never
            // answered a read-write prompt. Only ever call performChanges with a
            // creation request inside; the first save pays the connection start-up.
            // Read the model list EVERY time the tab appears, not only at launch: a
            // model imported or deleted in Settings since then has to enter (or
            // leave) the menu - it used to stay invisible until the app was
            // relaunched. discover() only lists files, nothing is compiled; it
            // stays off the main thread, behind the Initializing card at launch.
            Task.detached(priority: .userInitiated) {
                let found = BundledModel.discover()
                await MainActor.run {
                    if found != models { models = found }
                    initializing = false
                    let keep = BundledModel.reselect(selectedModel, in: found)
                    if keep != selectedModel { selectedModel = keep }
                    // cold-start auto-run, auto-resume on tab return - unless the tab
                    // was left again while the list was being read
                    if wantRun, visible { startLoop() }
                }
            }
        }
        .onDisappear { visible = false; suspendLoop(); camera.stop() }  // pause, intent preserved
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                // Only when Live is the tab on screen. This handler runs whichever tab is
                // showing: coming back from the Home screen on Photo, Bench or Settings
                // used to restart the camera and the detection loop behind that tab - and
                // would now hold the screen awake there as well (see keepAwake).
                guard visible else { return }
                camera.start()
                if wantRun { startLoop() }             // auto-resume from background
            } else {
                suspendLoop()
            }
        }
    }

    /// Camera-app style: whole numbers bare (1x, 2x), fractions one decimal (0.5x, 2.4x).
    private func zoomLabel(_ z: CGFloat) -> String {
        let r = (z * 10).rounded() / 10
        return r == r.rounded() ? "\(Int(r))x" : String(format: "%.1fx", r)
    }

    /// The native lens the current zoom is riding on (nearest stop at or below).
    private var activeLens: CGFloat {
        camera.lensFactors.last { camera.displayZoom >= $0 - 0.05 }
            ?? camera.lensFactors.first ?? 1
    }

    /// Collapsed: one chip with the current factor. Tapped: one button per
    /// native lens (camera-app style); collapses back after 3s of no input.
    private var lensControl: some View {
        HStack(spacing: 4) {
            if lensExpanded {
                ForEach(camera.lensFactors, id: \.self) { f in
                    let isActive = f == activeLens
                    Button {
                        zoomBase = camera.setZoom(f)
                        bumpLensTimer()
                    } label: {
                        // active button reads the LIVE zoom, others their stop
                        Text(isActive ? zoomLabel(camera.displayZoom) : zoomLabel(f))
                            .font(.caption.monospacedDigit().bold())
                            .foregroundStyle(isActive ? .yellow : .primary)
                            .frame(minWidth: 34)
                            .padding(.vertical, 6)
                    }
                }
            } else {
                Button {
                    withAnimation(.spring(duration: 0.25)) { lensExpanded = true }
                    bumpLensTimer()
                } label: {
                    Text(zoomLabel(camera.displayZoom))
                        .font(.caption.monospacedDigit().bold())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                }
            }
        }
        .padding(.horizontal, lensExpanded ? 8 : 0)
        .background(.ultraThinMaterial, in: Capsule())
        .buttonStyle(.plain)
        .background(GeometryReader { g in
            Color.clear
                .onAppear { lensRect = g.frame(in: .global) }
                .onChange(of: g.frame(in: .global)) { _, r in lensRect = r }
        })
    }

    /// Restart the 3s auto-collapse window.
    private func bumpLensTimer() {
        lensCollapse?.cancel()
        lensCollapse = Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(.spring(duration: 0.25)) { lensExpanded = false }
        }
    }

    /// Back/front camera toggle, to the right of the shutter. The two arrows
    /// make a half-turn on every flip (the glyph looks the same after 180
    /// degrees, so it always comes to rest upright).
    private var flipButton: some View {
        Button { flipCamera() } label: {
            Image(systemName: "arrow.triangle.2.circlepath")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.primary)
                .rotationEffect(.degrees(flipTurns * 180))
                .frame(width: 46, height: 46)
                .background(.ultraThinMaterial, in: Circle())
        }
        .buttonStyle(ShutterStyle())
        .disabled(switchingCamera || capturing)
        .accessibilityLabel(camera.isFront ? "Switch to the back camera"
                                           : "Switch to the front camera")
    }

    /// Flip between the back and the front camera.
    /// Feel: a firm (rigid, full-strength) haptic on the tap; the preview turns
    /// away about its vertical axis while a black cover fades in; the camera is
    /// swapped behind the cover; the new preview turns in from the opposite
    /// side and a second firm haptic lands with it. With Reduce Motion on, the
    /// turn is skipped and only the fade remains.
    private func flipCamera() {
        guard !switchingCamera, !capturing else { return }
        switchingCamera = true
        firmHaptic.impactOccurred(intensity: 1.0)
        firmHaptic.prepare()
        lensCollapse?.cancel()
        let turnOut = reduceMotion ? 0.12 : 0.2
        withAnimation(.easeIn(duration: turnOut)) {
            if !reduceMotion { flipAngle = 90 }
            flipCover = 1
            lensExpanded = false
            focusVisible = false
        }
        withAnimation(.spring(duration: 0.55, bounce: 0.3)) { flipTurns += 1 }
        DispatchQueue.main.asyncAfter(deadline: .now() + turnOut) {
            // hidden now: wipe the outgoing camera's overlay and swap underneath
            detections = []
            maskImage = nil
            camera.switchCamera { switched in
                if switched { zoomBase = 1 }
                // jump to the far side with animations off (invisible: edge-on
                // and covered), then turn in on the next runloop pass so the
                // jump is committed before the animated change starts
                var jump = Transaction()
                jump.disablesAnimations = true
                withTransaction(jump) { if !reduceMotion { flipAngle = -90 } }
                DispatchQueue.main.async {
                    withAnimation(.spring(duration: 0.42, bounce: 0.2)) { flipAngle = 0 }
                    withAnimation(.easeOut(duration: 0.3)) { flipCover = 0 }
                    if switched {
                        firmHaptic.impactOccurred(intensity: 0.8)
                        firmHaptic.prepare()
                    }
                    switchingCamera = false
                }
            }
        }
    }

    private var controls: some View {
        HStack {
            Menu {
                Picker("Model", selection: $selectedModel) {
                    ForEach(models) { m in
                        Text(m.fullName).tag(Optional(m))
                    }
                }
            } label: {
                HStack(spacing: 3) {
                    Text(selectedModel?.shortID ?? "Model").lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                }
            }
            .fixedSize()
            .disabled(wantRun)          // pause inference before switching (mac GUI rule)
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.available(allowCPU: allowCPU)) { c in
                    Text(c.rawValue).lineLimit(1).fixedSize().tag(c)
                }
            }
            .fixedSize()
            .disabled(wantRun)
            Button {
                lightHaptic.impactOccurred()
                lightHaptic.prepare()
                camera.setTorch(!camera.torchOn)
            } label: {
                // flashlight glyphs are taller than the other symbols - pin the
                // label box so every bordered button comes out the same height
                Image(systemName: camera.torchOn ? "flashlight.on.fill" : "flashlight.off.fill")
                    .font(.system(size: 15))
                    .frame(width: 22, height: 20)
            }
            .buttonStyle(.bordered)
            .tint(camera.torchOn ? .yellow : nil)
            .disabled(camera.isFront)   // the front camera has no torch
            Button {
                withAnimation { showTuning.toggle() }
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .frame(width: 22, height: 20)
            }
            .buttonStyle(.bordered)
            Button {
                if longPressFired { longPressFired = false; return }
                togglePlay(strong: false)
            } label: {
                Image(systemName: wantRun ? "pause.fill" : "play.fill")
                    .frame(minWidth: 44, minHeight: 20)
            }
            .buttonStyle(.borderedProminent)
            .tint(wantRun ? .red : .accentColor)
            .disabled(selectedModel == nil)
            .simultaneousGesture(
                LongPressGesture(minimumDuration: 0.4).onEnded { _ in
                    longPressFired = true
                    togglePlay(strong: true)
                    // if the finger lifts outside the button no tap follows -
                    // do not leave the swallow flag armed
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                        longPressFired = false
                    }
                }
            )
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
                    let frameRect = CGRect(x: ox, y: oy, width: frameSize.width * scale,
                                           height: frameSize.height * scale)
                    if frameMirrored {
                        // front camera: the preview is a mirror image, the frame
                        // (and so the mask) is not - flip the mask about the
                        // frame's vertical centre line: x -> minX + maxX - x
                        var flipped = ctx
                        flipped.translateBy(x: frameRect.minX + frameRect.maxX, y: 0)
                        flipped.scaleBy(x: -1, y: 1)
                        flipped.draw(Image(decorative: mask, scale: 1), in: frameRect)
                    } else {
                        ctx.draw(Image(decorative: mask, scale: 1), in: frameRect)
                    }
                }
                DetOverlay.draw(ctx, detections, scale: scale, ox: ox, oy: oy, style: style,
                                names: classNames,
                                boxes: !(isSegModel && tuning.segOverlay == .masks) && !skeletonOnly,
                                labels: !skeletonOnly,
                                mirrorWidth: frameMirrored ? frameSize.width : nil,
                                pose: tuning.pose)
            }
        }
        .allowsHitTesting(false)
    }

    /// A pose model set to show the skeleton only: no person boxes, no labels - on screen and
    /// in a saved photo alike.
    private var skeletonOnly: Bool { poseKeypoints > 0 && tuning.pose.overlay == .skeleton }

    /// Hold the screen awake while the camera is detecting: a viewfinder that dims and locks
    /// after 30 seconds without a touch is no use for watching detections. Released whenever
    /// the loop stops - pause, another tab, the Home screen, a model that fails to load - so
    /// the phone sleeps on its normal timer everywhere else. Main thread only.
    private func keepAwake(_ on: Bool) {
        UIApplication.shared.isIdleTimerDisabled = on
    }

    private func startLoop() {
        guard let model = selectedModel, !running else { return }
        running = true
        keepAwake(true)
        loadingModel = true
        let mode = compute.mode
        let tuningRef = tuning      // plain class ref: detached-safe, no wrapper
        let cameraRef = camera
        loadError = nil
        loopTask = Task.detached(priority: .userInitiated) {
            let det: Detector
            do {
                det = try model.makeDetector(compute: mode)
            } catch {
                // A model that cannot be loaded (a damaged package, or one that is not a
                // detector the Kit can decode): say why and go back to the paused state.
                // It used to look like a hung run - pause glyph, pickers greyed out,
                // nothing drawn, no message.
                let why = error.localizedDescription
                await MainActor.run {
                    running = false
                    keepAwake(false)
                    loadingModel = false
                    wantRun = false
                    loadError = "\(model.fullName) could not be loaded. \(why)"
                }
                return
            }
            let seg = det.isSegment
            let nmsFree = det.end2end
            let names = det.classNames
            let kpts = det.isPose ? det.keypointCount : 0
            await MainActor.run {
                isSegModel = seg; isNMSFree = nmsFree; classNames = names; loadingModel = false
                poseKeypoints = kpts
            }
            var uiTicks = 0
            var uiWindowStart = CFAbsoluteTimeGetCurrent()
            var uiHz = 0.0
            var lastFrameID: UInt64 = 0
            while !Task.isCancelled {
                guard let (pb, fid, front) = cameraRef.grabLatest() else {
                    try? await Task.sleep(nanoseconds: 20_000_000); continue
                }
                if fid == lastFrameID {                    // stale frame - don't reprocess
                    try? await Task.sleep(nanoseconds: 5_000_000); continue
                }
                lastFrameID = fid
                let confNow = Float(tuningRef.conf)
                // an NMS-free model ignores the (hidden) IoU lever
                let iouNow = det.nmsThreshold(lever: CGFloat(tuningRef.iou))
                let t0 = CFAbsoluteTimeGetCurrent()
                // autoreleasepool per frame: without it, autoreleased CoreML/CG
                // transients accumulate on the cooperative thread until jetsam
                // kills the app (~40s in CPU mode, whose transients are largest)
                // masks only when wanted this frame: boxes-only mode skips the
                // whole SGEMM+composite, and the stage bar drops to ~0
                let segNow = seg && tuningRef.segOverlay != .boxes
                // The detector and the overlay both work on an UPRIGHT frame.
                // The back camera delivers one. The front camera's frames
                // arrive in the sensor's landscape orientation (the capture
                // pipeline does not rotate them on device), so a sideways
                // buffer is turned upright here, whichever camera it is from.
                // Its cost lands in the preprocess bar.
                let sideways = CVPixelBufferGetWidth(pb) > CVPixelBufferGetHeight(pb)
                let frame: ([Detection], CGImage?, Double, Double, Double, Double)? = autoreleasepool {
                    let forwarded: Detector.RawOutput?
                    if sideways {
                        forwarded = uprightFrame(pb, front: front).flatMap { try? det.forward($0) }
                    } else {
                        forwarded = try? det.forward(pb)
                    }
                    guard let raw = forwarded else { return nil }
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
                let camHz = cameraRef.deliveryRate()
                let ageMs = cameraRef.frameAgeMs()

                // size of the UPRIGHT frame the detections are in: a sideways
                // buffer's width and height swap when it is turned
                let bw = CGFloat(CVPixelBufferGetWidth(pb))
                let bh = CGFloat(CVPixelBufferGetHeight(pb))
                let w = sideways ? bh : bw
                let h = sideways ? bw : bh
                // fire-and-forget: NEVER block the inference loop on the main
                // thread - a busy render loop otherwise throttles detection to
                // its leftover scheduling gaps
                // model end-to-end capability, NOT the paced loop rate: this
                // can exceed the camera's 30 and peg the dial purple
                let e2eFPS = 1000.0 / max(preMS + infMS + decMS + maskMS, 0.1)
                Task { @MainActor in
                    guard running else { return }   // zombie frame after pause: drop
                    // a result from the camera we have just switched away from
                    // must not repaint the cleared overlay (it stayed frozen
                    // on screen when the new camera delivered nothing)
                    guard front == cameraRef.isFront else { return }
                    detections = dets
                    maskImage = mask
                    frameMirrored = front           // travels WITH its detections
                    statPre = preMS
                    statInf = infMS
                    statDec = decMS
                    statMask = maskMS
                    statHz = e2eFPS
                    statLoopHz = uiHz
                    statCamHz = camHz
                    statAgeMs = ageMs
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

    /// Shutter: sound + haptic + black blink, then a FULL-RESOLUTION still via
    /// AVCapturePhotoOutput, cropped to the video FOV so the live overlay maps
    /// onto it by pure scaling; mask composite + Kit-rendered boxes/labels are
    /// baked in off-main and the result saved to Photos. Falls back to the
    /// 720p video frame if the photo capture fails.
    private func captureFrame() {
        guard !capturing else { return }
        capturing = true
        AudioServicesPlaySystemSound(1108)                       // classic shutter
        haptic.impactOccurred()
        haptic.prepare()
        shutterFlash = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
            withAnimation(.easeOut(duration: 0.2)) { shutterFlash = false }
        }
        let dets = detections
        let mask = maskImage
        let drawMask = isSegModel && tuning.segOverlay != .boxes
        let drawBoxes = !(isSegModel && tuning.segOverlay == .masks) && !skeletonOnly
        // the saved photo shows what the screen shows: same pose settings, and in
        // skeleton-only mode neither person boxes nor labels
        let labelMode: LabelMode = skeletonOnly ? .off : .full
        let poseNow = tuning.pose
        let names = classNames
        let fSize = frameSize
        let kitStyle = style.kitStyle
        let front = camera.isFront
        let fallback = camera.grabLatest().flatMap { uprightFrame($0.0, front: $0.2) }
        let t0 = CFAbsoluteTimeGetCurrent()
        camera.capturePhoto { shot in
            Task.detached(priority: .userInitiated) {
                var base: CGImage? = nil
                var s: CGFloat = 1
                // The detections are in an upright portrait frame. A still that
                // comes back LANDSCAPE was not rotated by the capture pipeline
                // (the front camera): turn it the same way as its video frames.
                let still = shot.flatMap { (img: CGImage) -> CGImage? in
                    img.width > img.height
                        ? quarterTurned(img, clockwise: CameraController.sidewaysTurnIsClockwise(front: front))
                        : img
                }
                if let photo = still {
                    // photo sensor FOV is wider than the 16:9 video FOV the
                    // detections live in - crop to the centered video-aspect
                    // region, then overlay coords map by one scale factor
                    let pw = CGFloat(photo.width), ph = CGFloat(photo.height)
                    let va = fSize.width / fSize.height
                    var crop = CGRect(x: 0, y: 0, width: pw, height: ph)
                    if pw / ph > va {
                        let cw = (ph * va).rounded()
                        crop = CGRect(x: ((pw - cw) / 2).rounded(), y: 0, width: cw, height: ph)
                    } else if pw / ph < va {
                        let ch = (pw / va).rounded()
                        crop = CGRect(x: 0, y: ((ph - ch) / 2).rounded(), width: pw, height: ch)
                    }
                    if let c = photo.cropping(to: crop) {
                        base = c
                        s = crop.height / fSize.height
                    }
                }
                if base == nil { base = fallback; s = 1 }
                guard var img = base else {
                    await rearmShutter(t0)
                    return
                }
                // boxes AND pose keypoints scale onto the larger still
                let scaled = s == 1 ? dets : dets.map { $0.scaled(by: s) }
                if drawMask, let mask,
                   let ctx = CGContext(data: nil, width: img.width, height: img.height,
                                       bitsPerComponent: 8, bytesPerRow: 0,
                                       space: CGColorSpaceCreateDeviceRGB(),
                                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                    ctx.interpolationQuality = .high
                    let full = CGRect(x: 0, y: 0, width: img.width, height: img.height)
                    ctx.draw(img, in: full)
                    ctx.draw(mask, in: full)   // mask covers the video FOV = the cropped base
                    if let out = ctx.makeImage() { img = out }
                }
                if !scaled.isEmpty {
                    img = annotate(img, scaled, names: names, style: kitStyle,
                                   label: labelMode, drawBoxes: drawBoxes, pose: poseNow) ?? img
                }
                let ui = UIImage(cgImage: img)
                if PHPhotoLibrary.authorizationStatus(for: .addOnly) == .notDetermined {
                    _ = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
                }
                try? await PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAsset(from: ui)
                }
                await rearmShutter(t0)
            }
        }
    }

    /// Re-enable the shutter no sooner than ~1s after the tap: the system
    /// shutter sound is about that long, and a fast save re-arming earlier lets
    /// rapid taps overlap sounds into a stutter.
    private func rearmShutter(_ t0: CFAbsoluteTime) async {
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        if elapsed < 1.0 {
            try? await Task.sleep(nanoseconds: UInt64((1.0 - elapsed) * 1e9))
        }
        await MainActor.run { capturing = false }
    }

    /// Stop the loop and WIPE the overlay; keeps `wantRun` untouched so
    /// lifecycle pauses auto-resume.
    private func suspendLoop() {
        running = false
        keepAwake(false)
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
        statLoopHz = 0
        statCamHz = 0
        statAgeMs = 0
        statDets = 0
    }

    /// User-initiated stop (play/pause button).
    private func stopLoop() {
        suspendLoop()
    }

    /// Play/pause with haptics: medium on tap, heavy full-intensity on the
    /// force/long press path.
    private func togglePlay(strong: Bool) {
        if strong {
            heavyHaptic.impactOccurred(intensity: 1.0)
            heavyHaptic.prepare()
        } else {
            haptic.impactOccurred()
            haptic.prepare()
        }
        if wantRun { wantRun = false; stopLoop() }
        else { wantRun = true; startLoop() }
    }
}

/// `image` turned a quarter-turn, as a new bitmap (width and height swap).
///
/// Needed because a camera's sensor is mounted landscape: when the capture
/// pipeline does not rotate a frame, it arrives lying on its side. Which way
/// it has to turn depends on the phone's sensor, not only on front or back:
/// `CameraController.sidewaysTurnIsClockwise(front:)` decides.
///
/// Core Graphics is y-up, so a positive angle is counter-clockwise on screen.
/// The translation moves the turned image back into the positive quadrant:
///   counter-clockwise: (x, y) -> (h - y, x)   the right edge becomes the top
///   clockwise:         (x, y) -> (y, w - x)   the left edge becomes the top
private func quarterTurned(_ image: CGImage, clockwise: Bool) -> CGImage? {
    let w = image.width, h = image.height
    guard let ctx = CGContext(data: nil, width: h, height: w, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                                  | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
    ctx.interpolationQuality = .none      // an exact quarter-turn maps pixel to pixel
    if clockwise {
        ctx.translateBy(x: 0, y: CGFloat(w))
        ctx.rotate(by: -.pi / 2)
    } else {
        ctx.translateBy(x: CGFloat(h), y: 0)
        ctx.rotate(by: .pi / 2)
    }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage()
}

/// An upright image of a camera frame for the detector. A frame that is
/// already portrait is wrapped as it is; a landscape one is the sensor's own
/// orientation and gets a quarter-turn (see `quarterTurned`).
private func uprightFrame(_ pb: CVPixelBuffer, front: Bool) -> CGImage? {
    guard let cg = Detector.cgImage(from: pb) else { return nil }
    return cg.width > cg.height
        ? quarterTurned(cg, clockwise: CameraController.sidewaysTurnIsClockwise(front: front)) : cg
}

/// Camera-app shutter feel: quick shrink while pressed.
struct ShutterStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.85 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}
