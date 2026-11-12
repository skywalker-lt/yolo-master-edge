// AVFoundation capture -> CVPixelBuffer stream for the Kit's forward(pixelBuffer:).
// iOS-specific by nature (the mac app's Camera.swift is AVCaptureScreen-flavored);
// everything downstream of the buffer is shared Kit code.
import AVFoundation
import CoreVideo
import Foundation
import UIKit

final class CameraController: NSObject, ObservableObject,
                              AVCaptureVideoDataOutputSampleBufferDelegate,
                              AVCapturePhotoCaptureDelegate {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "yolomaster.camera", qos: .userInitiated)
    /// Latest frame, overwritten continuously; the detection loop pulls at its own pace
    /// (frames are dropped, never queued - live detection wants freshness, not history).
    private(set) var latest: CVPixelBuffer?
    private var frameID: UInt64 = 0
    private let latestLock = NSLock()
    private var rateCount = 0
    private var rateStart = CFAbsoluteTimeGetCurrent()
    private var rateValue: Double = 0
    private var ageValue: Double = 0   // EMA of frame age at delivery, ms
    private var device: AVCaptureDevice?
    @Published var authorized = false
    /// videoZoomFactor of the 1x (wide) lens. On virtual multi-cam devices the
    /// factor space starts at the ULTRA-WIDE lens, so 1x = the first switch-over
    /// factor; on single-camera phones (iPhone Air) it stays 1.
    private(set) var zoomBias: CGFloat = 1
    /// User-facing zoom (0.5x / 1x / 2x...), published for the HUD chip.
    @Published var displayZoom: CGFloat = 1
    /// Native lens factors in display units (e.g. [0.5, 1, 4]) for the lens picker.
    @Published var lensFactors: [CGFloat] = [1]
    @Published var torchOn = false
    /// True while the FRONT camera feeds the session (published for the UI).
    /// The preview layer mirrors the front camera like a selfie view; the frames
    /// handed to the detector are NOT mirrored (see `pinUnmirrored`).
    @Published var isFront = false
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private var photoDone: ((CGImage?) -> Void)?
    private var focusRevert: DispatchWorkItem?
    private var latestIsFront = false   // which camera the `latest` frame came from

    /// Pinch-zoom in DISPLAY units (1 = wide lens, 0.5 = ultra-wide). The
    /// virtual device switches lenses automatically as the factor crosses its
    /// switch-over points. Returns the clamped display factor actually applied.
    @discardableResult
    func setZoom(_ display: CGFloat) -> CGFloat {
        guard let dev = device else { return 1 }
        let maxD = min(dev.activeFormat.videoMaxZoomFactor / zoomBias, 16)
        let d = min(max(display, 1 / zoomBias), maxD)
        if (try? dev.lockForConfiguration()) != nil {
            dev.videoZoomFactor = d * zoomBias
            dev.unlockForConfiguration()
        }
        DispatchQueue.main.async { self.displayZoom = d }
        return d
    }

    /// Tap-to-focus: point in capture-device space (from
    /// `captureDevicePointConverted`). Also re-anchors exposure there. Reverts
    /// to continuous AF/AE after 5s: a lingering one-shot focus lock pins the
    /// active constituent lens, which blocks the tele switch-over on zoom.
    func focus(atDevicePoint p: CGPoint) {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if dev.isFocusPointOfInterestSupported {
            dev.focusPointOfInterest = p
            dev.focusMode = .autoFocus
        }
        if dev.isExposurePointOfInterestSupported {
            dev.exposurePointOfInterest = p
            dev.exposureMode = .autoExpose
        }
        dev.unlockForConfiguration()
        focusRevert?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.revertFocus() }
        focusRevert = work
        queue.asyncAfter(deadline: .now() + 5, execute: work)
    }

    private func revertFocus() {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
        dev.unlockForConfiguration()
    }

    /// Full-resolution still through AVCapturePhotoOutput (the video frames are
    /// only 720p). Delivered upright with EXIF orientation baked into pixels;
    /// nil when the output is unavailable (caller falls back to the video frame).
    func capturePhoto(_ done: @escaping (CGImage?) -> Void) {
        queue.async {
            guard self.session.outputs.contains(self.photoOutput) else {
                DispatchQueue.main.async { done(nil) }
                return
            }
            self.photoDone = done
            let st = AVCapturePhotoSettings()
            st.maxPhotoDimensions = self.photoOutput.maxPhotoDimensions
            if let c = self.photoOutput.connection(with: .video) {
                // same convention as the video frames - upright and unmirrored -
                // so the live overlay maps onto the still on either camera
                CameraController.pinUnmirrored(c)
                // back: upright from the pipeline; front: the sensor's own
                // orientation, turned upright by the caller (see pipelineAngle)
                let angle = CameraController.pipelineAngle(front: self.device?.position == .front)
                if c.isVideoRotationAngleSupported(angle) { c.videoRotationAngle = angle }
            }
            self.photoOutput.capturePhoto(with: st, delegate: self)
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let data = error == nil ? photo.fileDataRepresentation() : nil
        let cb = photoDone
        photoDone = nil
        // the orientation bake redraws a 12-24MP image - never on the capture
        // callback queue (it stalls the session), never on main
        DispatchQueue.global(qos: .userInitiated).async {
            var cg: CGImage?
            if let data, let ui = UIImage(data: data) {
                if ui.imageOrientation == .up {
                    cg = ui.cgImage
                } else {
                    let fmt = UIGraphicsImageRendererFormat()
                    fmt.scale = 1
                    cg = UIGraphicsImageRenderer(size: ui.size, format: fmt).image { _ in
                        ui.draw(in: CGRect(origin: .zero, size: ui.size))
                    }.cgImage
                }
            }
            DispatchQueue.main.async { cb?(cg) }
        }
    }

    /// Persistent torch: the desired state is kept and re-asserted every time
    /// the session (re)starts, since AVFoundation drops the torch on stop.
    func setTorch(_ on: Bool) {
        torchOn = on
        queue.async { self.applyTorch() }
    }

    private func applyTorch() {
        guard let dev = device, dev.hasTorch,
              (try? dev.lockForConfiguration()) != nil else { return }
        dev.torchMode = torchOn && dev.isTorchAvailable ? .on : .off
        dev.unlockForConfiguration()
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            DispatchQueue.main.async { self?.authorized = ok }
            guard ok else { return }
            self?.queue.async { self?.configure() }
        }
    }

    /// The camera at `position`. Back: the best virtual multi-cam first (the
    /// device auto-switches lenses as the zoom factor crosses its switch-over
    /// points: ultra-wide / wide / tele). Front: the plain wide-angle camera
    /// (the TrueDepth device adds a depth pipeline this app has no use for).
    private func findCamera(_ position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let types: [AVCaptureDevice.DeviceType] = position == .front
            ? [.builtInWideAngleCamera]
            : [.builtInTripleCamera, .builtInDualWideCamera,
               .builtInDualCamera, .builtInWideAngleCamera]
        return AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video,
                                                position: position).devices.first
    }

    /// Device-side setup shared by first configuration and camera switches:
    /// zoom bias and lens stops, unrestricted lens hand-off, SDR, start at 1x.
    /// Returns the native lens stops in display units (e.g. [0.5, 1, 4]).
    private func prepareDevice(_ dev: AVCaptureDevice) -> [CGFloat] {
        let switchOvers = dev.virtualDeviceSwitchOverVideoZoomFactors.map { CGFloat(truncating: $0) }
        let hasUW = dev.constituentDevices.contains { $0.deviceType == .builtInUltraWideCamera }
        zoomBias = 1    // single-lens devices (the front camera): factor 1 IS 1x
        if dev.isVirtualDevice, hasUW, let so = switchOvers.first {
            zoomBias = so
        }
        // display-unit native lens stops: 1x wide, 0.5x ultra-wide when present,
        // every further switch-over (tele) above
        var stops: [CGFloat] = [1]
        if hasUW { stops.append(1 / zoomBias) }
        stops.append(contentsOf: (hasUW ? Array(switchOvers.dropFirst()) : switchOvers)
            .map { $0 / zoomBias })
        if (try? dev.lockForConfiguration()) != nil {
            if dev.isVirtualDevice {
                // no focus/exposure restrictions on lens hand-off: without this
                // (and with a stale focus lock) the tele never engages
                dev.setPrimaryConstituentDeviceSwitchingBehavior(
                    .auto, restrictedSwitchingBehaviorConditions: [])
            }
            if dev.activeFormat.isVideoHDRSupported {
                // video HDR runs frames through extra fusion stages: deeper
                // buffering = staler data-output frames = overlay trailing the
                // preview. A detection feed wants the low-latency SDR path.
                dev.automaticallyAdjustsVideoHDREnabled = false
                dev.isVideoHDREnabled = false
            }
            dev.videoZoomFactor = zoomBias    // start on the wide lens = 1x
            dev.unlockForConfiguration()
        }
        return stops.sorted()
    }

    /// Deliver frames exactly as the camera sees them. The system mirrors the
    /// front camera's PREVIEW; the data and photo connections must not follow
    /// it, because the detector runs on the true image and the overlay does
    /// the flipping (LiveView) to line up with the mirrored preview.
    private static func pinUnmirrored(_ c: AVCaptureConnection) {
        guard c.isVideoMirroringSupported else { return }
        if c.automaticallyAdjustsVideoMirroring { c.automaticallyAdjustsVideoMirroring = false }
        if c.isVideoMirrored { c.isVideoMirrored = false }
    }

    /// Rotation asked of the capture pipeline. Back camera: 90 (portrait), as
    /// this app has always done. Front camera: 0, the sensor's own landscape
    /// buffer - ON PURPOSE. On an iPhone Air the pipeline never rotated the
    /// front camera's frames, whatever angle was set and whenever it was set,
    /// so the app does not depend on it there: LiveView turns a landscape
    /// frame upright itself, in the direction `sidewaysTurnIsClockwise` gives
    /// for this phone's sensor. Asking for 0 keeps that starting point fixed
    /// on every phone instead of leaving it to whatever the pipeline decides
    /// to honour.
    private static func pipelineAngle(front: Bool) -> CGFloat { front ? 0 : 90 }

    /// Which way a frame that arrives lying on its side must be turned to
    /// stand upright - it depends on how the sensor is mounted, so it is NOT
    /// the same on every iPhone.
    ///
    /// Up to the iPhone 16 family both sensors are mounted the same way round:
    /// an unrotated frame needs a CLOCKWISE quarter-turn. That is the portrait
    /// orientation Apple documents for camera buffers: `.right` for the back
    /// camera and `.leftMirrored` for the front one - the same turn plus the
    /// selfie mirror, which this app applies in the overlay instead.
    /// The square front sensor that came with the iPhone 17 family (measured
    /// on an iPhone Air) is mounted the other way round: its frames need a
    /// COUNTER-CLOCKWISE quarter-turn.
    ///
    /// 1.1.0 build 2 turned every front frame counter-clockwise, a rule taken
    /// from the iPhone Air alone. On an iPhone 15 Pro Max that left the frame
    /// upside down: the detector saw the room inverted (a clock came back as
    /// "frisbee") and every box and mask landed mirrored through the centre
    /// of the screen.
    static func sidewaysTurnIsClockwise(front: Bool) -> Bool {
        guard front else { return true }
        return (iPhoneHardwareGeneration ?? 0) < 18
    }

    /// Orientation and mirroring of the frames handed to the detector.
    /// Connections are rebuilt whenever the input changes, so this goes back
    /// on inside the swap transaction and again once it is committed.
    /// Idempotent. Frames are NEVER dropped for being landscape: LiveView
    /// handles either orientation.
    private func applyConnectionSettings(front: Bool) {
        guard let c = videoOutput.connection(with: .video) else { return }
        CameraController.pinUnmirrored(c)
        let angle = CameraController.pipelineAngle(front: front)
        if c.videoRotationAngle != angle, c.isVideoRotationAngleSupported(angle) {
            c.videoRotationAngle = angle
        }
    }

    /// A still size the attached camera supports (the photo output keeps the
    /// previous camera's size across an input change otherwise). Call inside a
    /// configuration transaction, after the input is attached.
    private func applyPhotoSize(for dev: AVCaptureDevice) {
        guard session.outputs.contains(photoOutput) else { return }
        // stock-camera default output class (<= 24MP): the 48MP mode doubles
        // per-shot decode/compose cost for no visible gain in an annotated capture
        let dims = dev.activeFormat.supportedMaxPhotoDimensions
        if let pick = dims.last(where: { Int($0.width) * Int($0.height) <= 25_000_000 })
            ?? dims.first {
            photoOutput.maxPhotoDimensions = pick
        }
    }

    /// Lock capture to 30 fps when the active format allows it (setting a
    /// duration outside the format's ranges raises an exception).
    private func lockFrameRate(_ dev: AVCaptureDevice) {
        let ok = dev.activeFormat.videoSupportedFrameRateRanges.contains {
            $0.minFrameRate <= 30 && $0.maxFrameRate >= 30
        }
        guard ok, (try? dev.lockForConfiguration()) != nil else { return }
        dev.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        dev.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        dev.unlockForConfiguration()
    }

    /// Pre-warm the photo pipeline: without prepared settings the FIRST capture
    /// on a camera pays ~0.5-1s of one-time buffer/settings negotiation.
    private func warmPhotoPipeline() {
        guard session.outputs.contains(photoOutput) else { return }
        let warm = AVCapturePhotoSettings()
        warm.maxPhotoDimensions = photoOutput.maxPhotoDimensions
        photoOutput.setPreparedPhotoSettingsArray([warm], completionHandler: nil)
    }

    private func configure() {
        guard session.inputs.isEmpty else {
            session.startRunning()
            applyTorch()
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        guard let dev = findCamera(.back),
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            session.commitConfiguration(); return
        }
        device = dev
        let lens = prepareDevice(dev)
        DispatchQueue.main.async {
            self.isFront = false
            self.displayZoom = 1
            self.lensFactors = lens
        }
        session.addInput(input)
        if !session.outputs.contains(videoOutput) {
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                         kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: queue)
            guard session.canAddOutput(videoOutput) else { session.commitConfiguration(); return }
            session.addOutput(videoOutput)
        }
        if !session.outputs.contains(photoOutput), session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
        }
        applyConnectionSettings(front: false)
        applyPhotoSize(for: dev)
        session.commitConfiguration()
        applyConnectionSettings(front: false)   // no-op when the first one held
        lockFrameRate(dev)
        session.startRunning()
        applyTorch()
        warmPhotoPipeline()
    }

    /// Swap between the back and the front camera without restarting the
    /// session: the input is replaced inside one configuration transaction, so
    /// the preview cuts straight over. `done(switched)` fires on the main queue
    /// once the session has the new camera; `false` means the other camera
    /// could not be attached and nothing changed.
    func switchCamera(_ done: @escaping (Bool) -> Void) {
        queue.async {
            let target: AVCaptureDevice.Position = self.device?.position == .front ? .back : .front
            guard let old = self.session.inputs.first,
                  let dev = self.findCamera(target),
                  let input = try? AVCaptureDeviceInput(device: dev) else {
                DispatchQueue.main.async { done(false) }
                return
            }
            self.session.beginConfiguration()
            self.session.removeInput(old)
            guard self.session.canAddInput(input) else {
                // roll back to the camera we had; its connections are rebuilt,
                // so the output settings go back on as well
                if self.session.canAddInput(old) { self.session.addInput(old) }
                let wasFront = self.device?.position == .front
                if let cur = self.device { self.applyPhotoSize(for: cur) }
                self.session.commitConfiguration()
                self.applyConnectionSettings(front: wasFront)
                DispatchQueue.main.async { done(false) }
                return
            }
            self.device = dev
            let front = target == .front
            let lens = self.prepareDevice(dev)
            self.session.addInput(input)
            self.applyConnectionSettings(front: front)
            self.applyPhotoSize(for: dev)
            self.session.commitConfiguration()
            // again on the committed connection, in case the transaction
            // rebuilt it after the first call
            self.applyConnectionSettings(front: front)
            self.lockFrameRate(dev)
            // drop the outgoing camera's last frame: the detection loop then
            // waits for the first frame of the new camera instead of re-reading
            // a stale one
            self.latestLock.lock()
            self.latest = nil
            self.ageValue = 0
            self.latestLock.unlock()
            self.warmPhotoPipeline()
            DispatchQueue.main.async {
                self.focusRevert?.cancel()   // a pending one-shot focus revert was for the old camera
                self.isFront = front
                if front { self.torchOn = false }   // the front camera has no torch
                self.displayZoom = 1
                self.lensFactors = lens
                done(true)
            }
        }
    }

    func stop() { queue.async { self.session.stopRunning() } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // frame age: capture timestamp vs the session clock NOW. The preview
        // layer shows frames at near-zero age; if the data-output path hands us
        // frames tens of ms old (virtual-device fusion/HDR pipelining), the
        // overlay trails the preview by that much before we spend a single ms.
        var ageMs: Double = 0
        if let clock = session.synchronizationClock {
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            ageMs = max(0, CMTimeGetSeconds(CMTimeSubtract(CMClockGetTime(clock), pts)) * 1000)
        }
        // tag the frame with the camera it came from, read off the connection
        // itself so a frame still in flight across a switch keeps its own tag
        let front = connection.inputPorts.first?.sourceDevicePosition == .front
        latestLock.lock()
        latest = pb
        latestIsFront = front
        frameID &+= 1
        ageValue = ageValue == 0 ? ageMs : ageValue * 0.9 + ageMs * 0.1
        rateCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        if now - rateStart >= 1 {
            rateValue = Double(rateCount) / (now - rateStart)
            rateCount = 0
            rateStart = now
        }
        latestLock.unlock()
    }

    /// Sensor->app frame delivery rate (Hz), 1s window. THE diagnostic for
    /// "labels lag the preview": if this reads below ~29 the capture pipeline
    /// is starving the loop, and no decode optimization can help.
    func deliveryRate() -> Double {
        latestLock.lock(); defer { latestLock.unlock() }
        return rateValue
    }

    /// Smoothed frame age at delivery (ms) - the capture pipeline's own latency.
    func frameAgeMs() -> Double {
        latestLock.lock(); defer { latestLock.unlock() }
        return ageValue
    }

    /// Freshest frame + a monotonically increasing id so consumers can skip
    /// buffers they have already processed, + whether it came from the front
    /// camera (its overlay must be flipped to match the mirrored preview).
    func grabLatest() -> (CVPixelBuffer, UInt64, Bool)? {
        latestLock.lock(); defer { latestLock.unlock() }
        guard let pb = latest else { return nil }
        return (pb, frameID, latestIsFront)
    }
}
