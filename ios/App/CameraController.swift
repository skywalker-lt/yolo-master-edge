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
    private let photoOutput = AVCapturePhotoOutput()
    private var photoDone: ((CGImage?) -> Void)?
    private var focusRevert: DispatchWorkItem?

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
            if let c = self.photoOutput.connection(with: .video) { c.videoRotationAngle = 90 }
            self.photoOutput.capturePhoto(with: st, delegate: self)
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        var cg: CGImage?
        if error == nil, let data = photo.fileDataRepresentation(),
           let ui = UIImage(data: data) {
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
        let out = cg, cb = photoDone
        photoDone = nil
        DispatchQueue.main.async { cb?(out) }
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

    private func configure() {
        guard session.inputs.isEmpty else {
            session.startRunning()
            applyTorch()
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        // best virtual multi-cam first: the device auto-switches lenses as the
        // zoom factor crosses its switch-over points (ultra-wide/wide/tele)
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTripleCamera, .builtInDualWideCamera,
                          .builtInDualCamera, .builtInWideAngleCamera],
            mediaType: .video, position: .back)
        guard let dev = discovery.devices.first,
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            session.commitConfiguration(); return
        }
        device = dev
        let switchOvers = dev.virtualDeviceSwitchOverVideoZoomFactors.map { CGFloat(truncating: $0) }
        let hasUW = dev.constituentDevices.contains { $0.deviceType == .builtInUltraWideCamera }
        if dev.isVirtualDevice, hasUW, let so = switchOvers.first {
            zoomBias = so
        }
        // display-unit native lens stops: 1x wide, 0.5x ultra-wide when present,
        // every further switch-over (tele) above
        var stops: [CGFloat] = [1]
        if hasUW { stops.append(1 / zoomBias) }
        stops.append(contentsOf: (hasUW ? Array(switchOvers.dropFirst()) : switchOvers)
            .map { $0 / zoomBias })
        let lens = stops.sorted()
        if (try? dev.lockForConfiguration()) != nil {
            if dev.isVirtualDevice {
                // no focus/exposure restrictions on lens hand-off: without this
                // (and with a stale focus lock) the tele never engages
                dev.setPrimaryConstituentDeviceSwitchingBehavior(
                    .auto, restrictedSwitchingBehaviorConditions: [])
            }
            dev.videoZoomFactor = zoomBias    // start on the wide lens = 1x
            dev.unlockForConfiguration()
        }
        DispatchQueue.main.async {
            self.displayZoom = 1
            self.lensFactors = lens
        }
        session.addInput(input)
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                             kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(out) else { session.commitConfiguration(); return }
        session.addOutput(out)
        out.connection(with: .video)?.videoRotationAngle = 90   // portrait
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
            if let dims = dev.activeFormat.supportedMaxPhotoDimensions.last {
                photoOutput.maxPhotoDimensions = dims   // full stills off a video format
            }
        }
        session.commitConfiguration()
        session.startRunning()
        applyTorch()
    }

    func stop() { queue.async { self.session.stopRunning() } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestLock.lock(); latest = pb; frameID &+= 1; latestLock.unlock()
    }

    /// Freshest frame + a monotonically increasing id so consumers can skip
    /// buffers they have already processed.
    func grabLatest() -> (CVPixelBuffer, UInt64)? {
        latestLock.lock(); defer { latestLock.unlock() }
        guard let pb = latest else { return nil }
        return (pb, frameID)
    }
}
