// AVFoundation capture -> CVPixelBuffer stream for the Kit's forward(pixelBuffer:).
// iOS-specific by nature (the mac app's Camera.swift is AVCaptureScreen-flavored);
// everything downstream of the buffer is shared Kit code.
import AVFoundation
import CoreVideo
import Foundation

final class CameraController: NSObject, ObservableObject,
                              AVCaptureVideoDataOutputSampleBufferDelegate {
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
    @Published var torchOn = false

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
    /// `captureDevicePointConverted`). Also re-anchors exposure there.
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
        if dev.isVirtualDevice,
           dev.constituentDevices.contains(where: { $0.deviceType == .builtInUltraWideCamera }),
           let so = dev.virtualDeviceSwitchOverVideoZoomFactors.first {
            zoomBias = CGFloat(truncating: so)
        }
        if (try? dev.lockForConfiguration()) != nil {
            dev.videoZoomFactor = zoomBias    // start on the wide lens = 1x
            dev.unlockForConfiguration()
        }
        DispatchQueue.main.async { self.displayZoom = 1 }
        session.addInput(input)
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                             kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(out) else { session.commitConfiguration(); return }
        session.addOutput(out)
        out.connection(with: .video)?.videoRotationAngle = 90   // portrait
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
