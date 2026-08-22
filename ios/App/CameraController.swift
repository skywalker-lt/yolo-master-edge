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

    /// Pinch-zoom: set the capture zoom factor (focal change happens in the
    /// capture pipeline, so detection frames and preview zoom together).
    /// Returns the clamped factor actually applied.
    @discardableResult
    func setZoom(_ factor: CGFloat) -> CGFloat {
        guard let dev = device else { return 1 }
        let maxZ = min(dev.activeFormat.videoMaxZoomFactor, 16)
        let z = min(max(factor, 1), maxZ)
        if (try? dev.lockForConfiguration()) != nil {
            dev.videoZoomFactor = z
            dev.unlockForConfiguration()
        }
        return z
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            DispatchQueue.main.async { self?.authorized = ok }
            guard ok else { return }
            self?.queue.async { self?.configure() }
        }
    }

    private func configure() {
        guard session.inputs.isEmpty else { session.startRunning(); return }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            session.commitConfiguration(); return
        }
        device = dev
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
