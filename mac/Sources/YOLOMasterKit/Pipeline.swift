// Folder-batch and video pipelines — shared by the CLI and the GUI so both process
// images / folders / videos through the same Detector + annotate path. Progress callbacks
// let a GUI drive a progress bar and live preview.
import Foundation
import CoreGraphics
import AVFoundation
import CoreImage
import CoreVideo

public enum PipelineError: Error { case noVideoTrack, readerInit, writerInit }

public let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp"]
public let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "avi"]

public enum SourceKind: Sendable { case image, folder, video, unknown }

/// image / folder / video / unknown from a URL on disk.
public func classifySource(_ url: URL) -> SourceKind {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return .unknown }
    if isDir.boolValue { return .folder }
    let ext = url.pathExtension.lowercased()
    if videoExtensions.contains(ext) { return .video }
    if imageExtensions.contains(ext) { return .image }
    return .unknown
}

public func listImages(_ dir: URL) -> [URL] {
    ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// ---- source resize (output-resolution knob; independent of the model's fixed inference size) ----
public func resizeExact(_ image: CGImage, _ w: Int, _ h: Int) -> CGImage {
    guard w > 0, h > 0, (w != image.width || h != image.height),
          let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return image }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage() ?? image
}
public func fitLongSide(_ w: Int, _ h: Int, _ longSide: Int) -> (Int, Int) {
    if longSide <= 0 || max(w, h) == 0 { return (w, h) }
    let s = Double(longSide) / Double(max(w, h))
    return (max(1, Int((Double(w) * s).rounded())), max(1, Int((Double(h) * s).rounded())))
}
public func resizeLong(_ image: CGImage, _ longSide: Int) -> CGImage {
    let (nw, nh) = fitLongSide(image.width, image.height, longSide)
    return resizeExact(image, nw, nh)
}

public struct BatchStats: Sendable { public let processed: Int; public let total: Int; public let meanMs: Double }
public struct VideoStats: Sendable {
    public let frames: Int; public let meanMs: Double
    public let outW: Int; public let outH: Int; public let fps: Int
}

/// Run every image in `input`, annotate with the given params, write to `output` (nil = don't save).
/// `progress(done, total, lastAnnotated)` fires per image.
@discardableResult
public func runFolder(_ det: Detector, input: URL, output: URL?,
                      conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode, resize: Int = 0,
                      progress: ((_ done: Int, _ total: Int, _ lastAnnotated: CGImage?) -> Void)? = nil) -> BatchStats {
    let files = listImages(input)
    if let output { try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true) }
    var times: [Double] = []
    for (i, src) in files.enumerated() {
        guard var cg = loadCGImage(src) else { continue }
        if resize > 0 { cg = resizeLong(cg, resize) }
        guard let res = try? det.detect(cg, conf: conf, iou: iou) else { continue }
        times.append(res.inferMs)
        let annotated = annotate(cg, res.detections, names: det.classNames, style: style, label: label)
        if let output, let a = annotated {
            saveCGImage(a, to: output.appendingPathComponent(src.deletingPathExtension().lastPathComponent + ".jpg"))
        }
        progress?(i + 1, files.count, annotated)
    }
    let steady = times.count > 1 ? times[1...].reduce(0, +) / Double(times.count - 1) : (times.first ?? 0)
    return BatchStats(processed: times.count, total: files.count, meanMs: steady)
}

/// Decode `input` video, detect+annotate each frame, encode to `output` (h264, size/fps preserved).
/// `progress(frames, lastAnnotated)` fires periodically.
@discardableResult
public func runVideo(_ det: Detector, input: URL, output: URL,
                     conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode, resize: Int = 0,
                     progress: ((_ frames: Int, _ lastAnnotated: CGImage?) -> Void)? = nil) async throws -> VideoStats {
    let asset = AVURLAsset(url: input)
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else {
        throw PipelineError.noVideoTrack
    }
    let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 0
    let fps = nominalFps > 0 ? nominalFps : 30
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let naturalSize = (try? await track.load(.naturalSize)) ?? .zero
    let disp = naturalSize.applying(transform)
    let natW = Int(abs(disp.width).rounded()), natH = Int(abs(disp.height).rounded())
    let (outW, outH) = resize > 0 ? fitLongSide(natW, natH, resize) : (natW, natH)

    guard let reader = try? AVAssetReader(asset: asset) else { throw PipelineError.readerInit }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false
    reader.add(rout)

    try? FileManager.default.removeItem(at: output)
    let ftype: AVFileType = output.pathExtension.lowercased() == "mov" ? .mov : .mp4
    guard let writer = try? AVAssetWriter(outputURL: output, fileType: ftype) else { throw PipelineError.writerInit }
    let winput = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: outW, AVVideoHeightKey: outH])
    winput.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: winput, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: outW, kCVPixelBufferHeightKey as String: outH])
    writer.add(winput)
    reader.startReading(); writer.startWriting(); writer.startSession(atSourceTime: .zero)

    let cictx = CIContext()
    var n = 0, times: [Double] = []
    while reader.status == .reading {
        guard let sb = rout.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) else { break }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        var ci = CIImage(cvPixelBuffer: pb)
        if !transform.isIdentity {
            ci = ci.transformed(by: transform)
            ci = ci.transformed(by: CGAffineTransform(translationX: -ci.extent.origin.x, y: -ci.extent.origin.y))
        }
        guard var cg = cictx.createCGImage(ci, from: CGRect(x: 0, y: 0, width: CGFloat(natW), height: CGFloat(natH)))
        else { continue }
        if resize > 0 { cg = resizeExact(cg, outW, outH) }
        guard let res = try? det.detect(cg, conf: conf, iou: iou) else { continue }
        times.append(res.inferMs)
        guard let annotated = annotate(cg, res.detections, names: det.classNames, style: style, label: label)
        else { continue }
        guard let pool = adaptor.pixelBufferPool else { continue }
        var opb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &opb)
        guard let dst = opb else { continue }
        CVPixelBufferLockBaseAddress(dst, [])
        if let base = CVPixelBufferGetBaseAddress(dst),
           let c = CGContext(data: base, width: outW, height: outH, bitsPerComponent: 8,
                             bytesPerRow: CVPixelBufferGetBytesPerRow(dst), space: CGColorSpaceCreateDeviceRGB(),
                             bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
            c.draw(annotated, in: CGRect(x: 0, y: 0, width: outW, height: outH))
        }
        CVPixelBufferUnlockBaseAddress(dst, [])
        while !winput.isReadyForMoreMediaData { usleep(2000) }
        adaptor.append(dst, withPresentationTime: pts)
        n += 1
        if n % 10 == 0 { progress?(n, annotated) }
    }
    winput.markAsFinished()
    await writer.finishWriting()
    let mean = times.count > 1 ? times[1...].reduce(0, +) / Double(times.count - 1) : (times.first ?? 0)
    return VideoStats(frames: n, meanMs: mean, outW: outW, outH: outH, fps: Int(fps.rounded()))
}

/// Video duration in seconds (0 if unknown) — for a scrubber range.
public func videoDuration(_ url: URL) async -> Double {
    let d = (try? await AVURLAsset(url: url).load(.duration)) ?? .zero
    return d.seconds.isFinite ? d.seconds : 0
}

/// Decode a single upright frame at `atSeconds` — for the in-app preview/tune finder.
public func extractFrame(_ url: URL, atSeconds t: Double) async -> CGImage? {
    let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .zero
    let time = CMTime(seconds: max(0, t), preferredTimescale: 600)
    return try? await gen.image(at: time).image
}

// ---- two-phase folder flow: infer-all-once (cache candidates) -> tune -> export ----
public struct FolderItem: Sendable { public let url: URL; public let candidates: [Detection] }

/// Aggregate inference timing over a set of forward passes.
public struct InferSummary: Sendable {
    public let count: Int, meanMs: Double, minMs: Double, maxMs: Double, totalMs: Double
    public var fps: Double { meanMs > 0 ? 1000 / meanMs : 0 }
    public init(_ times: [Double]) {
        count = times.count
        totalMs = times.reduce(0, +)
        meanMs = count > 0 ? totalMs / Double(count) : 0
        minMs = times.min() ?? 0
        maxMs = times.max() ?? 0
    }
}

/// Phase 1: forward every image once, caching pre-NMS candidates (conf/iou tuning stays cheap).
/// Returns the cached items + inference timing summary.
public func inferFolder(_ det: Detector, input: URL, confFloor: Float = 0.05,
                        progress: ((_ done: Int, _ total: Int) -> Void)? = nil) -> (items: [FolderItem], summary: InferSummary) {
    let files = listImages(input)
    var out: [FolderItem] = []; out.reserveCapacity(files.count)
    var times: [Double] = []
    for (i, url) in files.enumerated() {
        if let cg = loadCGImage(url), let raw = try? det.forward(cg) {
            out.append(FolderItem(url: url, candidates: det.candidates(raw, confFloor: confFloor)))
            times.append(raw.inferMs)
        }
        progress?(i + 1, files.count)
    }
    return (out, InferSummary(times))
}

/// Phase 3: write annotated images from cached candidates + the tuned params — NO inference.
@discardableResult
public func exportFolderCached(_ items: [FolderItem], output: URL, names: [String],
                               conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode,
                               progress: ((_ done: Int, _ total: Int) -> Void)? = nil) -> Int {
    try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    var written = 0
    for (i, item) in items.enumerated() {
        if let cg = loadCGImage(item.url) {
            let dets = Detector.nms(item.candidates, conf: conf, iou: iou)
            if let a = annotate(cg, dets, names: names, style: style, label: label) {
                saveCGImage(a, to: output.appendingPathComponent(item.url.deletingPathExtension().lastPathComponent + ".jpg"))
                written += 1
            }
        }
        progress?(i + 1, items.count)
    }
    return written
}

// ---- two-phase video flow: infer every frame once (cache candidates) -> scrub/tune -> export ----

/// Phase 1: stream + forward every frame once, caching per-frame candidates + timing.
public func inferVideo(_ det: Detector, input: URL, confFloor: Float = 0.05,
                       progress: ((_ done: Int, _ estTotal: Int) -> Void)? = nil) async throws -> (frames: [[Detection]], summary: InferSummary, fps: Double) {
    let asset = AVURLAsset(url: input)
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else { throw PipelineError.noVideoTrack }
    let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 0
    let fps = nominalFps > 0 ? nominalFps : 30
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let naturalSize = (try? await track.load(.naturalSize)) ?? .zero
    let dur = (try? await asset.load(.duration))?.seconds ?? 0
    let estTotal = Int((dur * Double(fps)).rounded())
    let disp = naturalSize.applying(transform)
    let natW = Int(abs(disp.width).rounded()), natH = Int(abs(disp.height).rounded())
    guard let reader = try? AVAssetReader(asset: asset) else { throw PipelineError.readerInit }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false; reader.add(rout); reader.startReading()
    let cictx = CIContext()
    var frames: [[Detection]] = [], times: [Double] = [], n = 0
    while reader.status == .reading {
        guard let sb = rout.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) else { break }
        var ci = CIImage(cvPixelBuffer: pb)
        if !transform.isIdentity {
            ci = ci.transformed(by: transform)
            ci = ci.transformed(by: CGAffineTransform(translationX: -ci.extent.origin.x, y: -ci.extent.origin.y))
        }
        if let cg = cictx.createCGImage(ci, from: CGRect(x: 0, y: 0, width: CGFloat(natW), height: CGFloat(natH))),
           let raw = try? det.forward(cg) {
            frames.append(det.candidates(raw, confFloor: confFloor)); times.append(raw.inferMs)
        } else { frames.append([]) }
        n += 1
        if n % 4 == 0 { progress?(n, estTotal) }
    }
    return (frames, InferSummary(times), Double(fps))
}

/// Phase 3: re-stream frames in order, apply cached candidates[i] + tuned params, encode. NO inference.
@discardableResult
public func exportVideoCached(input: URL, output: URL, framesCands: [[Detection]], names: [String],
                              conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode, resize: Int = 0,
                              progress: ((_ done: Int, _ total: Int) -> Void)? = nil) async throws -> VideoStats {
    let asset = AVURLAsset(url: input)
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else { throw PipelineError.noVideoTrack }
    let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 0
    let fps = nominalFps > 0 ? nominalFps : 30
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let naturalSize = (try? await track.load(.naturalSize)) ?? .zero
    let disp = naturalSize.applying(transform)
    let natW = Int(abs(disp.width).rounded()), natH = Int(abs(disp.height).rounded())
    let (outW, outH) = resize > 0 ? fitLongSide(natW, natH, resize) : (natW, natH)
    guard let reader = try? AVAssetReader(asset: asset) else { throw PipelineError.readerInit }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false; reader.add(rout)
    try? FileManager.default.removeItem(at: output)
    let ftype: AVFileType = output.pathExtension.lowercased() == "mov" ? .mov : .mp4
    guard let writer = try? AVAssetWriter(outputURL: output, fileType: ftype) else { throw PipelineError.writerInit }
    let winput = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: outW, AVVideoHeightKey: outH])
    winput.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: winput, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: outW, kCVPixelBufferHeightKey as String: outH])
    writer.add(winput); reader.startReading(); writer.startWriting(); writer.startSession(atSourceTime: .zero)
    let cictx = CIContext(); var n = 0; let total = framesCands.count
    while reader.status == .reading {
        guard let sb = rout.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) else { break }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        var ci = CIImage(cvPixelBuffer: pb)
        if !transform.isIdentity {
            ci = ci.transformed(by: transform)
            ci = ci.transformed(by: CGAffineTransform(translationX: -ci.extent.origin.x, y: -ci.extent.origin.y))
        }
        guard var cg = cictx.createCGImage(ci, from: CGRect(x: 0, y: 0, width: CGFloat(natW), height: CGFloat(natH))) else { n += 1; continue }
        if resize > 0 { cg = resizeExact(cg, outW, outH) }
        let dets = Detector.nms(n < framesCands.count ? framesCands[n] : [], conf: conf, iou: iou)
        guard let annotated = annotate(cg, dets, names: names, style: style, label: label), let pool = adaptor.pixelBufferPool else { n += 1; continue }
        var opb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &opb)
        guard let dst = opb else { n += 1; continue }
        CVPixelBufferLockBaseAddress(dst, [])
        if let base = CVPixelBufferGetBaseAddress(dst),
           let c = CGContext(data: base, width: outW, height: outH, bitsPerComponent: 8,
                             bytesPerRow: CVPixelBufferGetBytesPerRow(dst), space: CGColorSpaceCreateDeviceRGB(),
                             bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
            c.draw(annotated, in: CGRect(x: 0, y: 0, width: outW, height: outH))
        }
        CVPixelBufferUnlockBaseAddress(dst, [])
        while !winput.isReadyForMoreMediaData { usleep(2000) }
        adaptor.append(dst, withPresentationTime: pts)
        n += 1
        if n % 10 == 0 { progress?(n, total) }
    }
    winput.markAsFinished(); await writer.finishWriting()
    return VideoStats(frames: n, meanMs: 0, outW: outW, outH: outH, fps: Int(fps.rounded()))
}
