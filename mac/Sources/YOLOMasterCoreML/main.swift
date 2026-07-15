// yolomaster-coreml — command-line Core ML runner for YOLO-Master detectors.
//
// Thin CLI over YOLOMasterKit (the shared inference backend). --source: a single image, a folder
// of images, OR a video (.mp4/.mov/.m4v). Modes:
//   image      -> annotated image (--out out.jpg)
//   folder     -> annotated folder (--out preds/), + batch timing summary
//   video      -> annotated video  (--out out.mp4), preserves size/fps
//   --benchmark-> model-only latency benchmark (no decode/draw/save), percentiles + img/s
// Compute defaults to CPU+GPU (the ANE can crash on this fragmented MoE+attention graph); --compute all|cpu.
//
// Build:  swift build -c release --package-path mac
import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import AVFoundation
import YOLOMasterKit

// ---------- args ----------
func argValue(_ name: String, _ def: String? = nil) -> String? {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count { return a[i + 1] }
    return def
}
func hasFlag(_ name: String) -> Bool { CommandLine.arguments.contains(name) }
func die(_ msg: String, _ code: Int32 = 1) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!); exit(code)
}
func logErr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
func f1(_ v: Double) -> String { String(format: "%.1f", v) }
guard let modelPath = argValue("--model"), let srcPath = argValue("--source") else {
    die("usage: yolomaster-coreml --model M.mlpackage --source img|dir/|vid.mp4 [--out o] " +
        "[--conf 0.25] [--iou 0.5] [--compute cpuAndGPU|all|cpu] [--style hud|solid|neon] " +
        "[--label full|min|off] [--resize N] [--benchmark [--iters 200]] [--no-save]", 2)
}
let conf = Float(argValue("--conf", "0.25")!) ?? 0.25
let iouT = CGFloat(Float(argValue("--iou", "0.5")!) ?? 0.5)
let outArg = argValue("--out")
let compute = ComputeMode(argValue("--compute", "cpuAndGPU")!)
let benchmark = hasFlag("--benchmark")
let noSave = hasFlag("--no-save")
let iters = Int(argValue("--iters", "200")!) ?? 200
let resize = Int(argValue("--resize", "0")!) ?? 0   // resize source so long side = N px (0 = keep original)
let boxStyle = BoxStyle(rawValue: (argValue("--style", "hud")!).lowercased()) ?? .hud
let labelMode = LabelMode(rawValue: (argValue("--label", "full")!).lowercased()) ?? .full

// ---------- backend (shared) ----------
let detector: Detector
do { detector = try Detector(modelURL: URL(fileURLWithPath: modelPath), compute: compute) }
catch { die("model load failed: \(error)", 3) }
print("[model] \(detector.summary)")
let imgsz = detector.imgsz

// ---------- source-resize (independent of the model's fixed inference size) ----------
func resizeExact(_ image: CGImage, _ w: Int, _ h: Int) -> CGImage {
    guard w > 0, h > 0, (w != image.width || h != image.height),
          let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return image }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))   // no flip -> upright
    return ctx.makeImage() ?? image
}
func fitLongSide(_ w: Int, _ h: Int, _ longSide: Int) -> (Int, Int) {
    if longSide <= 0 || max(w, h) == 0 { return (w, h) }
    let s = Double(longSide) / Double(max(w, h))
    return (max(1, Int((Double(w) * s).rounded())), max(1, Int((Double(h) * s).rounded())))
}
func resizeLong(_ image: CGImage, _ longSide: Int) -> CGImage {
    let (nw, nh) = fitLongSide(image.width, image.height, longSide)
    return resizeExact(image, nw, nh)
}

// ---------- per-image pipeline; returns model-inference ms (or -1 on skip/fail) ----------
func process(_ srcPath: String, _ outPath: String) -> Double {
    guard var cg = loadCGImage(URL(fileURLWithPath: srcPath)) else { logErr("skip (unreadable): \(srcPath)"); return -1 }
    if resize > 0 { cg = resizeLong(cg, resize) }              // output-resolution knob (not the model's)
    guard let res = try? detector.detect(cg, conf: conf, iou: iouT) else { logErr("predict failed: \(srcPath)"); return -1 }
    print("[det] \((srcPath as NSString).lastPathComponent)  dets=\(res.detections.count)  infer=\(f1(res.inferMs))ms")
    if !noSave, let annotated = annotate(cg, res.detections, names: detector.classNames, style: boxStyle, label: labelMode) {
        saveCGImage(annotated, to: URL(fileURLWithPath: outPath))
    }
    return res.inferMs
}

// ---------- benchmark: model-only latency (no decode/draw/save) ----------
func runBenchmark(_ paths: [String]) {
    guard let cg0 = loadCGImage(URL(fileURLWithPath: paths[0])) else { die("bench: cannot read \(paths[0])", 4) }
    for _ in 0..<10 { _ = try? detector.inferOnly(cg0) }           // warmup
    var times: [Double] = []
    if paths.count == 1 {
        for _ in 0..<iters { if let t = try? detector.inferOnly(cg0) { times.append(t) } }
    } else {
        for p in paths {
            guard let cg = loadCGImage(URL(fileURLWithPath: p)) else { continue }
            if let t = try? detector.inferOnly(cg) { times.append(t) }
        }
    }
    times.sort()
    func pct(_ p: Double) -> Double { times.isEmpty ? 0 : times[min(times.count - 1, Int(p * Double(times.count)))] }
    let mean = times.reduce(0, +) / Double(max(times.count, 1))
    print("[bench] n=\(times.count) compute=\(compute.rawValue) imgsz=\(imgsz)")
    print("[bench] latency ms:  mean \(String(format: "%.2f", mean))  min \(String(format: "%.2f", times.first ?? 0))" +
          "  p50 \(String(format: "%.2f", pct(0.5)))  p90 \(String(format: "%.2f", pct(0.9)))  p99 \(String(format: "%.2f", pct(0.99)))")
    print("[bench] throughput:  \(f1(mean > 0 ? 1000 / mean : 0)) img/s (model-only)")
}

// ---------- video: decode -> detect -> annotate -> encode ----------
func processVideo(_ srcPath: String, _ outPath: String) async {
    let asset = AVURLAsset(url: URL(fileURLWithPath: srcPath))
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else {
        die("no video track: \(srcPath)", 4)
    }
    let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 0
    let fps = nominalFps > 0 ? nominalFps : 30
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let naturalSize = (try? await track.load(.naturalSize)) ?? .zero
    let disp = naturalSize.applying(transform)
    let natW = Int(abs(disp.width).rounded()), natH = Int(abs(disp.height).rounded())
    let (outW, outH) = resize > 0 ? fitLongSide(natW, natH, resize) : (natW, natH)   // output-resolution knob

    guard let reader = try? AVAssetReader(asset: asset) else { die("reader init failed", 5) }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false
    reader.add(rout)

    let outURL = URL(fileURLWithPath: outPath)
    try? FileManager.default.removeItem(at: outURL)
    let ftype: AVFileType = outPath.hasSuffix(".mov") ? .mov : .mp4
    guard let writer = try? AVAssetWriter(outputURL: outURL, fileType: ftype) else { die("writer init failed", 5) }
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
    let t0 = Date()
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
        guard let res = try? detector.detect(cg, conf: conf, iou: iouT) else { continue }
        times.append(res.inferMs)
        guard let annotated = annotate(cg, res.detections, names: detector.classNames, style: boxStyle, label: labelMode)
        else { continue }
        // draw annotated CGImage into a BGRA pixel buffer, append at the source timestamp
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
        if n % 60 == 0 { logErr("  \(n) frames…") }
    }
    winput.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
    if writer.status == .failed { die("video write failed: \(writer.error?.localizedDescription ?? "?")", 5) }
    let wall = Date().timeIntervalSince(t0)
    let mean = times.count > 1 ? times[1...].reduce(0, +) / Double(times.count - 1) : (times.first ?? 0)
    print("[video] \(n) frames -> \(outPath)  (\(outW)x\(outH) @\(Int(fps.rounded()))fps)")
    print("[video] wall \(f1(wall))s  |  model-infer mean \(f1(mean))ms -> \(f1(mean > 0 ? 1000 / mean : 0)) fps (model-only)")
}

// ---------- dispatch ----------
let fm = FileManager.default
var isDir: ObjCBool = false
guard fm.fileExists(atPath: srcPath, isDirectory: &isDir) else { die("source not found: \(srcPath)", 4) }
let imgExts: Set<String> = ["jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp"]
let vidExts: Set<String> = ["mp4", "mov", "m4v", "avi"]
let ext = (srcPath as NSString).pathExtension.lowercased()

func listImages(_ dir: String) -> [String] {
    ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
        .filter { imgExts.contains(($0 as NSString).pathExtension.lowercased()) }.sorted()
        .map { (dir as NSString).appendingPathComponent($0) }
}

if benchmark {
    let paths = isDir.boolValue ? listImages(srcPath) : [srcPath]
    if paths.isEmpty { die("benchmark: no images in \(srcPath)", 4) }
    runBenchmark(paths)
} else if !isDir.boolValue && vidExts.contains(ext) {
    await processVideo(srcPath, outArg ?? "out.mp4")
} else if isDir.boolValue {
    let outDir = outArg ?? "preds"
    try? fm.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let items = ((try? fm.contentsOfDirectory(atPath: srcPath)) ?? [])
        .filter { imgExts.contains(($0 as NSString).pathExtension.lowercased()) }.sorted()
    if items.isEmpty { die("no images in \(srcPath)", 4) }
    print("[batch] \(items.count) images  \(srcPath) -> \(outDir)/\(noSave ? "  (--no-save)" : "")")
    let t0 = Date()
    var times: [Double] = []
    for name in items {
        let src = (srcPath as NSString).appendingPathComponent(name)
        let outp = (outDir as NSString).appendingPathComponent((name as NSString).deletingPathExtension + ".jpg")
        let ms = process(src, outp); if ms >= 0 { times.append(ms) }
    }
    let wall = Date().timeIntervalSince(t0)
    let steady = times.count > 1 ? times[1...].reduce(0, +) / Double(times.count - 1) : (times.first ?? 0)
    print("[batch] \(times.count) ok / \(items.count)  |  wall \(f1(wall))s  |  " +
          "model-infer mean \(f1(steady))ms (warmup \(f1(times.first ?? 0))ms) -> \(f1(steady > 0 ? 1000 / steady : 0)) img/s steady")
} else {
    _ = process(srcPath, outArg ?? "out.jpg")
    if !noSave { print("[saved] \(outArg ?? "out.jpg")") }
}
