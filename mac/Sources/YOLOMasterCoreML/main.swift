// yolomaster-coreml — Core ML runner for YOLO-Master-EsMoE-N on Apple Silicon.
//
// --source: a single image, a folder of images, OR a video (.mp4/.mov/.m4v). Letterboxes exactly like
// the C++/ONNX pipeline, runs Core ML, decodes [1,4+nc,anchors] with multi-label + per-class NMS, and
// writes HUD-style annotations. Modes:
//   image      -> annotated image (--out out.jpg)
//   folder     -> annotated folder (--out preds/), + batch timing summary
//   video      -> annotated video  (--out out.mp4), preserves size/fps
//   --benchmark-> model-only latency benchmark (no decode/draw/save), percentiles + img/s
// Compute defaults to CPU+GPU (the ANE can crash on this fragmented MoE+attention graph); --compute all|cpu.
//
// Build:  swift build -c release --package-path mac

import Foundation
import CoreML
import CoreGraphics
import CoreText
import CoreImage
import CoreVideo
import AVFoundation
import ImageIO
import UniformTypeIdentifiers

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
        "[--conf 0.25] [--iou 0.5] [--compute cpuAndGPU|all|cpu] [--benchmark [--iters 200]] [--no-save]", 2)
}
let conf = Float(argValue("--conf", "0.25")!) ?? 0.25
let iouT = CGFloat(Float(argValue("--iou", "0.5")!) ?? 0.5)
let outArg = argValue("--out")
let computeArg = (argValue("--compute", "cpuAndGPU")!).lowercased()
let benchmark = hasFlag("--benchmark")
let noSave = hasFlag("--no-save")
let iters = Int(argValue("--iters", "200")!) ?? 200
let resize = Int(argValue("--resize", "0")!) ?? 0   // resize source so long side = N px (0 = keep original)
func mlUnits(_ s: String) -> MLComputeUnits {
    switch s { case "all": return .all; case "cpu", "cpuonly": return .cpuOnly; default: return .cpuAndGPU }
}

// ---------- load + compile model (once) ----------
func loadModel(_ path: String) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    let cfg = MLModelConfiguration(); cfg.computeUnits = mlUnits(computeArg)
    if path.hasSuffix(".mlmodelc") { return try MLModel(contentsOf: url, configuration: cfg) }
    let compiled = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiled, configuration: cfg)
}
let model: MLModel
do { model = try loadModel(modelPath) } catch { die("model load failed: \(error)", 3) }

let inputName = model.modelDescription.inputDescriptionsByName.keys.sorted().first ?? "images"
let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
let names = meta["names"]?.split(separator: ",").map(String.init)
    ?? ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
let outputName = meta["output"] ?? model.modelDescription.outputDescriptionsByName.keys.sorted().first ?? "output0"
let nc = names.count
// Input resolution is FIXED at export time — read it from the model ([1,3,H,W]) so preprocessing always
// matches the .mlpackage. To use a different resolution, re-export: `export_coreml.py --imgsz N`.
let imgsz: Int = {
    if let shape = model.modelDescription.inputDescriptionsByName[inputName]?.multiArrayConstraint?.shape,
       shape.count >= 4, shape[2].intValue > 0 { return shape[2].intValue }
    if let s = meta["imgsz"], let v = Int(s), v > 0 { return v }
    return 640
}()
print("[model] input=\(inputName) [\(imgsz)x\(imgsz)] output=\(outputName) classes=\(nc) compute=\(computeArg)")

// ---------- preprocess ----------
func loadCGImage(_ path: String) -> CGImage? {
    guard let s = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(s, 0, nil)
}
func letterbox(_ image: CGImage, _ size: Int) -> (px: [UInt8], scale: CGFloat, padX: CGFloat, padY: CGFloat) {
    let w = image.width, h = image.height
    let scale = min(CGFloat(size) / CGFloat(w), CGFloat(size) / CGFloat(h))
    let nw = Int((CGFloat(w) * scale).rounded()), nh = Int((CGFloat(h) * scale).rounded())
    let padX = CGFloat(size - nw) / 2, padY = CGFloat(size - nh) / 2
    var px = [UInt8](repeating: 114, count: size * size * 4)
    px.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: padX, y: padY, width: CGFloat(nw), height: CGFloat(nh)))  // no flip -> top-down
    }
    return (px, scale, padX, padY)
}
// Resize the SOURCE (independent of the model's fixed inference size) — for smaller/faster output.
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
func fillInput(_ raster: [UInt8]) -> MLDictionaryFeatureProvider? {
    guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: imgsz), NSNumber(value: imgsz)], dataType: .float32)
    else { return nil }
    let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
    let plane = imgsz * imgsz
    raster.withUnsafeBufferPointer { rb in
        for yy in 0..<imgsz {
            for xx in 0..<imgsz {
                let o = (yy * imgsz + xx) * 4, idx = yy * imgsz + xx
                p[idx] = Float32(rb[o]) / 255
                p[plane + idx] = Float32(rb[o + 1]) / 255
                p[2 * plane + idx] = Float32(rb[o + 2]) / 255
            }
        }
    }
    return try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
}

// ---------- decode + NMS ----------
struct Det { let cls: Int; let score: Float; let rect: CGRect }
func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let i = a.intersection(b); if i.isNull { return 0 }
    let ia = i.width * i.height
    return ia / (a.width * a.height + b.width * b.height - ia + 1e-6)
}
func decodeAndNMS(_ y: MLMultiArray, _ origW: Int, _ origH: Int, _ scale: CGFloat, _ padX: CGFloat, _ padY: CGFloat) -> [Det] {
    let na = y.shape[2].intValue
    let s1 = y.strides[1].intValue, s2 = y.strides[2].intValue
    var dets: [Det] = []
    func decodeAnchors(_ at: (Int, Int) -> Float32) {
        for a in 0..<na {
            let cx = CGFloat(at(0, a)), cy = CGFloat(at(1, a)), bw = CGFloat(at(2, a)), bh = CGFloat(at(3, a))
            for c in 0..<nc {
                let s = at(4 + c, a)
                if s <= conf { continue }
                var x1 = (cx - bw / 2 - padX) / scale, y1 = (cy - bh / 2 - padY) / scale
                var x2 = (cx + bw / 2 - padX) / scale, y2 = (cy + bh / 2 - padY) / scale
                x1 = max(0, min(CGFloat(origW), x1)); x2 = max(0, min(CGFloat(origW), x2))
                y1 = max(0, min(CGFloat(origH), y1)); y2 = max(0, min(CGFloat(origH), y2))
                if x2 > x1 && y2 > y1 {
                    dets.append(Det(cls: c, score: s, rect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)))
                }
            }
        }
    }
    if y.dataType == .float16 {
        y.withUnsafeBufferPointer(ofType: Float16.self) { buf in
            guard let yp = buf.baseAddress else { return }
            decodeAnchors { c, a in Float32(yp[c * s1 + a * s2]) }
        }
    } else {
        y.withUnsafeBufferPointer(ofType: Float32.self) { buf in
            guard let yp = buf.baseAddress else { return }
            decodeAnchors { c, a in yp[c * s1 + a * s2] }
        }
    }
    dets.sort { $0.score > $1.score }
    var keep: [Det] = []
    for d in dets {
        if keep.count >= 300 { break }
        if !keep.contains(where: { $0.cls == d.cls && iou($0.rect, d.rect) > iouT }) { keep.append(d) }
    }
    return keep
}

// ---------- annotation (HUD boxes + translucent label pills) ----------
let palette: [CGColor] = [
    (0.98, 0.26, 0.30), (0.20, 0.71, 0.98), (0.16, 0.85, 0.52), (0.99, 0.79, 0.12),
    (0.72, 0.40, 0.98), (0.99, 0.55, 0.18), (0.10, 0.83, 0.80), (0.98, 0.36, 0.66),
    (0.55, 0.82, 0.28), (0.40, 0.52, 0.98),
].map { CGColor(red: CGFloat($0.0), green: CGFloat($0.1), blue: CGFloat($0.2), alpha: 1) }
func labelTextColor(on bg: CGColor) -> CGColor {
    let c = bg.components ?? [0, 0, 0]
    let lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return lum > 0.62 ? CGColor(gray: 0.05, alpha: 1) : CGColor(gray: 1, alpha: 1)
}
func annotate(_ image: CGImage, _ dets: [Det]) -> CGImage? {
    let w = image.width, h = image.height
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))  // no flip; boxes convert y = h - topY
    ctx.setLineJoin(.round); ctx.setLineCap(.round)
    let lw = max(CGFloat(2), CGFloat(w) / 640)
    let fontSize = max(CGFloat(12), CGFloat(w) / 95)
    let font = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, fontSize, nil)
    for d in dets {
        let color = palette[d.cls % palette.count]
        let box = CGRect(x: d.rect.minX, y: CGFloat(h) - d.rect.maxY, width: d.rect.width, height: d.rect.height)
        ctx.addRect(box); ctx.setFillColor(color.copy(alpha: 0.08) ?? color); ctx.fillPath()
        ctx.addRect(box); ctx.setStrokeColor(color.copy(alpha: 0.35) ?? color); ctx.setLineWidth(lw * 0.6); ctx.strokePath()
        let arm = min(min(box.width, box.height) * 0.28, lw * 22)
        ctx.setStrokeColor(color); ctx.setLineWidth(lw * 1.4)
        for (cx, cy, sx, sy) in [(box.minX, box.minY, 1.0, 1.0), (box.maxX, box.minY, -1.0, 1.0),
                                 (box.minX, box.maxY, 1.0, -1.0), (box.maxX, box.maxY, -1.0, -1.0)] {
            ctx.move(to: CGPoint(x: cx + arm * CGFloat(sx), y: cy))
            ctx.addLine(to: CGPoint(x: cx, y: cy))
            ctx.addLine(to: CGPoint(x: cx, y: cy + arm * CGFloat(sy)))
        }
        ctx.strokePath()
        let label = "\(names[d.cls])  \(String(format: "%.2f", d.score))"
        let attr = NSAttributedString(string: label, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): labelTextColor(on: color)])
        let line = CTLineCreateWithAttributedString(attr)
        let tw = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        let padX = fontSize * 0.5, chipH = fontSize + 6, chipW = tw + padX * 2
        var chipY = box.maxY - lw / 2
        if chipY + chipH > CGFloat(h) { chipY = box.maxY - chipH }
        let chip = CGRect(x: box.minX - lw / 2, y: chipY, width: chipW, height: chipH)
        let chipPath = CGPath(roundedRect: chip, cornerWidth: chipH * 0.28, cornerHeight: chipH * 0.28, transform: nil)
        ctx.addPath(chipPath); ctx.setFillColor(color.copy(alpha: 0.72) ?? color); ctx.fillPath()
        ctx.textPosition = CGPoint(x: chip.minX + padX, y: chipY + (chipH - fontSize) / 2 + fontSize * 0.2)
        CTLineDraw(line, ctx)
    }
    return ctx.makeImage()
}
func saveCGImage(_ img: CGImage, _ path: String) {
    let type: CFString = path.hasSuffix(".png") ? UTType.png.identifier as CFString : UTType.jpeg.identifier as CFString
    if let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, type, 1, nil) {
        CGImageDestinationAddImage(dest, img, nil)
        CGImageDestinationFinalize(dest)
    }
}

// ---------- per-image pipeline; returns model-inference ms (or -1 on skip/fail) ----------
func process(_ srcPath: String, _ outPath: String) -> Double {
    guard var cg = loadCGImage(srcPath) else { logErr("skip (unreadable): \(srcPath)"); return -1 }
    if resize > 0 { cg = resizeLong(cg, resize) }              // output-resolution knob (not the model's)
    let (raster, scale, padX, padY) = letterbox(cg, imgsz)
    guard let input = fillInput(raster) else { return -1 }
    let t0 = Date()
    guard let out = try? model.prediction(from: input) else { logErr("predict failed: \(srcPath)"); return -1 }
    let infMs = Date().timeIntervalSince(t0) * 1000
    guard let y = out.featureValue(for: outputName)?.multiArrayValue, y.shape.count == 3 else {
        logErr("bad output: \(srcPath)"); return -1
    }
    let dets = decodeAndNMS(y, cg.width, cg.height, scale, padX, padY)
    print("[det] \((srcPath as NSString).lastPathComponent)  dets=\(dets.count)  infer=\(f1(infMs))ms")
    if !noSave, let annotated = annotate(cg, dets) { saveCGImage(annotated, outPath) }
    return infMs
}

// ---------- benchmark: model-only latency (no decode/draw/save) ----------
func runBenchmark(_ paths: [String]) {
    guard let cg0 = loadCGImage(paths[0]) else { die("bench: cannot read \(paths[0])", 4) }
    let (r0, _, _, _) = letterbox(cg0, imgsz)
    guard let in0 = fillInput(r0) else { die("bench: input build failed", 5) }
    for _ in 0..<10 { _ = try? model.prediction(from: in0) }          // warmup
    var times: [Double] = []
    if paths.count == 1 {
        for _ in 0..<iters { let t = Date(); _ = try? model.prediction(from: in0); times.append(Date().timeIntervalSince(t) * 1000) }
    } else {
        for p in paths {
            guard let cg = loadCGImage(p) else { continue }
            let (r, _, _, _) = letterbox(cg, imgsz)
            guard let inp = fillInput(r) else { continue }
            let t = Date(); _ = try? model.prediction(from: inp); times.append(Date().timeIntervalSince(t) * 1000)
        }
    }
    times.sort()
    func pct(_ p: Double) -> Double { times.isEmpty ? 0 : times[min(times.count - 1, Int(p * Double(times.count)))] }
    let mean = times.reduce(0, +) / Double(max(times.count, 1))
    print("[bench] n=\(times.count) compute=\(computeArg) imgsz=\(imgsz)")
    print("[bench] latency ms:  mean \(String(format: "%.2f", mean))  min \(String(format: "%.2f", times.first ?? 0))" +
          "  p50 \(String(format: "%.2f", pct(0.5)))  p90 \(String(format: "%.2f", pct(0.9)))  p99 \(String(format: "%.2f", pct(0.99)))")
    print("[bench] throughput:  \(f1(mean > 0 ? 1000 / mean : 0)) img/s (model-only)")
}

// ---------- video: decode -> detect -> annotate -> encode ----------
func processVideo(_ srcPath: String, _ outPath: String) {
    let asset = AVURLAsset(url: URL(fileURLWithPath: srcPath))
    guard let track = asset.tracks(withMediaType: .video).first else { die("no video track: \(srcPath)", 4) }
    let fps = track.nominalFrameRate > 0 ? track.nominalFrameRate : 30
    let transform = track.preferredTransform
    let disp = track.naturalSize.applying(transform)
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
        let (raster, scale, padX, padY) = letterbox(cg, imgsz)
        guard let input = fillInput(raster) else { continue }
        let ts = Date()
        guard let out = try? model.prediction(from: input) else { continue }
        times.append(Date().timeIntervalSince(ts) * 1000)
        guard let y = out.featureValue(for: outputName)?.multiArrayValue, y.shape.count == 3,
              let annotated = annotate(cg, decodeAndNMS(y, cg.width, cg.height, scale, padX, padY)) else { continue }
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
    processVideo(srcPath, outArg ?? "out.mp4")
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
