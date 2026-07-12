// yolomaster-coreml — Core ML runner for YOLO-Master-EsMoE-N on Apple Silicon.
//
// Loads a .mlpackage (tensor input [1,3,640,640]), letterboxes each image the same way the C++/ONNX
// pipeline does, runs inference via Core ML, decodes [1, 4+nc, anchors] with multi-label + per-class
// NMS, and writes an annotated image (HUD-style boxes). --source may be a single image OR a folder
// (then --out is treated as an output folder). Compute defaults to CPU+GPU (the ANE can crash on this
// fragmented MoE+attention graph); override with --compute all|cpu.
//
// Build:  swift build -c release --package-path mac
// Run:    .build/release/YOLOMasterCoreML --model EsMoE-N.mlpackage --source img.jpg|dir/ [--out out.jpg|dir/]

import Foundation
import CoreML
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

// ---------- args ----------
func argValue(_ name: String, _ def: String? = nil) -> String? {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count { return a[i + 1] }
    return def
}
func die(_ msg: String, _ code: Int32 = 1) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!); exit(code)
}
func logErr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
guard let modelPath = argValue("--model"), let srcPath = argValue("--source") else {
    die("usage: yolomaster-coreml --model M.mlpackage --source img.jpg|dir/ [--out out.jpg|dir/] " +
        "[--conf 0.25] [--iou 0.5] [--compute cpuAndGPU|all|cpu]", 2)
}
let conf = Float(argValue("--conf", "0.25")!) ?? 0.25
let iouT = CGFloat(Float(argValue("--iou", "0.5")!) ?? 0.5)
let outArg = argValue("--out")                 // file (single) or dir (batch); default chosen per mode
let imgsz = 640
let computeArg = (argValue("--compute", "cpuAndGPU")!).lowercased()
func mlUnits(_ s: String) -> MLComputeUnits {
    switch s { case "all": return .all; case "cpu", "cpuonly": return .cpuOnly; default: return .cpuAndGPU }
}

// ---------- load + compile model (once) ----------
func loadModel(_ path: String) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    let cfg = MLModelConfiguration(); cfg.computeUnits = mlUnits(computeArg)
    if path.hasSuffix(".mlmodelc") { return try MLModel(contentsOf: url, configuration: cfg) }
    let compiled = try MLModel.compileModel(at: url)      // .mlpackage/.mlmodel -> compiled .mlmodelc
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
print("[model] input=\(inputName) output=\(outputName) classes=\(nc) compute=\(computeArg)")

// ---------- image helpers ----------
func loadCGImage(_ path: String) -> CGImage? {
    guard let s = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(s, 0, nil)
}
func letterbox(_ image: CGImage, _ size: Int) -> (px: [UInt8], scale: CGFloat, padX: CGFloat, padY: CGFloat) {
    let w = image.width, h = image.height
    let scale = min(CGFloat(size) / CGFloat(w), CGFloat(size) / CGFloat(h))
    let nw = Int((CGFloat(w) * scale).rounded()), nh = Int((CGFloat(h) * scale).rounded())
    let padX = CGFloat(size - nw) / 2, padY = CGFloat(size - nh) / 2
    var px = [UInt8](repeating: 114, count: size * size * 4)   // 114 gray pad; alpha byte ignored
    px.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: size, height: size,
                                  bitsPerComponent: 8, bytesPerRow: size * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
        ctx.interpolationQuality = .high
        // NO flip: CGContextDrawImage into a bitmap context yields TOP-DOWN pixels (row 0 = top), matching
        // the C++/OpenCV pipeline (imread top-down, no vertical flip in letterbox).
        ctx.draw(image, in: CGRect(x: padX, y: padY, width: CGFloat(nw), height: CGFloat(nh)))
    }
    return (px, scale, padX, padY)
}

struct Det { let cls: Int; let score: Float; let rect: CGRect }
func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let i = a.intersection(b); if i.isNull { return 0 }
    let ia = i.width * i.height
    return ia / (a.width * a.height + b.width * b.height - ia + 1e-6)
}

// ---------- annotation (HUD-style boxes + translucent label pills) ----------
let palette: [CGColor] = [
    (0.98, 0.26, 0.30), (0.20, 0.71, 0.98), (0.16, 0.85, 0.52), (0.99, 0.79, 0.12),
    (0.72, 0.40, 0.98), (0.99, 0.55, 0.18), (0.10, 0.83, 0.80), (0.98, 0.36, 0.66),
    (0.55, 0.82, 0.28), (0.40, 0.52, 0.98),
].map { CGColor(red: CGFloat($0.0), green: CGFloat($0.1), blue: CGFloat($0.2), alpha: 1) }
func labelTextColor(on bg: CGColor) -> CGColor {   // black on light hues, white on dark — readable on any color
    let c = bg.components ?? [0, 0, 0]
    let lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return lum > 0.62 ? CGColor(gray: 0.05, alpha: 1) : CGColor(gray: 1, alpha: 1)
}
func drawAndSave(_ image: CGImage, _ dets: [Det], _ path: String) {
    let w = image.width, h = image.height
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    // No flip (upright save). Context is bottom-up while detections are TOP-DOWN px, so convert box y.
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    ctx.setLineJoin(.round); ctx.setLineCap(.round)
    let lw = max(CGFloat(2), CGFloat(w) / 640)
    let fontSize = max(CGFloat(12), CGFloat(w) / 95)
    let font = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, fontSize, nil)
    for d in dets {
        let color = palette[d.cls % palette.count]
        let box = CGRect(x: d.rect.minX, y: CGFloat(h) - d.rect.maxY, width: d.rect.width, height: d.rect.height)
        // faint fill + thin full outline + bold rounded corner brackets
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
        // translucent rounded label pill "name 0.83", contrast-aware text
        let label = "\(names[d.cls])  \(String(format: "%.2f", d.score))"
        let attr = NSAttributedString(string: label, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): labelTextColor(on: color)])
        let line = CTLineCreateWithAttributedString(attr)
        let tw = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        let padX = fontSize * 0.5, chipH = fontSize + 6, chipW = tw + padX * 2
        var chipY = box.maxY - lw / 2
        if chipY + chipH > CGFloat(h) { chipY = box.maxY - chipH }    // near image top -> put inside
        let chip = CGRect(x: box.minX - lw / 2, y: chipY, width: chipW, height: chipH)
        let chipPath = CGPath(roundedRect: chip, cornerWidth: chipH * 0.28, cornerHeight: chipH * 0.28, transform: nil)
        ctx.addPath(chipPath); ctx.setFillColor(color.copy(alpha: 0.72) ?? color); ctx.fillPath()
        ctx.textPosition = CGPoint(x: chip.minX + padX, y: chipY + (chipH - fontSize) / 2 + fontSize * 0.2)
        CTLineDraw(line, ctx)
    }
    guard let outImg = ctx.makeImage() else { return }
    let type: CFString = path.hasSuffix(".png") ? UTType.png.identifier as CFString : UTType.jpeg.identifier as CFString
    if let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, type, 1, nil) {
        CGImageDestinationAddImage(dest, outImg, nil)
        CGImageDestinationFinalize(dest)
    }
}

// ---------- per-image pipeline; returns model-inference ms (or -1 on skip/fail) ----------
func process(_ srcPath: String, _ outPath: String) -> Double {
    guard let cg = loadCGImage(srcPath) else { logErr("skip (unreadable): \(srcPath)"); return -1 }
    let origW = cg.width, origH = cg.height
    let (raster, scale, padX, padY) = letterbox(cg, imgsz)

    guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: imgsz), NSNumber(value: imgsz)], dataType: .float32)
    else { return -1 }
    do {
        let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        let plane = imgsz * imgsz
        for yy in 0..<imgsz {
            for xx in 0..<imgsz {
                let o = (yy * imgsz + xx) * 4, idx = yy * imgsz + xx
                p[idx] = Float32(raster[o]) / 255            // R
                p[plane + idx] = Float32(raster[o + 1]) / 255 // G
                p[2 * plane + idx] = Float32(raster[o + 2]) / 255 // B
            }
        }
    }

    guard let input = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
    else { return -1 }
    let t0 = Date()
    guard let out = try? model.prediction(from: input) else { logErr("predict failed: \(srcPath)"); return -1 }
    let infMs = Date().timeIntervalSince(t0) * 1000
    guard let y = out.featureValue(for: outputName)?.multiArrayValue, y.shape.count == 3 else {
        logErr("bad output: \(srcPath)"); return -1
    }
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
    let nm = (srcPath as NSString).lastPathComponent
    print("[det] \(nm)  dets=\(keep.count)  infer=\(String(format: "%.1f", infMs))ms")
    drawAndSave(cg, keep, outPath)
    return infMs
}

// ---------- dispatch: single image or folder ----------
let fm = FileManager.default
var isDir: ObjCBool = false
guard fm.fileExists(atPath: srcPath, isDirectory: &isDir) else { die("source not found: \(srcPath)", 4) }
let imgExts: Set<String> = ["jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp"]

if isDir.boolValue {
    let outDir = outArg ?? "preds"
    try? fm.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let items = ((try? fm.contentsOfDirectory(atPath: srcPath)) ?? [])
        .filter { imgExts.contains(($0 as NSString).pathExtension.lowercased()) }
        .sorted()
    if items.isEmpty { die("no images in \(srcPath)", 4) }
    print("[batch] \(items.count) images  \(srcPath) -> \(outDir)/")
    let t0 = Date()
    var times: [Double] = []
    for name in items {
        let src = (srcPath as NSString).appendingPathComponent(name)
        let outp = (outDir as NSString).appendingPathComponent((name as NSString).deletingPathExtension + ".jpg")
        let ms = process(src, outp)
        if ms >= 0 { times.append(ms) }
    }
    let wall = Date().timeIntervalSince(t0)
    // steady-state = mean excluding the first image (compile + warmup outlier)
    let steady = times.count > 1 ? times[1...].reduce(0, +) / Double(times.count - 1) : (times.first ?? 0)
    let fps = steady > 0 ? 1000 / steady : 0
    func f1(_ v: Double) -> String { String(format: "%.1f", v) }
    print("[batch] \(times.count) ok / \(items.count)  |  wall \(f1(wall))s  |  " +
          "model-infer mean \(f1(steady))ms (warmup \(f1(times.first ?? 0))ms) -> \(f1(fps)) img/s steady")
} else {
    _ = process(srcPath, outArg ?? "out.jpg")
    print("[saved] \(outArg ?? "out.jpg")")
}
