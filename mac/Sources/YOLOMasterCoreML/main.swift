// yolomaster-coreml — first Core ML runner for YOLO-Master-EsMoE-N on Apple Silicon.
//
// Loads a .mlpackage (tensor input [1,3,640,640]), letterboxes an image the same way the
// C++/ONNX pipeline does, runs inference on the ANE/GPU via Core ML, decodes [1, 4+nc, anchors]
// with multi-label + per-class NMS, and writes an annotated image. Core ML picks ANE/GPU/CPU
// per layer (computeUnits = .all); on Apple Silicon UMA the letterbox/decode share memory with
// the model with no host<->device copies.
//
// Build:  swift build -c release
// Run:    .build/release/YOLOMasterCoreML --model EsMoE-N.mlpackage --source img.jpg --out out.jpg

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
guard let modelPath = argValue("--model"), let srcPath = argValue("--source") else {
    die("usage: yolomaster-coreml --model M.mlpackage --source img.jpg [--conf 0.25] [--iou 0.5] [--out out.jpg]", 2)
}
let conf = Float(argValue("--conf", "0.25")!) ?? 0.25
let iouT = CGFloat(Float(argValue("--iou", "0.5")!) ?? 0.5)
let outPath = argValue("--out", "out.jpg")!
let imgsz = 640
// The ANE can HARD-CRASH on this MoE+attention graph (heavy fragmentation), and that crash is native —
// try? can't catch it. Default to CPU+GPU; use --compute all to opt back into the ANE, cpu for CPU-only.
let computeArg = (argValue("--compute", "cpuAndGPU")!).lowercased()
func mlUnits(_ s: String) -> MLComputeUnits {
    switch s { case "all": return .all; case "cpu", "cpuonly": return .cpuOnly; default: return .cpuAndGPU }
}
func logErr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

// ---------- load + compile model ----------
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

// ---------- image -> letterboxed raster (top-down RGBX, 114 pad) ----------
func loadCGImage(_ path: String) -> CGImage? {
    guard let s = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(s, 0, nil)
}
guard let cg = loadCGImage(srcPath) else { die("cannot read image: \(srcPath)", 4) }
let origW = cg.width, origH = cg.height

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
        // NO flip: CGContextDrawImage into a bitmap context yields TOP-DOWN pixels (row 0 = top), which
        // is what the fill loop and the model expect — matches the C++/OpenCV pipeline (imread is top-down,
        // letterbox does no vertical flip). The earlier translate/scale flip fed the model an upside-down
        // image (that Swift path was never testable until the .mlpackage converted).
        ctx.draw(image, in: CGRect(x: padX, y: padY, width: CGFloat(nw), height: CGFloat(nh)))
    }
    return (px, scale, padX, padY)
}
let (raster, scale, padX, padY) = letterbox(cg, imgsz)

// ---------- fill MLMultiArray [1,3,H,W] with RGB/255 ----------
let arr = try! MLMultiArray(shape: [1, 3, NSNumber(value: imgsz), NSNumber(value: imgsz)], dataType: .float32)
do {
    let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
    let plane = imgsz * imgsz
    for y in 0..<imgsz {
        for x in 0..<imgsz {
            let o = (y * imgsz + x) * 4
            let idx = y * imgsz + x
            p[0 * plane + idx] = Float32(raster[o + 0]) / 255   // R
            p[1 * plane + idx] = Float32(raster[o + 1]) / 255   // G
            p[2 * plane + idx] = Float32(raster[o + 2]) / 255   // B
        }
    }
}

// ---------- predict ----------
let input = try! MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
logErr("[infer] running prediction (compute=\(computeArg))…")
let t0 = Date()
guard let out = try? model.prediction(from: input) else { die("prediction failed", 5) }
let infMs = Date().timeIntervalSince(t0) * 1000
guard let y = out.featureValue(for: outputName)?.multiArrayValue else { die("no output '\(outputName)'", 5) }

// output [1, 4+nc, anchors]
guard y.shape.count == 3 else { die("unexpected output rank \(y.shape.count) (want 3: [1,C,anchors])", 5) }
let na = y.shape[2].intValue
let s1 = y.strides[1].intValue, s2 = y.strides[2].intValue
if y.shape[1].intValue != 4 + nc {
    logErr("warn: output channels \(y.shape[1]) != 4+nc \(4 + nc)")
}
logErr("[infer] prediction ok in \(String(format: "%.1f", infMs))ms, shape=\(y.shape.map { $0.intValue }); decoding…")

// ---------- decode (multi-label: one det per class>conf per anchor) ----------
struct Det { let cls: Int; let score: Float; let rect: CGRect }
var dets: [Det] = []
// `at(c,a)` returns Float32; the mlprogram output is Float16 by default (Core ML fp16 compute), so
// read the buffer at its real dtype and widen. withUnsafeBufferPointer is the SAFE accessor (raw
// .dataPointer can be invalid for a non-CPU/non-contiguous backing); strides keep it layout-agnostic.
func decodeAnchors(_ at: (Int, Int) -> Float32) {
    for a in 0..<na {
        let cx = CGFloat(at(0, a)), cy = CGFloat(at(1, a)), bw = CGFloat(at(2, a)), bh = CGFloat(at(3, a))
        for c in 0..<nc {
            let s = at(4 + c, a)
            if s <= conf { continue }
            // letterbox px -> original px
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

// ---------- per-class greedy NMS (cap 300) ----------
func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let i = a.intersection(b); if i.isNull { return 0 }
    let ia = i.width * i.height
    return ia / (a.width * a.height + b.width * b.height - ia + 1e-6)
}
dets.sort { $0.score > $1.score }
var keep: [Det] = []
for d in dets {
    if keep.count >= 300 { break }
    if !keep.contains(where: { $0.cls == d.cls && iou($0.rect, d.rect) > iouT }) { keep.append(d) }
}

print(String(format: "[result] %@  dets=%d  infer=%.1fms",
             (srcPath as NSString).lastPathComponent, keep.count, infMs))
for d in keep.prefix(50) {
    print(String(format: "  %@  %.2f  [%.0f %.0f %.0f %.0f]",
                 names[d.cls], d.score, d.rect.minX, d.rect.minY, d.rect.maxX, d.rect.maxY))
}

// ---------- draw boxes + save ----------
// vivid, distinct palette (cycles for >10 classes)
let palette: [CGColor] = [
    (0.98, 0.26, 0.30), (0.20, 0.71, 0.98), (0.16, 0.85, 0.52), (0.99, 0.79, 0.12),
    (0.72, 0.40, 0.98), (0.99, 0.55, 0.18), (0.10, 0.83, 0.80), (0.98, 0.36, 0.66),
    (0.55, 0.82, 0.28), (0.40, 0.52, 0.98),
].map { CGColor(red: CGFloat($0.0), green: CGFloat($0.1), blue: CGFloat($0.2), alpha: 1) }
func labelTextColor(on bg: CGColor) -> CGColor {   // black on light fills, white on dark — readable on any hue
    let c = bg.components ?? [0, 0, 0]
    let lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return lum > 0.62 ? CGColor(gray: 0.05, alpha: 1) : CGColor(gray: 1, alpha: 1)
}
func drawAndSave(_ image: CGImage, _ dets: [Det], _ path: String) {
    let w = image.width, h = image.height
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    // No flip (upright save, standard CGImage<->pixels). The context is bottom-up while detections are
    // TOP-DOWN px, so convert each box's y: ctxY = h - topDownY. Text draws upright in a bottom-up ctx.
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    ctx.setLineJoin(.round)
    let lw = max(CGFloat(2), CGFloat(w) / 640)
    let fontSize = max(CGFloat(12), CGFloat(w) / 95)
    let font = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, fontSize, nil)
    for d in dets {
        let color = palette[d.cls % palette.count]
        let box = CGRect(x: d.rect.minX, y: CGFloat(h) - d.rect.maxY, width: d.rect.width, height: d.rect.height)
        // rounded box: subtle inner tint + crisp stroke
        let radius = min(min(box.width, box.height) * 0.18, lw * 4)
        let boxPath = CGPath(roundedRect: box, cornerWidth: radius, cornerHeight: radius, transform: nil)
        ctx.addPath(boxPath); ctx.setFillColor(color.copy(alpha: 0.12) ?? color); ctx.fillPath()
        ctx.addPath(boxPath); ctx.setStrokeColor(color); ctx.setLineWidth(lw); ctx.strokePath()
        // rounded label pill "name 0.83" flush on the box top-left, contrast-aware text
        let label = "\(names[d.cls])  \(String(format: "%.2f", d.score))"
        let attr = NSAttributedString(string: label, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): labelTextColor(on: color)])
        let line = CTLineCreateWithAttributedString(attr)
        let tw = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        let padX = fontSize * 0.5, chipH = fontSize + 6, chipW = tw + padX * 2
        var chipY = box.maxY - lw / 2                                 // flush on the box top edge
        if chipY + chipH > CGFloat(h) { chipY = box.maxY - chipH }    // near image top -> put inside
        let chip = CGRect(x: box.minX - lw / 2, y: chipY, width: chipW, height: chipH)
        let chipPath = CGPath(roundedRect: chip, cornerWidth: chipH * 0.28, cornerHeight: chipH * 0.28, transform: nil)
        ctx.addPath(chipPath); ctx.setFillColor(color); ctx.fillPath()
        ctx.textPosition = CGPoint(x: chip.minX + padX, y: chipY + (chipH - fontSize) / 2 + fontSize * 0.2)
        CTLineDraw(line, ctx)
    }
    guard let outImg = ctx.makeImage() else { return }
    let type: CFString = path.hasSuffix(".png") ? UTType.png.identifier as CFString
                                                : UTType.jpeg.identifier as CFString
    if let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, type, 1, nil) {
        CGImageDestinationAddImage(dest, outImg, nil)
        CGImageDestinationFinalize(dest)
        print("[saved] \(path)")
    }
}
drawAndSave(cg, keep, outPath)
