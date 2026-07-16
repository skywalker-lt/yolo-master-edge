// YOLOMasterKit — shared Core ML inference backend for YOLO-Master detectors.
//
// Extracted VERBATIM from the CLI runner (Sources/YOLOMasterCoreML/main.swift) so the
// command-line tool and the SwiftUI app run the exact same letterbox → Core ML →
// decode([1,4+nc,anchors]) → per-class NMS path. One backend, two frontends.
import Foundation
import CoreML
import CoreGraphics
import CoreVideo

/// IEEE half bits (UInt16) -> Float32, done by hand. On x86_64 the `Float16` *instance* API
/// (conversions, `.bitPattern`) is unavailable with the current SDK (arm64 has native Float16); the
/// MLMultiArray buffer must still be typed as Float16 (only MLShapedArrayScalar types are allowed),
/// so each element is bit-reinterpreted to UInt16 via unsafeBitCast and converted here. Keeps the
/// universal (arm64 + x86_64) build compiling with identical numerics on both slices.
@inline(__always) func halfToFloat(_ h: UInt16) -> Float32 {
    let sign = UInt32(h & 0x8000) << 16
    let exp  = UInt32(h & 0x7C00) >> 10
    let mant = UInt32(h & 0x03FF)
    let bits: UInt32
    if exp == 0 {
        if mant == 0 { bits = sign }                              // +/-0
        else {                                                    // subnormal -> normalized
            var e: UInt32 = 127 - 15 + 1, m = mant
            while (m & 0x0400) == 0 { m <<= 1; e -= 1 }
            bits = sign | (e << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 0x1F {
        bits = sign | 0x7F80_0000 | (mant << 13)                  // +/-inf / NaN
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13)   // normal
    }
    return Float32(bitPattern: bits)
}

/// A single detection in ORIGINAL-image pixel coordinates (top-left origin).
/// `maskCoeffs` is non-empty only for segmentation models (nm mask-prototype coefficients).
public struct Detection: Sendable {
    public let cls: Int
    public let score: Float
    public let rect: CGRect
    public let maskCoeffs: [Float]
    public init(cls: Int, score: Float, rect: CGRect, maskCoeffs: [Float] = []) {
        self.cls = cls; self.score = score; self.rect = rect; self.maskCoeffs = maskCoeffs
    }
}

/// A rendered instance mask (segmentation): a proto-resolution tinted RGBA image, the unit
/// sub-rect of the proto grid mapping to the full original image, and the box to clip it to.
public struct MaskBitmap: @unchecked Sendable {
    public let image: CGImage
    public let protoCrop: CGRect   // unit coords of the proto grid (top-left origin) = full original image
    public let clip: CGRect        // detection box, original-image pixels (top-left origin)
}

/// Core ML compute unit selection. Default cpuAndGPU: the ANE can crash on this
/// fragmented MoE+attention graph.
public enum ComputeMode: String, CaseIterable, Sendable {
    case cpuAndGPU, all, cpu
    public var mlUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpu: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        }
    }
    /// Human-readable label for the UI.
    public var label: String {
        switch self {
        case .cpuAndGPU: return "CPU + GPU"
        case .all: return "CPU + GPU + Neural Engine"
        case .cpu: return "CPU only"
        }
    }
    public init(_ s: String) {
        switch s.lowercased() {
        case "all": self = .all
        case "cpu", "cpuonly": self = .cpu
        default: self = .cpuAndGPU
        }
    }
}

public enum DetectorError: Error { case inputBuildFailed, badOutput }

/// IoU of two rects (used by NMS).
func rectIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let i = a.intersection(b); if i.isNull { return 0 }
    let ia = i.width * i.height
    return ia / (a.width * a.height + b.width * b.height - ia + 1e-6)
}

/// Loads a `.mlpackage`/`.mlmodelc` YOLO-Master detector and runs inference.
/// `imgsz`, class count and names are read from the model (works for any exported
/// ultralytics detect model), so preprocessing always matches the checkpoint.
public final class Detector {
    public let imgsz: Int
    public let nc: Int
    public let classNames: [String]
    public let computeMode: ComputeMode

    public let isSegment: Bool
    public let nm: Int   // mask-coeff count (segmentation); 0 for detection

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let protoName: String

    public init(modelURL: URL, compute: ComputeMode = .cpuAndGPU) throws {
        self.computeMode = compute
        let cfg = MLModelConfiguration(); cfg.computeUnits = compute.mlUnits
        let loaded: MLModel
        if modelURL.pathExtension.lowercased() == "mlmodelc" {
            loaded = try MLModel(contentsOf: modelURL, configuration: cfg)
        } else {
            let compiled = try MLModel.compileModel(at: modelURL)
            loaded = try MLModel(contentsOf: compiled, configuration: cfg)
        }
        self.model = loaded
        let md = loaded.modelDescription

        let inName = md.inputDescriptionsByName.keys.sorted().first ?? "images"
        let meta = md.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let metaNames = meta["names"]?.split(separator: ",").map(String.init)
            ?? ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
        let outName = meta["output"] ?? md.outputDescriptionsByName.keys.sorted().first ?? "output0"
        // class count from the output shape [1, 4+nc, anchors] (authoritative for ANY model);
        // fall back to the metadata names count.
        let ncFromShape: Int? = {
            if let sh = md.outputDescriptionsByName[outName]?.multiArrayConstraint?.shape,
               sh.count >= 2, sh[1].intValue > 4 { return sh[1].intValue - 4 }
            return nil
        }()
        let ncResolved = ncFromShape ?? metaNames.count
        // Input resolution is FIXED at export time — read it from the model ([1,3,H,W]).
        let szResolved: Int = {
            if let shape = md.inputDescriptionsByName[inName]?.multiArrayConstraint?.shape,
               shape.count >= 4, shape[2].intValue > 0 { return shape[2].intValue }
            if let s = meta["imgsz"], let v = Int(s), v > 0 { return v }
            return 640
        }()

        // segmentation: detection tensor is [1, 4+nc+nm, anchors] -> nc from names, plus a proto output
        let segTask = (meta["task"] ?? "detect") == "segment"
        let ncFinal = segTask ? metaNames.count : ncResolved

        self.inputName = inName
        self.outputName = outName
        self.protoName = meta["proto"] ?? ""
        self.isSegment = segTask
        self.nm = Int(meta["nm"] ?? "0") ?? 0
        self.nc = ncFinal
        self.classNames = metaNames.count == ncFinal ? metaNames : (0..<ncFinal).map { "class\($0)" }
        self.imgsz = szResolved
    }

    /// Human-readable one-line model summary (parity with the CLI `[model]` banner).
    public var summary: String {
        "input=\(inputName) [\(imgsz)x\(imgsz)] output=\(outputName) classes=\(nc) compute=\(computeMode.rawValue)"
    }

    // ---------- preprocess ----------
    /// How the source image is fit into the model's fixed imgsz×imgsz input. This is a PREPROCESSING
    /// choice (changes the forward-pass input), not a tuning param — switching it requires re-inference.
    /// `.letterbox`: aspect-preserving fit + gray padding (YOLO default). `.stretch`: force-resize the
    /// whole image to imgsz×imgsz (no padding; the size the model was trained on), distorting aspect.
    public enum PreprocessMode: String, CaseIterable, Sendable { case letterbox, stretch }
    public var preprocess: PreprocessMode = .letterbox

    private struct LB { let px: [UInt8]; let scaleX: CGFloat; let scaleY: CGFloat; let padX: CGFloat; let padY: CGFloat }

    private func letterbox(_ image: CGImage) -> LB {
        let size = imgsz
        let w = image.width, h = image.height
        let scaleX: CGFloat, scaleY: CGFloat, nw: Int, nh: Int, padX: CGFloat, padY: CGFloat
        switch preprocess {
        case .letterbox:                                    // aspect-preserving fit + centered padding
            let s = min(CGFloat(size) / CGFloat(w), CGFloat(size) / CGFloat(h))
            scaleX = s; scaleY = s
            nw = Int((CGFloat(w) * s).rounded()); nh = Int((CGFloat(h) * s).rounded())
            padX = CGFloat(size - nw) / 2; padY = CGFloat(size - nh) / 2
        case .stretch:                                      // force-resize to the full imgsz square
            scaleX = CGFloat(size) / CGFloat(w); scaleY = CGFloat(size) / CGFloat(h)
            nw = size; nh = size; padX = 0; padY = 0
        }
        var px = [UInt8](repeating: 114, count: size * size * 4)
        px.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: size, height: size, bitsPerComponent: 8,
                                      bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: padX, y: padY, width: CGFloat(nw), height: CGFloat(nh)))  // no flip -> top-down
        }
        return LB(px: px, scaleX: scaleX, scaleY: scaleY, padX: padX, padY: padY)
    }

    private func fillInput(_ raster: [UInt8]) -> MLDictionaryFeatureProvider? {
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

    // ---------- decode + NMS (split so forward is cached once, tuning stays cheap) ----------
    /// All boxes above `confFloor` (NO NMS), ORIGINAL-image pixels, sorted by score desc.
    /// Cache this once per image after `forward`; then re-run `nms(_:conf:iou:)` for cheap tuning.
    public func candidates(_ raw: RawOutput, confFloor: Float = 0.05) -> [Detection] {
        let y = raw.y
        let na = y.shape[2].intValue
        let s1 = y.strides[1].intValue, s2 = y.strides[2].intValue
        let scaleX = raw.scaleX, scaleY = raw.scaleY, padX = raw.padX, padY = raw.padY
        let origW = raw.origW, origH = raw.origH
        var dets: [Detection] = []
        func decodeAnchors(_ at: (Int, Int) -> Float32) {
            for a in 0..<na {
                let cx = CGFloat(at(0, a)), cy = CGFloat(at(1, a)), bw = CGFloat(at(2, a)), bh = CGFloat(at(3, a))
                for c in 0..<nc {
                    let s = at(4 + c, a)
                    if s <= confFloor { continue }
                    var x1 = (cx - bw / 2 - padX) / scaleX, y1 = (cy - bh / 2 - padY) / scaleY
                    var x2 = (cx + bw / 2 - padX) / scaleX, y2 = (cy + bh / 2 - padY) / scaleY
                    x1 = max(0, min(CGFloat(origW), x1)); x2 = max(0, min(CGFloat(origW), x2))
                    y1 = max(0, min(CGFloat(origH), y1)); y2 = max(0, min(CGFloat(origH), y2))
                    if x2 > x1 && y2 > y1 {
                        var coeffs: [Float] = []
                        if nm > 0 {
                            coeffs.reserveCapacity(nm)
                            for k in 0..<nm { coeffs.append(Float(at(4 + nc + k, a))) }
                        }
                        dets.append(Detection(cls: c, score: s,
                                              rect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
                                              maskCoeffs: coeffs))
                    }
                }
            }
        }
        if y.dataType == .float16 {
            y.withUnsafeBufferPointer(ofType: Float16.self) { buf in
                guard let yp = buf.baseAddress else { return }
                decodeAnchors { c, a in halfToFloat(unsafeBitCast(yp[c * s1 + a * s2], to: UInt16.self)) }
            }
        } else {
            y.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                guard let yp = buf.baseAddress else { return }
                decodeAnchors { c, a in yp[c * s1 + a * s2] }
            }
        }
        dets.sort { $0.score > $1.score }
        return dets
    }

    /// Filter cached `candidates` by `conf` + per-class greedy NMS (cap 300). Cheap — no model call.
    public static func nms(_ dets: [Detection], conf: Float, iou iouT: CGFloat, maxDet: Int = 300) -> [Detection] {
        var keep: [Detection] = []
        for d in dets where d.score > conf {
            if keep.count >= maxDet { break }
            if !keep.contains(where: { $0.cls == d.cls && rectIoU($0.rect, d.rect) > iouT }) { keep.append(d) }
        }
        return keep
    }

    // ---------- public inference ----------
    public struct Result: Sendable { public let detections: [Detection]; public let inferMs: Double }

    /// Cached forward-pass output + letterbox geometry. Hold onto this and re-decode with
    /// different conf/iou via `decode(_:conf:iou:)` — no second model call. Post-processing
    /// (conf/iou threshold, NMS) is a frontend concern, not an inference one.
    public final class RawOutput {
        fileprivate let y: MLMultiArray
        fileprivate let proto: MLMultiArray?   // segmentation prototypes [1, nm, mh, mw]
        fileprivate let scaleX, scaleY, padX, padY: CGFloat   // per-axis (scaleX==scaleY for letterbox)
        public let origW, origH: Int
        public let inferMs: Double
        fileprivate init(y: MLMultiArray, proto: MLMultiArray?, scaleX: CGFloat, scaleY: CGFloat, padX: CGFloat, padY: CGFloat,
                         origW: Int, origH: Int, inferMs: Double) {
            self.y = y; self.proto = proto; self.scaleX = scaleX; self.scaleY = scaleY; self.padX = padX; self.padY = padY
            self.origW = origW; self.origH = origH; self.inferMs = inferMs
        }
    }

    /// Core ML forward pass only (letterbox → predict). Cache the result and re-`decode`.
    public func forward(_ image: CGImage) throws -> RawOutput {
        let lb = letterbox(image)
        guard let input = fillInput(lb.px) else { throw DetectorError.inputBuildFailed }
        let t0 = Date()
        let out = try model.prediction(from: input)
        let infMs = Date().timeIntervalSince(t0) * 1000
        guard let y = out.featureValue(for: outputName)?.multiArrayValue, y.shape.count == 3 else {
            throw DetectorError.badOutput
        }
        let proto = isSegment ? out.featureValue(for: protoName)?.multiArrayValue : nil
        return RawOutput(y: y, proto: proto, scaleX: lb.scaleX, scaleY: lb.scaleY, padX: lb.padX, padY: lb.padY,
                         origW: image.width, origH: image.height, inferMs: infMs)
    }

    /// Low-latency forward from a camera `CVPixelBuffer` (BGRA). Wraps the buffer as a CGImage with a
    /// single copy (no CIContext) then runs the same letterbox → predict path. For real-time streaming.
    public func forward(_ pixelBuffer: CVPixelBuffer) throws -> RawOutput {
        guard let cg = Detector.cgImage(from: pixelBuffer) else { throw DetectorError.inputBuildFailed }
        return try forward(cg)
    }

    /// Cheap BGRA `CVPixelBuffer` → `CGImage` (one memcpy via a buffer-backed context; no Core Image).
    static func cgImage(from pb: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let bmp = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue  // 32BGRA
        guard let ctx = CGContext(data: base, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bmp) else { return nil }
        return ctx.makeImage()
    }

    /// Decode + per-class NMS from a cached forward pass. Cheap — no model call.
    public func decode(_ raw: RawOutput, conf: Float, iou iouT: CGFloat) -> [Detection] {
        Detector.nms(candidates(raw, confFloor: conf), conf: conf, iou: iouT)
    }

    /// Convenience: forward + decode in one call (used by the CLI). `inferMs` is model-only latency.
    public func detect(_ image: CGImage, conf: Float = 0.25, iou iouT: CGFloat = 0.5) throws -> Result {
        let raw = try forward(image)
        return Result(detections: decode(raw, conf: conf, iou: iouT), inferMs: raw.inferMs)
    }

    // ---------- segmentation masks ----------
    /// Instance mask for one detection: threshold(sigmoid(coeffs · protos)), tinted by class color,
    /// as a proto-resolution RGBA image plus the unit sub-rect of the proto grid that maps to the full
    /// original image (`protoCrop`) and the box to clip it to (`clip`, original-image pixels). nil if
    /// the model isn't segmentation or there's no proto tensor.
    public func maskImage(_ det: Detection, _ raw: RawOutput, threshold: Float = 0.5, alpha: UInt8 = 165) -> MaskBitmap? {
        guard let proto = raw.proto, proto.shape.count == 4, !det.maskCoeffs.isEmpty else { return nil }
        let cm = proto.shape[1].intValue, mh = proto.shape[2].intValue, mw = proto.shape[3].intValue
        let nmv = min(cm, det.maskCoeffs.count)
        let s1 = proto.strides[1].intValue, s2 = proto.strides[2].intValue, s3 = proto.strides[3].intValue
        let coeffs = det.maskCoeffs
        // tint from the class palette (context is premultipliedLast -> RGB carries color*alpha)
        let comps = classColor(det.cls).components ?? [1, 0.25, 0.25, 1]
        let cr = Float(comps[0]), cg = Float(comps[1]), cb = Float(comps[2]), aMax = Float(alpha)
        // Anti-aliased coverage instead of a hard threshold: smoothstep the sigmoid across a soft band
        // around `threshold`. Combined with bilinear upscaling this removes the classic serrated edge.
        let band: Float = 0.14, e0 = threshold - band, e1 = threshold + band, inv = 1 / (e1 - e0)
        var px = [UInt8](repeating: 0, count: mw * mh * 4)
        func fill(_ at: (Int, Int, Int) -> Float32) {
            for i in 0..<mh {
                for j in 0..<mw {
                    var acc: Float = 0
                    for k in 0..<nmv { acc += coeffs[k] * Float(at(k, i, j)) }
                    let v = 1 / (1 + expf(-acc))
                    var t = (v - e0) * inv
                    if t <= 0 { continue }
                    if t > 1 { t = 1 }
                    let a = t * t * (3 - 2 * t) * aMax          // smoothstep coverage * max alpha
                    let o = (i * mw + j) * 4
                    px[o] = UInt8(min(255, cr * a)); px[o + 1] = UInt8(min(255, cg * a))
                    px[o + 2] = UInt8(min(255, cb * a)); px[o + 3] = UInt8(min(255, a))
                }
            }
        }
        if proto.dataType == .float16 {
            proto.withUnsafeBufferPointer(ofType: Float16.self) { buf in
                guard let pp = buf.baseAddress else { return }
                fill { k, i, j in halfToFloat(unsafeBitCast(pp[k * s1 + i * s2 + j * s3], to: UInt16.self)) }
            }
        } else {
            proto.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                guard let pp = buf.baseAddress else { return }
                fill { k, i, j in pp[k * s1 + i * s2 + j * s3] }
            }
        }
        let bmp = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: &px, width: mw, height: mh, bitsPerComponent: 8, bytesPerRow: mw * 4,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bmp),
              let img = ctx.makeImage() else { return nil }
        // proto grid covers the letterboxed imgsz×imgsz input; the original image occupies
        // [pad, pad + orig*scale]. Express that as a unit crop of the proto grid (top-left origin).
        let sz = CGFloat(imgsz)
        let crop = CGRect(x: raw.padX / sz, y: raw.padY / sz,
                          width: CGFloat(raw.origW) * raw.scaleX / sz, height: CGFloat(raw.origH) * raw.scaleY / sz)
        return MaskBitmap(image: img, protoCrop: crop, clip: det.rect)
    }

    /// Transparent original-image-sized overlay compositing every detection's instance mask (box-clipped),
    /// for the Canvas-based stages (video / live camera) that draw over a raw frame rather than through
    /// `annotate`. Returns nil for non-seg models or when no mask survives. `dets` should be post-NMS.
    public func maskOverlay(_ dets: [Detection], _ raw: RawOutput) -> CGImage? {
        guard isSegment, raw.proto != nil, !dets.isEmpty else { return nil }
        let w = raw.origW, h = raw.origH
        guard w > 0, h > 0,
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high   // bilinear upscale proto->full res -> smooth mask edges
        var drew = false
        for d in dets {
            guard let m = maskImage(d, raw) else { continue }
            let pc = m.protoCrop, iw = CGFloat(m.image.width), ih = CGFloat(m.image.height)
            let sub = CGRect(x: pc.minX * iw, y: pc.minY * ih, width: pc.width * iw, height: pc.height * ih)
            guard sub.width > 0, sub.height > 0, let cropped = m.image.cropping(to: sub) else { continue }
            ctx.saveGState()
            ctx.clip(to: CGRect(x: m.clip.minX, y: CGFloat(h) - m.clip.maxY, width: m.clip.width, height: m.clip.height))
            ctx.draw(cropped, in: CGRect(x: 0, y: 0, width: w, height: h))  // upright, matches Canvas top-left mapping
            ctx.restoreGState()
            drew = true
        }
        return drew ? ctx.makeImage() : nil
    }

    /// Model-only forward (no decode/draw) — for latency benchmarking. Returns ms.
    @discardableResult
    public func inferOnly(_ image: CGImage) throws -> Double {
        let lb = letterbox(image)
        guard let input = fillInput(lb.px) else { throw DetectorError.inputBuildFailed }
        let t0 = Date()
        _ = try model.prediction(from: input)
        return Date().timeIntervalSince(t0) * 1000
    }
}
