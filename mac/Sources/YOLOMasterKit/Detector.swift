// YOLOMasterKit — shared Core ML inference backend for YOLO-Master detectors.
//
// Extracted VERBATIM from the CLI runner (Sources/YOLOMasterCoreML/main.swift) so the
// command-line tool and the SwiftUI app run the exact same letterbox → Core ML →
// decode([1,4+nc,anchors]) → per-class NMS path. One backend, two frontends.
import Foundation
import CoreML
import CoreGraphics

/// A single detection in ORIGINAL-image pixel coordinates (top-left origin).
public struct Detection: Sendable {
    public let cls: Int
    public let score: Float
    public let rect: CGRect
    public init(cls: Int, score: Float, rect: CGRect) {
        self.cls = cls; self.score = score; self.rect = rect
    }
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
    public init(_ s: String) {
        switch s.lowercased() {
        case "all": self = .all
        case "cpu", "cpuonly": self = .cpu
        default: self = .cpuAndGPU
        }
    }
}

public enum DetectorError: Error { case inputBuildFailed, badOutput }

/// Loads a `.mlpackage`/`.mlmodelc` YOLO-Master detector and runs inference.
/// `imgsz`, class count and names are read from the model (works for any exported
/// ultralytics detect model), so preprocessing always matches the checkpoint.
public final class Detector {
    public let imgsz: Int
    public let nc: Int
    public let classNames: [String]
    public let computeMode: ComputeMode

    private let model: MLModel
    private let inputName: String
    private let outputName: String

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

        self.inputName = inName
        self.outputName = outName
        self.nc = ncResolved
        self.classNames = metaNames.count == ncResolved ? metaNames : (0..<ncResolved).map { "class\($0)" }
        self.imgsz = szResolved
    }

    /// Human-readable one-line model summary (parity with the CLI `[model]` banner).
    public var summary: String {
        "input=\(inputName) [\(imgsz)x\(imgsz)] output=\(outputName) classes=\(nc) compute=\(computeMode.rawValue)"
    }

    // ---------- preprocess ----------
    private struct LB { let px: [UInt8]; let scale: CGFloat; let padX: CGFloat; let padY: CGFloat }

    private func letterbox(_ image: CGImage) -> LB {
        let size = imgsz
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
        return LB(px: px, scale: scale, padX: padX, padY: padY)
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

    // ---------- decode + NMS ----------
    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let i = a.intersection(b); if i.isNull { return 0 }
        let ia = i.width * i.height
        return ia / (a.width * a.height + b.width * b.height - ia + 1e-6)
    }

    private func decodeAndNMS(_ y: MLMultiArray, _ origW: Int, _ origH: Int,
                              _ scale: CGFloat, _ padX: CGFloat, _ padY: CGFloat,
                              conf: Float, iouT: CGFloat) -> [Detection] {
        let na = y.shape[2].intValue
        let s1 = y.strides[1].intValue, s2 = y.strides[2].intValue
        var dets: [Detection] = []
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
                        dets.append(Detection(cls: c, score: s, rect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)))
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
        var keep: [Detection] = []
        for d in dets {
            if keep.count >= 300 { break }
            if !keep.contains(where: { $0.cls == d.cls && iou($0.rect, d.rect) > iouT }) { keep.append(d) }
        }
        return keep
    }

    // ---------- public inference ----------
    public struct Result: Sendable { public let detections: [Detection]; public let inferMs: Double }

    /// Full pipeline: letterbox → Core ML → decode → NMS. `inferMs` is model-only latency.
    public func detect(_ image: CGImage, conf: Float = 0.25, iou iouT: CGFloat = 0.5) throws -> Result {
        let lb = letterbox(image)
        guard let input = fillInput(lb.px) else { throw DetectorError.inputBuildFailed }
        let t0 = Date()
        let out = try model.prediction(from: input)
        let infMs = Date().timeIntervalSince(t0) * 1000
        guard let y = out.featureValue(for: outputName)?.multiArrayValue, y.shape.count == 3 else {
            throw DetectorError.badOutput
        }
        let dets = decodeAndNMS(y, image.width, image.height, lb.scale, lb.padX, lb.padY, conf: conf, iouT: iouT)
        return Result(detections: dets, inferMs: infMs)
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
