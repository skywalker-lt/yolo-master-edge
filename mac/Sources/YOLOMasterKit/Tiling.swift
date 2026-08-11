// Tiled inference for YOLOMasterKit — a faithful port of upstream YOLO-Master's Sparse SAHI
// Mode (ultralytics/engine/predictor.py:542-699, `_run_sparse_sahi_single`) plus a dense
// ("traditional tiling") variant that runs every grid tile unconditionally.
//
// Shape of the port (deviations from upstream are deliberate and documented):
//   * Upstream NMSes each tile at the caller's conf/iou, then merges with one more NMS.
//     Here NO per-tile NMS runs: the merged pool is PRE-NMS candidates (conf floor 0.05) from
//     the global pass + every executed tile, offset into full-image coordinates. The single
//     user-conf/iou NMS at render time subsumes per-tile NMS, keeps the cached pool independent
//     of the UI sliders (the app's cache-and-retune pattern), and hands CW-NMS the pre-NMS pool
//     its weighting wants.
//   * Tile detections are clipped to real crop content (via forwardPadded's origW/H = crop
//     dims); upstream lets boxes live on the gray padding.
//   * maskCoeffs are stripped from every detection in a tiled pool — tile coefficients are
//     meaningless against a full-image proto tensor, so segmentation masks are structurally
//     disabled in tiled modes (boxes only).
//   * A maxTiles safety cap bounds the tiles RUN per image (row-major truncation, surfaced in
//     `TiledOutput.capped`) so a gigapixel input degrades gracefully instead of hanging.
import Foundation
import CoreGraphics

/// Tiled-inference mode. `.dense` = global pass + every grid tile ("traditional tiling").
/// `.sparse` = Sparse SAHI: global pass + only the tiles where the global pass found evidence
/// (objectness mask > threshold), with an all-or-nothing dense fallback when nothing is found.
public enum TilingMode: String, CaseIterable, Sendable {
    case off, dense, sparse
    public var label: String {
        switch self {
        case .off: return "Off"
        case .dense: return "Dense"
        case .sparse: return "Sparse SAHI"
        }
    }
}

/// Tiling parameters. The UI exposes only `mode`; everything else stays at the upstream
/// defaults (default.yaml:75-79) and is exposed here for the CLI/tests.
public struct TilingConfig: Sendable {
    public var mode: TilingMode = .off
    public var overlapRatio: CGFloat = 0.2        // overlap = Int(imgsz * ratio); upstream 0.2
    public var objectnessThreshold: Float = 0.15  // strict > gate on the painted mask
    public var fallback: Bool = true              // sparse only: zero active tiles -> run ALL
    public var maxTiles: Int = 256                // safety cap on tiles RUN per image
    public init(mode: TilingMode = .off) { self.mode = mode }
}

/// Aggregate tile statistics over one or many images (a folder run), for the stats card.
public struct TileStats: Sendable {
    public var tilesRun = 0
    public var tilesTotal = 0
    public var fallbacks = 0   // images where the sparse all-or-nothing fallback fired
    public var capped = 0      // images where the maxTiles cap truncated the run
    public init() {}
    public mutating func add(_ o: TiledOutput) {
        tilesRun += o.tilesRun; tilesTotal += o.tilesTotal
        fallbacks += o.usedFallback ? 1 : 0; capped += o.capped ? 1 : 0
    }
}

/// Result of a tiled inference: the merged candidate pool plus run statistics for the UI.
public struct TiledOutput: Sendable {
    public let candidates: [Detection]  // global + tile pool, coeffs stripped, sorted score desc
    public let tilesTotal: Int          // grid tiles surviving the drop check
    public let tilesRun: Int
    public let usedFallback: Bool       // sparse: zero active tiles -> dense fallback fired
    public let capped: Bool             // maxTiles truncation hit
    public let inferMs: Double          // SUM of all model-only forwards (global + tiles)
}

extension Detector {
    /// Upstream's mask resolution divisor (predictor.py:556, `mask_scale = 8`). Fixed, not
    /// derived from slice size. Also drives the tile drop check.
    private static let maskScale = 8

    /// The upstream slice grid (predictor.py:573-590), exactly: stride = sliceSize - overlap
    /// from 0 while < image dim; right/bottom tiles CLIPPED (not shifted back); a tile is
    /// dropped when integer division by the mask scale collapses it in either axis
    /// (`y/8 == yMax/8` — so a 7-px sliver at offset 1 survives, one at offset 0 doesn't).
    public static func tileGrid(width: Int, height: Int, sliceSize: Int, overlap: Int)
        -> [(x: Int, y: Int, w: Int, h: Int)] {
        let step = sliceSize - overlap
        guard step > 0, width > 0, height > 0 else { return [] }
        var tiles: [(x: Int, y: Int, w: Int, h: Int)] = []
        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let yMax = min(y + sliceSize, height), xMax = min(x + sliceSize, width)
                let mY1 = y / maskScale, mY2 = yMax / maskScale
                let mX1 = x / maskScale, mX2 = xMax / maskScale
                guard mY2 > mY1 && mX2 > mX1 else { continue }
                tiles.append((x: x, y: y, w: xMax - x, h: yMax - y))
            }
        }
        return tiles
    }

    /// Global forward + (dense: all tiles | sparse: mask-selected tiles) -> merged pre-NMS pool.
    /// See the file header for the exact semantics. `config.mode` must not be `.off` — callers
    /// branch to the ordinary single-pass path before getting here.
    public func tiledCandidates(_ image: CGImage, config: TilingConfig,
                                confFloor: Float = 0.05) throws -> TiledOutput {
        precondition(config.mode != .off, "tiledCandidates called with mode .off")
        let w = image.width, h = image.height

        // ---- global pass (honors the current preprocess mode, like any single-pass run) ----
        var totalMs = 0.0
        let globalCands: [Detection]
        do {
            let raw = try forward(image)          // RawOutput released at scope exit (memory)
            totalMs += raw.inferMs
            globalCands = candidates(raw, confFloor: confFloor)
        }

        let overlap = Int(CGFloat(imgsz) * config.overlapRatio)
        let grid = Detector.tileGrid(width: w, height: h, sliceSize: imgsz, overlap: overlap)

        // ---- tile selection ----
        var selected: [(x: Int, y: Int, w: Int, h: Int)]
        var usedFallback = false
        switch config.mode {
        case .dense, .off:
            selected = grid
        case .sparse:
            // 1/8-scale objectness mask painted from the global pass (predictor.py:555-568).
            // Painted from floor-NMS'd candidates at a FIXED IoU (0.5) so tile selection is
            // independent of the UI conf/iou sliders; the 0.15 objectness gate does the work.
            let mH = h / Detector.maskScale + 1, mW = w / Detector.maskScale + 1
            var mask = [Float](repeating: 0, count: mH * mW)
            for d in Detector.nms(globalCands, conf: confFloor, iou: 0.5) {
                // Upstream: (xyxy / 8).astype(int) — truncation — then clamp to the mask dims.
                let x1 = max(0, Int(d.rect.minX) / Detector.maskScale)
                let y1 = max(0, Int(d.rect.minY) / Detector.maskScale)
                let x2 = min(mW, Int(d.rect.maxX) / Detector.maskScale)
                let y2 = min(mH, Int(d.rect.maxY) / Detector.maskScale)
                guard x2 > x1, y2 > y1 else { continue }   // sub-8px box paints nothing (upstream)
                for yy in y1..<y2 {
                    for xx in x1..<x2 {
                        mask[yy * mW + xx] = max(mask[yy * mW + xx], d.score)
                    }
                }
            }
            func tileMax(_ t: (x: Int, y: Int, w: Int, h: Int)) -> Float {
                let mY1 = t.y / Detector.maskScale, mY2 = (t.y + t.h) / Detector.maskScale
                let mX1 = t.x / Detector.maskScale, mX2 = (t.x + t.w) / Detector.maskScale
                var v: Float = 0
                for yy in mY1..<min(mY2, mH) {
                    for xx in mX1..<min(mX2, mW) { v = max(v, mask[yy * mW + xx]) }
                }
                return v
            }
            selected = grid.filter { tileMax($0) > config.objectnessThreshold }   // strict >
            // Upstream's all-or-nothing fallback (predictor.py:632-654): ONLY when zero tiles
            // are active, run the whole grid. If any tile is active, inactive ones are skipped.
            if selected.isEmpty && config.fallback && !grid.isEmpty {
                selected = grid
                usedFallback = true
            }
        }

        let capped = selected.count > config.maxTiles
        if capped { selected = Array(selected.prefix(config.maxTiles)) }

        // ---- run tiles sequentially, offset into full-image coords, strip mask coeffs ----
        var pool: [Detection] = globalCands.map {
            Detection(cls: $0.cls, score: $0.score, rect: $0.rect)   // coeffs stripped
        }
        for t in selected {
            guard let crop = image.cropping(to: CGRect(x: t.x, y: t.y, width: t.w, height: t.h))
            else { continue }
            let raw = try forwardPadded(crop)
            totalMs += raw.inferMs
            for d in candidates(raw, confFloor: confFloor) {
                pool.append(Detection(cls: d.cls, score: d.score,
                                      rect: d.rect.offsetBy(dx: CGFloat(t.x), dy: CGFloat(t.y))))
            }
        }
        pool.sort { $0.score > $1.score }   // nms() requires score-desc input; concat isn't sorted

        return TiledOutput(candidates: pool, tilesTotal: grid.count, tilesRun: selected.count,
                           usedFallback: usedFallback, capped: capped, inferMs: totalMs)
    }

    /// Convenience for the CLI: tiled forward + the selected NMS in one call.
    public func detectTiled(_ image: CGImage, conf: Float, iou iouT: CGFloat,
                            tiling: TilingConfig, nmsMode: NMSMode = .standard,
                            sigma: Float = 0.1) throws -> (Result, TiledOutput) {
        let out = try tiledCandidates(image, config: tiling)
        let dets = Detector.nms(out.candidates, conf: conf, iou: iouT, mode: nmsMode, sigma: sigma)
        return (Result(detections: dets, inferMs: out.inferMs), out)
    }
}
