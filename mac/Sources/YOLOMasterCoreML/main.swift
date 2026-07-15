// yolomaster-coreml — command-line Core ML runner. Thin CLI over YOLOMasterKit
// (shared inference backend + folder/video pipelines).
//
// --source: image | folder/ | video.(mp4|mov|m4v) — mode auto-detected. Modes:
//   image   -> annotated image (--out out.jpg)
//   folder  -> annotated folder (--out preds/) + batch timing
//   video   -> annotated video  (--out out.mp4), size/fps preserved
//   --benchmark -> model-only latency (percentiles + img/s)
// Compute defaults cpuAndGPU (the ANE can crash on this fragmented MoE graph); --compute all|cpu.
//
// Build:  swift build -c release --package-path mac
import Foundation
import CoreGraphics
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
func fps(_ ms: Double) -> String { f1(ms > 0 ? 1000 / ms : 0) }

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
let resize = Int(argValue("--resize", "0")!) ?? 0
let boxStyle = BoxStyle(rawValue: (argValue("--style", "hud")!).lowercased()) ?? .hud
let labelMode = LabelMode(rawValue: (argValue("--label", "full")!).lowercased()) ?? .full

// ---------- backend (shared) ----------
let detector: Detector
do { detector = try Detector(modelURL: URL(fileURLWithPath: modelPath), compute: compute) }
catch { die("model load failed: \(error)", 3) }
print("[model] \(detector.summary)")

// ---------- single image ----------
func processImage(_ path: String, _ outPath: String) {
    guard var cg = loadCGImage(URL(fileURLWithPath: path)) else { logErr("skip (unreadable): \(path)"); return }
    if resize > 0 { cg = resizeLong(cg, resize) }
    guard let res = try? detector.detect(cg, conf: conf, iou: iouT) else { logErr("predict failed: \(path)"); return }
    print("[det] \((path as NSString).lastPathComponent)  dets=\(res.detections.count)  infer=\(f1(res.inferMs))ms")
    if !noSave, let a = annotate(cg, res.detections, names: detector.classNames, style: boxStyle, label: labelMode) {
        saveCGImage(a, to: URL(fileURLWithPath: outPath))
        print("[saved] \(outPath)")
    }
}

// ---------- benchmark: model-only latency ----------
func runBenchmark(_ paths: [URL]) {
    guard let cg0 = loadCGImage(paths[0]) else { die("bench: cannot read \(paths[0].path)", 4) }
    for _ in 0..<10 { _ = try? detector.inferOnly(cg0) }                 // warmup
    var times: [Double] = []
    if paths.count == 1 {
        for _ in 0..<iters { if let t = try? detector.inferOnly(cg0) { times.append(t) } }
    } else {
        for p in paths { if let cg = loadCGImage(p), let t = try? detector.inferOnly(cg) { times.append(t) } }
    }
    times.sort()
    func pct(_ p: Double) -> Double { times.isEmpty ? 0 : times[min(times.count - 1, Int(p * Double(times.count)))] }
    let mean = times.reduce(0, +) / Double(max(times.count, 1))
    print("[bench] n=\(times.count) compute=\(compute.rawValue) imgsz=\(detector.imgsz)")
    print("[bench] latency ms:  mean \(String(format: "%.2f", mean))  min \(String(format: "%.2f", times.first ?? 0))" +
          "  p50 \(String(format: "%.2f", pct(0.5)))  p90 \(String(format: "%.2f", pct(0.9)))  p99 \(String(format: "%.2f", pct(0.99)))")
    print("[bench] throughput:  \(fps(mean)) img/s (model-only)")
}

// ---------- dispatch (source auto-detected) ----------
let src = URL(fileURLWithPath: srcPath)
guard classifySource(src) != .unknown else { die("source not found / unsupported: \(srcPath)", 4) }

if benchmark {
    let paths = classifySource(src) == .folder ? listImages(src) : [src]
    if paths.isEmpty { die("benchmark: no images in \(srcPath)", 4) }
    runBenchmark(paths)
} else {
    switch classifySource(src) {
    case .video:
        let out = URL(fileURLWithPath: outArg ?? "out.mp4")
        do {
            let s = try await runVideo(detector, input: src, output: out, conf: conf, iou: iouT,
                                       style: boxStyle, label: labelMode, resize: resize) { n, _ in
                if n % 60 == 0 { logErr("  \(n) frames…") }
            }
            print("[video] \(s.frames) frames -> \(out.path)  (\(s.outW)x\(s.outH) @\(s.fps)fps)")
            print("[video] model-infer mean \(f1(s.meanMs))ms -> \(fps(s.meanMs)) fps (model-only)")
        } catch { die("video failed: \(error)", 5) }
    case .folder:
        let out = noSave ? nil : URL(fileURLWithPath: outArg ?? "preds")
        print("[batch] \(src.lastPathComponent) -> \(out?.path ?? "(--no-save)")")
        let s = runFolder(detector, input: src, output: out, conf: conf, iou: iouT,
                          style: boxStyle, label: labelMode, resize: resize)
        print("[batch] \(s.processed)/\(s.total) ok  |  model-infer mean \(f1(s.meanMs))ms -> \(fps(s.meanMs)) img/s steady")
    case .image:
        processImage(srcPath, outArg ?? "out.jpg")
    case .unknown:
        die("unsupported source: \(srcPath)", 4)
    }
}
