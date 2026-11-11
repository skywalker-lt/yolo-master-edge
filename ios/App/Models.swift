// Model discovery + compute-unit selection for the iOS app.
// .mlpackage bundles are compiled by Xcode into .mlmodelc resources; drop the
// p03 packages into ios/Models/ (see ios/README.md) and they appear here.
import CoreML
import Foundation
import YOLOMasterKit

struct BundledModel: Identifiable, Hashable {
    let id: String       // bundle stem, e.g. "p03_v01n_coco_fp16"
    let url: URL         // load source: a ready .mlmodelc, or a .mlpackage to compile on first use
    /// Distinctive part of the name (the canonical YOLO-Master prefix stripped).
    private var stem: String {
        for p in ["yolo-master-", "yolo-master_", "yolo_master_", "yolomaster-"]
        where id.lowercased().hasPrefix(p) {
            return String(id.dropFirst(p.count))
        }
        return id
    }
    /// A stock model from another family: its file name already says what it is (yolo11n,
    /// yolo12n, yolo26n-pose, yolov8n ...) and had no YOLO-Master prefix to strip. Showing it
    /// as "YOLO-Master-yolo11n" would put this project's name on someone else's model.
    private var isStock: Bool { id.lowercased().hasPrefix("yolo") && stem == id }
    /// Display name: YOLO-Master models are presented as YOLO-Master-<stem> (existing
    /// YOLO-Master prefixes are normalized, never doubled); stock models keep their own name.
    var fullName: String { isStock ? id : "YOLO-Master-" + stem }
    /// Collapsed menu-bar label: the first 6 characters of the model's own name
    /// (the stem - the shared YOLO-Master prefix carries no information there).
    var shortID: String { stem.count > 6 ? String(stem.prefix(6)) + "..." : stem }

    private static var cacheDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    }

    /// User-imported Core ML models (Settings > Custom models). Kept out of
    /// the bundle so they survive app updates and can be deleted individually.
    static var customModelsDir: URL {
        cacheDir.appendingPathComponent("CustomModels", isDirectory: true)
    }

    /// The models the user has imported (.mlmodelc / .mlpackage / .mlmodel).
    static func customModels() -> [BundledModel] {
        guard let items = try? FileManager.default.contentsOfDirectory(
            at: customModelsDir, includingPropertiesForKeys: nil) else { return [] }
        return items
            .filter { ["mlmodelc", "mlpackage", "mlmodel"].contains($0.pathExtension.lowercased()) }
            .map { BundledModel(id: $0.deletingPathExtension().lastPathComponent, url: $0) }
            .sorted { $0.id < $1.id }
    }

    /// Remove an imported model and any compiled cache it produced.
    static func deleteCustom(_ m: BundledModel) {
        try? FileManager.default.removeItem(at: m.url)
        try? FileManager.default.removeItem(at: cacheDir.appendingPathComponent(m.id + ".mlmodelc"))
    }

    /// A ready-to-load .mlmodelc. If this model is a .mlpackage, it is compiled
    /// and cached in Application Support on FIRST use (kept out of `discover` so
    /// launch stays instant); later launches reuse the cache. Compile is the
    /// slow, one-time CoreML source pass - do it lazily, only for the model the
    /// user actually loads, never for the whole bundle at startup.
    func compiledURL() throws -> URL {
        if url.pathExtension.lowercased() == "mlmodelc" { return url }
        let cached = BundledModel.cacheDir.appendingPathComponent(id + ".mlmodelc")
        if FileManager.default.fileExists(atPath: cached.path) { return cached }
        let tmp = try MLModel.compileModel(at: url)
        try? FileManager.default.createDirectory(at: BundledModel.cacheDir,
                                                 withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: cached)
        try FileManager.default.moveItem(at: tmp, to: cached)
        return cached
    }

    /// Enumeration only - NO compilation (that would block launch on every
    /// bundled model). Ready .mlmodelc are returned directly; .mlpackage are
    /// returned pointing at their cached compile if present, else at the package
    /// (compiled lazily by `compiledURL()` on first load).
    static func discover() -> [BundledModel] {
        var found: [BundledModel] = []
        var seen = Set<String>()
        for sub in [nil, "Models"] as [String?] {
            for url in Bundle.main.urls(forResourcesWithExtension: "mlmodelc",
                                        subdirectory: sub) ?? [] {
                let name = url.deletingPathExtension().lastPathComponent
                if seen.insert(name).inserted { found.append(BundledModel(id: name, url: url)) }
            }
        }
        for sub in [nil, "Models"] as [String?] {
            for pkg in Bundle.main.urls(forResourcesWithExtension: "mlpackage",
                                        subdirectory: sub) ?? [] {
                let name = pkg.deletingPathExtension().lastPathComponent
                if !seen.insert(name).inserted { continue }
                let cached = cacheDir.appendingPathComponent(name + ".mlmodelc")
                let ready = FileManager.default.fileExists(atPath: cached.path)
                found.append(BundledModel(id: name, url: ready ? cached : pkg))
            }
        }
        for m in customModels() where seen.insert(m.id).inserted { found.append(m) }
        return found.sorted { $0.id < $1.id }
    }

    /// Launch default: the bundled segmentation model when present (v0.1-seg-N,
    /// the richest demo), else the first model alphabetically.
    static func preferred(in models: [BundledModel]) -> BundledModel? {
        models.first { $0.id.localizedCaseInsensitiveContains("seg") } ?? models.first
    }
}

extension Detector {
    /// The IoU threshold to decode with, given the IoU lever's value. An NMS-free (end2end)
    /// model already emits one row per object, so its lever is hidden (`TuningPanel`) - and a
    /// hidden lever must not act either: the value left over from the previous model would
    /// silently suppress boxes. Infinity lets every row through `Detector.nms` (no overlap
    /// exceeds it), leaving only the confidence cut.
    func nmsThreshold(lever: CGFloat) -> CGFloat { end2end ? .infinity : lever }
}

/// UI wrapper over the Kit's ComputeMode.
///
/// `.auto` is Core ML's own placement (`MLComputeUnits.all`): Core ML decides
/// which parts of the model run on the Neural Engine, the GPU and the CPU. It
/// is NOT "the Neural Engine" - segments the Neural Engine refuses land on the
/// GPU or CPU - so the unit is labelled Auto (it was shown as "ANE" up to
/// 1.1.0 build 1). On A-series silicon Auto is the expected winner; the
/// per-unit bench exists to verify exactly that on-device.
enum ComputeChoice: String, CaseIterable, Identifiable, Codable {
    case auto = "Auto"      // Core ML decides: Neural Engine + GPU + CPU (.all)
    case gpu = "GPU"        // CPU + GPU
    case cpu = "CPU"        // CPU only
    var id: String { rawValue }

    /// Benchmark history written before the rename stored this unit as "ANE".
    /// Read it as `.auto` (those runs WERE Core ML's automatic split) instead
    /// of failing: BenchHistory treats an undecodable file as empty, so one
    /// unknown label would drop the user's whole saved history.
    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        if raw == "ANE" { self = .auto; return }
        guard let unit = ComputeChoice(rawValue: raw) else {
            throw DecodingError.dataCorrupted(DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "unknown compute unit \(raw)"))
        }
        self = unit
    }

    /// Units to expose in a compute picker. CPU-only inference can crash Live/Photo
    /// on some MoE models, so it is hidden unless the user opts in (Settings).
    static func available(allowCPU: Bool) -> [ComputeChoice] {
        allowCPU ? allCases : allCases.filter { $0 != .cpu }
    }

    var mode: ComputeMode {
        switch self {
        case .auto: return ComputeMode("all")
        case .gpu: return ComputeMode("cpuandgpu")
        case .cpu: return ComputeMode("cpu")
        }
    }

    /// Per-device default. The first-time Neural Engine compile that Auto
    /// triggers for these fragmented MoE+attention models is painfully slow on
    /// A17-and-earlier silicon (<= iPhone 15 Pro, hardware id "iPhoneN,M" with
    /// N <= 16), so those devices default to GPU (fast Metal compile); iPhone 16
    /// and newer default to Auto. Everything non-iPhone (iPad, simulator) also
    /// defaults to Auto.
    static var deviceDefault: ComputeChoice {
        var sys = utsname(); uname(&sys)
        let id = Mirror(reflecting: sys.machine).children.compactMap { c -> String? in
            guard let v = c.value as? Int8, v != 0 else { return nil }
            return String(UnicodeScalar(UInt8(v)))
        }.joined()                                   // e.g. "iPhone16,1"
        guard id.hasPrefix("iPhone"),
              let major = Int(id.dropFirst(6).prefix { $0.isNumber })
        else { return .auto }
        return major <= 16 ? .gpu : .auto
    }
}
