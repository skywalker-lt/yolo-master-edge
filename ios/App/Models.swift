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
    /// yolo12n, yolo26n-pose, yolov8n, dwpose-t ...) and had no YOLO-Master prefix to strip.
    /// Showing it as "YOLO-Master-yolo11n" would put this project's name on someone else's model.
    private var isStock: Bool {
        let name = id.lowercased()
        return stem == id && ["yolo", "dwpose", "rtmpose", "rtmw"].contains { name.hasPrefix($0) }
    }
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
        ModelFacts.forget(m.id)
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
        let sorted = found.sorted { $0.id < $1.id }
        // type and attributes of each model (a small JSON read, cached): the callers run this
        // off the main thread, so the pickers and the Settings gallery find them ready
        for m in sorted { _ = ModelFacts.of(m) }
        return sorted
    }

    /// Launch default: the bundled segmentation model when present (v0.1-seg-N,
    /// the richest demo), else the first model alphabetically.
    static func preferred(in models: [BundledModel]) -> BundledModel? {
        models.first { $0.id.localizedCaseInsensitiveContains("seg") } ?? models.first
    }

    /// The selection to keep after the model list has been read again (every tab re-reads
    /// it each time it appears, so a model imported or deleted in Settings enters or leaves
    /// its menu): the same model while it still exists - matched by id, so a package that
    /// has been compiled since keeps its place - otherwise the launch default.
    static func reselect(_ current: BundledModel?, in models: [BundledModel]) -> BundledModel? {
        if let current, let same = models.first(where: { $0.id == current.id }) { return same }
        return preferred(in: models)
    }

    /// Why this model cannot be used, or nil when it can. Core ML compiles the package (a
    /// damaged one fails there) and the Kit reads its declared input and output (a model that
    /// is not a detector it can decode fails there, with the reason). Loaded on the CPU only:
    /// nothing is compiled for the GPU or the Neural Engine just to check. Takes seconds for
    /// a package that still has to be compiled - keep it off the main thread.
    func loadProblem() -> String? {
        do {
            _ = try makeDetector(compute: ComputeMode("cpu"))
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    /// This model, loaded for inference - the one way the app creates a `Detector`.
    ///
    /// A top-down pose model (DWPose) does not find people itself: the Kit runs a detection
    /// model on the frame first and the pose model once per person. The Kit asks for that
    /// detector only when the package turns out to be top-down, so every other model loads
    /// exactly as before.
    func makeDetector(compute: ComputeMode) throws -> Detector {
        let own = id
        let det = try Detector(modelURL: try compiledURL(), compute: compute, personDetector: {
            try BundledModel.personDetector(compute: compute, excluding: own)
        })
        ModelFacts.note(own, loaded: det)   // no-op when the facts were already read from disk
        return det
    }

    /// The detector that finds the people for a top-down pose model: a plain detection model
    /// in the app whose classes include "person". yolo26n first - the pose model's accuracy
    /// was measured behind it - then the other stock nano detectors, then anything else by
    /// name. Models named as segmentation or pose are not tried (they would cost a load each
    /// to find out), and one that fails to load is skipped.
    static func personDetector(compute: ComputeMode, excluding: String) throws -> Detector {
        let preferred = ["yolo26n", "yolo11n", "yolo12n", "yolov8n", "yolov5nu"]
        let candidates = discover().filter { m in
            let name = m.id.lowercased()
            return m.id != excluding && !name.contains("seg") && !name.contains("pose")
                && !name.contains("wholebody") && !name.hasPrefix("rtm")
        }
        let ordered = candidates.sorted { a, b in
            let ra = preferred.firstIndex(of: a.id.lowercased()) ?? preferred.count
            let rb = preferred.firstIndex(of: b.id.lowercased()) ?? preferred.count
            return ra != rb ? ra < rb : a.id < b.id
        }
        for m in ordered {
            guard let url = try? m.compiledURL(),
                  let det = try? Detector(modelURL: url, compute: compute) else { continue }
            if !det.isSegment, !det.isPose, det.classNames.contains("person") { return det }
        }
        throw DetectorError.needsPersonDetector
    }
}

// MARK: - Model type and attributes

/// What a model does. Shown as an icon beside the model's name in every model picker and
/// spelled out in Settings > Models.
enum ModelKind: String {
    case detection, segmentation, pose, wholeBody

    var label: String {
        switch self {
        case .detection: return "Detection"
        case .segmentation: return "Segmentation"
        case .pose: return "Pose"
        case .wholeBody: return "Whole-body pose"
        }
    }

    /// SF Symbol standing for the kind: a box finder, a traced outline, a figure, a hand.
    var icon: String {
        switch self {
        case .detection: return "viewfinder"
        case .segmentation: return "lasso"
        case .pose: return "figure.stand"
        case .wholeBody: return "hand.raised"
        }
    }
}

/// One declared input or output of a model.
struct ModelPort: Hashable {
    let name: String
    let shape: String      // "1 x 3 x 640 x 640"; empty when the model leaves it open
    let dataType: String   // "Float16"
}

/// What can be said about a model WITHOUT loading it. Core ML's compiler writes a
/// `metadata.json` into every .mlmodelc - the exporter's own metadata (task, names, kpt_shape,
/// precision ...), the declared inputs and outputs, the storage precision - so a model's type
/// and attributes cost one small file read. Bundled models are compiled by Xcode and imported
/// ones by the import check, so the file is normally there. Where it is not (a package that
/// has not been compiled yet), the kind is guessed from the file name until the model has
/// been loaded once (`note`).
struct ModelFacts {
    var kind: ModelKind
    var kindIsGuess = false       // from the file name only
    var topDown = false           // pose model that runs once per person behind a detector
    var end2end = false           // NMS-free output
    var keypoints = 0             // per person; 0 unless a pose model
    var classNames: [String] = []
    var inputs: [ModelPort] = []
    var outputs: [ModelPort] = []
    var precision: String? = nil          // the exporter's stamp: fp16, w8a16
    var storagePrecision: String? = nil   // Core ML's reading of the stored weights
    var computePrecision: String? = nil
    var format: String? = nil             // "ML Program", "Neural network"
    var specVersion: Int? = nil
    var minimumIOS: String? = nil
    var operations: Int? = nil
    var summary: String? = nil
    var author: String? = nil
    var license: String? = nil
    var version: String? = nil
    var metadata: [String: String] = [:]  // the exporter's metadata as stamped

    private static var cache: [String: ModelFacts] = [:]
    private static let lock = NSLock()

    /// The facts of `m`: cached, else read from its compiled form, else a guess from its name
    /// (not cached, so it is replaced as soon as the model has been compiled or loaded).
    static func of(_ m: BundledModel) -> ModelFacts {
        lock.lock()
        let hit = cache[m.id]
        lock.unlock()
        if let hit { return hit }
        if let compiled = m.compiledIfReady, let facts = read(compiled) {
            lock.lock()
            cache[m.id] = facts
            lock.unlock()
            return facts
        }
        return ModelFacts(kind: guessKind(m.id), kindIsGuess: true)
    }

    /// A model that has just been loaded tells what it is, whatever its files say.
    static func note(_ id: String, loaded det: Detector) {
        lock.lock()
        defer { lock.unlock() }
        guard cache[id] == nil else { return }
        let kind: ModelKind
        if det.isSegment {
            kind = .segmentation
        } else if det.isPose {
            kind = det.keypointCount == 133 ? .wholeBody : .pose
        } else {
            kind = .detection
        }
        var f = ModelFacts(kind: kind)
        f.topDown = det.isTopDown
        f.end2end = det.end2end
        f.keypoints = det.isPose ? det.keypointCount : 0
        f.classNames = det.classNames
        f.precision = det.metadata["precision"]
        f.metadata = det.metadata
        cache[id] = f
    }

    static func forget(_ id: String) {
        lock.lock()
        cache[id] = nil
        lock.unlock()
    }

    /// The Kit's own rule (`Detector.init`): the exporter's `task`, with `kpt_shape` telling a
    /// 133-keypoint whole-body model from a 17-keypoint one.
    private static func kindOf(task: String, keypoints: Int) -> ModelKind {
        switch task {
        case "segment": return .segmentation
        case "pose", "pose-topdown": return keypoints == 133 ? .wholeBody : .pose
        default: return .detection
        }
    }

    private static func guessKind(_ id: String) -> ModelKind {
        let n = id.lowercased()
        if n.contains("wholebody") || n.hasPrefix("dwpose") || n.hasPrefix("rtmw") { return .wholeBody }
        if n.contains("pose") { return .pose }
        if n.contains("seg") { return .segmentation }
        return .detection
    }

    private static func read(_ compiled: URL) -> ModelFacts? {
        guard let data = try? Data(contentsOf: compiled.appendingPathComponent("metadata.json")),
              let root = try? JSONSerialization.jsonObject(with: data) else { return nil }
        // the file is an array holding one object
        guard let top = (root as? [[String: Any]])?.first ?? (root as? [String: Any]) else { return nil }
        let user = (top["userDefinedMetadata"] as? [String: Any] ?? [:]).compactMapValues { $0 as? String }
        let task = user["task"] ?? "detect"
        let isPose = task == "pose" || task == "pose-topdown"
        let keypoints: Int = {
            guard isPose else { return 0 }
            let parts = (user["kpt_shape"] ?? "").split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            return parts.first.map { $0 > 0 ? $0 : 17 } ?? 17
        }()
        func ports(_ key: String) -> [ModelPort] {
            (top[key] as? [[String: Any]] ?? []).map { p in
                let raw = (p["shape"] as? String ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "[] "))
                let dims = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                return ModelPort(name: p["name"] as? String ?? "?",
                                 shape: dims.joined(separator: " x "),
                                 dataType: p["dataType"] as? String ?? "")
            }
        }
        func text(_ key: String) -> String? {
            guard let v = top[key] as? String, !v.isEmpty else { return nil }
            return v
        }
        var f = ModelFacts(kind: kindOf(task: task, keypoints: keypoints))
        f.topDown = task == "pose-topdown"
        f.end2end = user["end2end"] == "true"
        f.keypoints = keypoints
        f.classNames = (user["names"] ?? "").split(separator: ",").map(String.init)
        f.inputs = ports("inputSchema")
        f.outputs = ports("outputSchema")
        f.precision = user["precision"]
        f.storagePrecision = text("storagePrecision")
        f.computePrecision = text("computePrecision")
        if let type = (top["modelType"] as? [String: Any])?["name"] as? String {
            switch type {
            case "MLModelType_mlProgram": f.format = "ML Program"
            case "MLModelType_neuralNetwork": f.format = "Neural network"
            default: f.format = type.replacingOccurrences(of: "MLModelType_", with: "")
            }
        }
        f.specVersion = top["specificationVersion"] as? Int
        f.minimumIOS = (top["availability"] as? [String: Any])?["iOS"] as? String
        if let histogram = top["mlProgramOperationTypeHistogram"] as? [String: Any] {
            f.operations = histogram.values.compactMap { $0 as? Int }.reduce(0, +)
        }
        f.summary = text("shortDescription")
        f.author = text("author")
        f.license = text("license")
        f.version = text("version")
        f.metadata = user
        return f
    }
}

extension BundledModel {
    /// The compiled model on disk if there is one already. Never compiles.
    var compiledIfReady: URL? {
        if url.pathExtension.lowercased() == "mlmodelc" { return url }
        let cached = BundledModel.cacheDir.appendingPathComponent(id + ".mlmodelc")
        return FileManager.default.fileExists(atPath: cached.path) ? cached : nil
    }

    /// Type and attributes, without loading the model (`ModelFacts`).
    var facts: ModelFacts { ModelFacts.of(self) }
    var kind: ModelKind { facts.kind }

    /// Imported in Settings (false: shipped inside the app).
    var isImported: Bool { url.path.hasPrefix(BundledModel.customModelsDir.path) }

    /// Bytes the model occupies in the app: the package or compiled folder it is loaded from.
    /// Walks the folder - keep it off the main thread.
    var diskSize: Int64 {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return 0 }
        if !isDirectory.boolValue {
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
            return (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        }
        var total: Int64 = 0
        if let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let file as URL in walker {
                total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
            }
        }
        return total
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
        guard let major = iPhoneHardwareGeneration else { return .auto }
        return major <= 16 ? .gpu : .auto
    }
}

/// The phone's hardware generation: N in the hardware identifier "iPhoneN,M" - 16 for the
/// iPhone 15 Pro / Pro Max, 17 for the iPhone 16 family, 18 for the iPhone 17 family and the
/// iPhone Air. nil when this is not an iPhone (iPad, simulator).
let iPhoneHardwareGeneration: Int? = {
    var sys = utsname(); uname(&sys)
    let id = Mirror(reflecting: sys.machine).children.compactMap { c -> String? in
        guard let v = c.value as? Int8, v != 0 else { return nil }
        return String(UnicodeScalar(UInt8(v)))
    }.joined()                                   // e.g. "iPhone16,1"
    guard id.hasPrefix("iPhone") else { return nil }
    return Int(id.dropFirst(6).prefix { $0.isNumber })
}()
