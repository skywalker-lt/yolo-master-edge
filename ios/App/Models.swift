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
    /// Canonical display name: every model is presented as YOLO-Master-<stem>
    /// (existing YOLO-Master prefixes are normalized, never doubled).
    var fullName: String { "YOLO-Master-" + stem }
    /// Collapsed menu-bar label: the first 6 characters of the model's own name
    /// (the stem - the shared YOLO-Master prefix carries no information there).
    var shortID: String { stem.count > 6 ? String(stem.prefix(6)) + "..." : stem }

    private static var cacheDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
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
        return found.sorted { $0.id < $1.id }
    }

    /// Launch default: the bundled segmentation model when present (v0.1-seg-N,
    /// the richest demo), else the first model alphabetically.
    static func preferred(in models: [BundledModel]) -> BundledModel? {
        models.first { $0.id.localizedCaseInsensitiveContains("seg") } ?? models.first
    }
}

/// UI wrapper over the Kit's ComputeMode. `.all` routes to the ANE - on A-series
/// silicon that is the expected winner (near-desktop ANE, small GPU); the
/// per-unit bench exists to verify exactly that on-device.
enum ComputeChoice: String, CaseIterable, Identifiable {
    case ane = "ANE"        // CPU + GPU + Neural Engine (.all)
    case gpu = "GPU"        // CPU + GPU
    case cpu = "CPU"        // CPU only
    var id: String { rawValue }
    var mode: ComputeMode {
        switch self {
        case .ane: return ComputeMode("all")
        case .gpu: return ComputeMode("cpuandgpu")
        case .cpu: return ComputeMode("cpu")
        }
    }
}
