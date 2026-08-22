// Model discovery + compute-unit selection for the iOS app.
// .mlpackage bundles are compiled by Xcode into .mlmodelc resources; drop the
// p03 packages into ios/Models/ (see ios/README.md) and they appear here.
import CoreML
import Foundation
import YOLOMasterKit

struct BundledModel: Identifiable, Hashable {
    let id: String       // bundle stem, e.g. "p03_v01n_coco_fp16"
    let url: URL         // compiled .mlmodelc inside the app bundle
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

    static func discover() -> [BundledModel] {
        var found: [BundledModel] = []
        // compiled models, bundle root or Models/ subdir
        for sub in [nil, "Models"] as [String?] {
            for url in Bundle.main.urls(forResourcesWithExtension: "mlmodelc",
                                        subdirectory: sub) ?? [] {
                found.append(BundledModel(id: url.deletingPathExtension().lastPathComponent,
                                          url: url))
            }
        }
        // uncompiled .mlpackage (folder-reference bundles): compile on first
        // launch, cache in Application Support
        for sub in [nil, "Models"] as [String?] {
            for pkg in Bundle.main.urls(forResourcesWithExtension: "mlpackage",
                                        subdirectory: sub) ?? [] {
                let name = pkg.deletingPathExtension().lastPathComponent
                if found.contains(where: { $0.id == name }) { continue }
                let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory,
                                                        in: .userDomainMask)[0]
                let cached = cacheDir.appendingPathComponent(name + ".mlmodelc")
                if FileManager.default.fileExists(atPath: cached.path) {
                    found.append(BundledModel(id: name, url: cached))
                } else if let tmp = try? MLModel.compileModel(at: pkg) {
                    try? FileManager.default.createDirectory(at: cacheDir,
                                                             withIntermediateDirectories: true)
                    try? FileManager.default.moveItem(at: tmp, to: cached)
                    found.append(BundledModel(id: name, url: cached))
                }
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
