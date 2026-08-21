// Model discovery + compute-unit selection for the iOS app.
// .mlpackage bundles are compiled by Xcode into .mlmodelc resources; drop the
// p03 packages into ios/Models/ (see ios/README.md) and they appear here.
import Foundation
import YOLOMasterKit

struct BundledModel: Identifiable, Hashable {
    let id: String       // display name, e.g. "p03_v01n_coco_fp16"
    let url: URL         // compiled .mlmodelc inside the app bundle

    static func discover() -> [BundledModel] {
        let exts = ["mlmodelc"]
        var found: [BundledModel] = []
        for ext in exts {
            for url in Bundle.main.urls(forResourcesWithExtension: ext, subdirectory: nil) ?? [] {
                found.append(BundledModel(id: url.deletingPathExtension().lastPathComponent, url: url))
            }
        }
        return found.sorted { $0.id < $1.id }
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
