// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YOLOMasterCoreML",
    platforms: [.macOS(.v15)],   // Float16 MLMultiArray access + the model's macOS15 deployment target
    targets: [
        .executableTarget(name: "YOLOMasterCoreML", path: "Sources/YOLOMasterCoreML")
    ]
)
