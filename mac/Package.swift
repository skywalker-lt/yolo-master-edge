// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YOLOMasterCoreML",
    platforms: [.macOS("15.0")],   // 15.0 deployment target (Float16 MLMultiArray access); string form keeps tools 5.9
    targets: [
        .executableTarget(name: "YOLOMasterCoreML", path: "Sources/YOLOMasterCoreML")
    ]
)
