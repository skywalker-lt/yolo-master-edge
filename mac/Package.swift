// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YOLOMasterCoreML",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "YOLOMasterCoreML", path: "Sources/YOLOMasterCoreML")
    ]
)
