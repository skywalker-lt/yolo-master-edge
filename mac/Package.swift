// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YOLOMaster",
    platforms: [.macOS("14.0")],   // Sonoma+. Floor is onKeyPress + zero-param onChange (SwiftUI 14); every other API used is 12–13 (Canvas/AV-async 13, Float16 MLMultiArray 12).
    products: [
        .library(name: "YOLOMasterKit", targets: ["YOLOMasterKit"]),
        .executable(name: "yolomaster-coreml", targets: ["YOLOMasterCoreML"]),   // CLI runner
        .executable(name: "YOLOMasterApp", targets: ["YOLOMasterApp"]),          // SwiftUI GUI
    ],
    targets: [
        // Shared Core ML inference backend (letterbox -> predict -> decode -> NMS -> annotate).
        .target(name: "YOLOMasterKit", path: "Sources/YOLOMasterKit"),
        // Command-line frontend.
        .executableTarget(
            name: "YOLOMasterCoreML",
            dependencies: ["YOLOMasterKit"],
            path: "Sources/YOLOMasterCoreML"
        ),
        // SwiftUI app frontend (same backend).
        .executableTarget(
            name: "YOLOMasterApp",
            dependencies: ["YOLOMasterKit"],
            path: "Sources/YOLOMasterApp"
        ),
    ]
)
