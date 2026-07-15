// Shared image load/save helpers (used by both the CLI and the app).
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public func loadCGImage(_ url: URL) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

public func saveCGImage(_ image: CGImage, to url: URL) {
    let type: CFString = url.pathExtension.lowercased() == "png"
        ? UTType.png.identifier as CFString
        : UTType.jpeg.identifier as CFString
    if let dest = CGImageDestinationCreateWithURL(url as CFURL, type, 1, nil) {
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }
}
