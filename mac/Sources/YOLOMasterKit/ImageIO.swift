// Shared image load/save helpers (used by both the CLI and the app).
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public func loadCGImage(_ url: URL) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    let orient = (props?[kCGImagePropertyOrientation] as? Int) ?? 1
    if orient == 1 { return CGImageSourceCreateImageAtIndex(src, 0, nil) }
    // EXIF orientation != up (e.g. portrait iPhone photos): bake the rotation/flip into the pixels so
    // inference AND the annotated overlay share one upright frame. Thumbnail-with-transform applies the
    // full 8-case orientation matrix; sizing the cap to the long edge keeps it full-resolution (no downscale).
    let pw = (props?[kCGImagePropertyPixelWidth] as? Int) ?? 0
    let ph = (props?[kCGImagePropertyPixelHeight] as? Int) ?? 0
    let cap = max(pw, ph, 1) > 1 ? max(pw, ph) : 16384
    let opts: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: cap,
    ]
    return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
        ?? CGImageSourceCreateImageAtIndex(src, 0, nil)
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
