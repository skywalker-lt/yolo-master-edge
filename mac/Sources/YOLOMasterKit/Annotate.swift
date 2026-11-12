// Annotation - HUD / solid / neon box templates + translucent label pills.
// Extracted verbatim from the CLI runner so CLI and app render identically.
import Foundation
import CoreGraphics
import CoreText

public enum BoxStyle: String, CaseIterable, Sendable { case hud, solid, neon }
public enum LabelMode: String, CaseIterable, Sendable { case full, min, off }
/// Segmentation display: masks only (boxes removed), boxes only, or both. Ignored for detectors.
public enum SegOverlay: String, CaseIterable, Sendable { case masks, boxes, both }

private let palette: [CGColor] = [
    (0.98, 0.26, 0.30), (0.20, 0.71, 0.98), (0.16, 0.85, 0.52), (0.99, 0.79, 0.12),
    (0.72, 0.40, 0.98), (0.99, 0.55, 0.18), (0.10, 0.83, 0.80), (0.98, 0.36, 0.66),
    (0.55, 0.82, 0.28), (0.40, 0.52, 0.98),
].map { CGColor(red: CGFloat($0.0), green: CGFloat($0.1), blue: CGFloat($0.2), alpha: 1) }

/// Stable per-class color (shared by boxes and segmentation masks).
public func classColor(_ cls: Int) -> CGColor { palette[((cls % palette.count) + palette.count) % palette.count] }

private func labelTextColor(on bg: CGColor) -> CGColor {
    let c = bg.components ?? [0, 0, 0]
    let lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return lum > 0.62 ? CGColor(gray: 0.05, alpha: 1) : CGColor(gray: 1, alpha: 1)
}

/// Pose rendering rules shared by every frontend (the baked `annotate` output below and the
/// SwiftUI overlays in the apps): which keypoints form a limb, how each body part is coloured,
/// and the visibility below which a keypoint is not drawn.
public enum PoseSkeleton {
    /// A keypoint the model is less than this sure it can see is not drawn, nor are its limbs.
    public static let minConfidence: Float = 0.5

    /// The COCO-17 skeleton (the ultralytics pose topology). `part`: 0 head, 1 arms, 2 torso, 3 legs.
    /// Keypoint order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
    /// 11-12 hips, 13-14 knees, 15-16 ankles.
    public static let coco17Limbs: [(a: Int, b: Int, part: Int)] = [
        (15, 13, 3), (13, 11, 3), (16, 14, 3), (14, 12, 3),                              // legs
        (11, 12, 2), (5, 11, 2), (6, 12, 2), (5, 6, 2),                                  // torso
        (5, 7, 1), (6, 8, 1), (7, 9, 1), (8, 10, 1),                                     // arms
        (1, 2, 0), (0, 1, 0), (0, 2, 0), (1, 3, 0), (2, 4, 0), (3, 5, 0), (4, 6, 0),     // head
    ]

    /// The COCO-WholeBody skeleton (133 keypoints): the 17 body keypoints first, in COCO order,
    /// then 17-22 feet (big toe, small toe, heel, left then right), 23-90 the 68 face points,
    /// 91-111 the left hand and 112-132 the right hand (wrist, then four joints per finger from
    /// thumb to little finger). Body limbs, each ankle to its toes and heel, and the finger
    /// bones are drawn; the face is points only.
    public static let wholeBodyLimbs: [(a: Int, b: Int, part: Int)] = {
        var all: [(a: Int, b: Int, part: Int)] = PoseSkeleton.coco17Limbs
        let feet: [(a: Int, b: Int, part: Int)] = [
            (15, 17, 3), (15, 18, 3), (15, 19, 3), (16, 20, 3), (16, 21, 3), (16, 22, 3),
        ]
        all.append(contentsOf: feet)
        for wrist in [91, 112] {
            for finger in 0..<5 {
                let first = wrist + 1 + finger * 4
                all.append((a: wrist, b: first, part: 1))
                for joint in 0..<3 {
                    all.append((a: first + joint, b: first + joint + 1, part: 1))
                }
            }
        }
        return all
    }()

    /// Limbs to draw for a pose of `count` keypoints: the COCO skeleton for 17, the whole-body
    /// one for 133, none otherwise (a custom keypoint set has no known topology here, so only
    /// its joints are drawn).
    public static func limbs(forKeypointCount count: Int) -> [(a: Int, b: Int, part: Int)] {
        switch count {
        case 17: return coco17Limbs
        case 133: return wholeBodyLimbs
        default: return []
        }
    }

    /// Body part of keypoint `index` in a pose of `count` keypoints (0 when the layout is unknown).
    public static func part(ofKeypoint index: Int, count: Int) -> Int {
        guard count == 17 || count == 133 else { return 0 }
        switch index {
        case 0...4: return 0        // head
        case 5...10: return 1       // arms
        case 11...22: return 3      // legs, and the feet of the whole-body layout
        case 23...90: return 0      // face
        default: return 1           // hands
        }
    }

    /// A keypoint of the whole-body layout beyond the 17 body joints (feet, face, hands). There
    /// are 116 of them, 68 on the face alone, so they are drawn at about half size, and so are
    /// the limbs that end in one.
    public static func isFine(keypoint index: Int, count: Int) -> Bool {
        count == 133 && index >= 17
    }

    /// Colour of a body part, from the class palette: head green, arms blue, torso purple, legs orange.
    public static func color(part: Int) -> CGColor {
        switch part {
        case 0: return palette[2]
        case 1: return palette[1]
        case 2: return palette[4]
        default: return palette[5]
        }
    }
}

/// One pose skeleton into a bottom-left-origin context of height `h` (keypoints are top-left origin).
private func drawPose(_ ctx: CGContext, _ kpts: [Keypoint], imageHeight h: CGFloat, lineWidth lw: CGFloat) {
    func pt(_ k: Keypoint) -> CGPoint { CGPoint(x: k.x, y: h - k.y) }
    let minConf = PoseSkeleton.minConfidence
    ctx.saveGState()
    ctx.setLineCap(.round)
    for limb in PoseSkeleton.limbs(forKeypointCount: kpts.count) {
        let a = kpts[limb.a], b = kpts[limb.b]
        guard a.conf >= minConf, b.conf >= minConf else { continue }
        let fine = PoseSkeleton.isFine(keypoint: max(limb.a, limb.b), count: kpts.count)
        ctx.setStrokeColor(PoseSkeleton.color(part: limb.part))
        ctx.setLineWidth(lw * (fine ? 0.7 : 1.3))
        ctx.move(to: pt(a)); ctx.addLine(to: pt(b)); ctx.strokePath()
    }
    for (i, k) in kpts.enumerated() where k.conf >= minConf {
        let fine = PoseSkeleton.isFine(keypoint: i, count: kpts.count)
        let r = lw * (fine ? 1 : 2)
        let c = pt(k)
        let dot = CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)
        ctx.setFillColor(PoseSkeleton.color(part: PoseSkeleton.part(ofKeypoint: i, count: kpts.count)))
        ctx.fillEllipse(in: dot)
        ctx.setStrokeColor(CGColor(gray: 1, alpha: 0.9))
        ctx.setLineWidth(lw * (fine ? 0.25 : 0.5))
        ctx.strokeEllipse(in: dot)
    }
    ctx.restoreGState()
}

/// Per-track colour: the same (k*37, k*91, k*173) mod 255 rule as the C++ draw_tracks.
func trackColor(_ id: Int) -> CGColor {
    CGColor(red: CGFloat((id * 37) % 255) / 255, green: CGFloat((id * 91) % 255) / 255,
            blue: CGFloat((id * 173) % 255) / 255, alpha: 1)
}

/// Draw detections onto `image`, returning a new annotated CGImage.
/// `masks` (segmentation) are composited under the boxes; pass `drawBoxes: false` to render masks only.
public func annotate(_ image: CGImage, _ dets: [Detection], names: [String],
                     style: BoxStyle = .hud, label: LabelMode = .full,
                     masks: [MaskBitmap] = [], drawBoxes: Bool = true) -> CGImage? {
    let w = image.width, h = image.height
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))  // no flip; boxes convert y = h - topY
    // ---- segmentation masks (under the boxes) ----
    ctx.interpolationQuality = .high   // bilinear upscale proto->full res -> smooth (non-serrated) mask edges
    for m in masks {
        let pc = m.protoCrop, iw = CGFloat(m.image.width), ih = CGFloat(m.image.height)
        let sub = CGRect(x: pc.minX * iw, y: pc.minY * ih, width: pc.width * iw, height: pc.height * ih)
        guard sub.width > 0, sub.height > 0, let cropped = m.image.cropping(to: sub) else { continue }
        ctx.saveGState()
        // clip to this instance's box (context is bottom-left origin -> flip y)
        ctx.clip(to: CGRect(x: m.clip.minX, y: CGFloat(h) - m.clip.maxY, width: m.clip.width, height: m.clip.height))
        ctx.draw(cropped, in: CGRect(x: 0, y: 0, width: w, height: h))  // proto crop stretched to full image, upright
        ctx.restoreGState()
    }
    let lw = max(CGFloat(2), CGFloat(w) / 640)
    // pose skeletons go on last, over the boxes and labels (no-op for detections without keypoints)
    func drawPoses() {
        for d in dets where !d.keypoints.isEmpty {
            drawPose(ctx, d.keypoints, imageHeight: CGFloat(h), lineWidth: lw)
        }
    }
    if !drawBoxes && label == .off { drawPoses(); return ctx.makeImage() }   // masks-only, no labels
    ctx.setLineJoin(.round); ctx.setLineCap(.round)
    let baseFont = max(CGFloat(12), CGFloat(w) / 95)
    for d in dets {
        // tracked detections are coloured by id (the C++ draw_tracks palette rule), so one object
        // keeps its colour across frames regardless of class
        let color = d.trackId.map { trackColor($0) } ?? palette[d.cls % palette.count]
        let box = CGRect(x: d.rect.minX, y: CGFloat(h) - d.rect.maxY, width: d.rect.width, height: d.rect.height)
        let r = min(min(box.width, box.height) * 0.14, lw * 5)
        let rpath = CGPath(roundedRect: box, cornerWidth: r, cornerHeight: r, transform: nil)
        // ---- box template (skipped when boxes are hidden, e.g. segmentation masks-only) ----
        if drawBoxes {
            switch style {
            case .solid:                                    // clean rounded rectangle
                ctx.addPath(rpath); ctx.setStrokeColor(color); ctx.setLineWidth(lw * 1.2); ctx.strokePath()
            case .neon:                                     // glowing rounded rectangle
                ctx.saveGState()
                ctx.setShadow(offset: .zero, blur: lw * 5, color: color.copy(alpha: 0.95) ?? color)
                ctx.addPath(rpath); ctx.setStrokeColor(color); ctx.setLineWidth(lw * 1.3); ctx.strokePath()
                ctx.addPath(rpath); ctx.strokePath()        // second pass = denser glow
                ctx.restoreGState()
            case .hud:                                      // faint fill + thin outline + corner brackets
                ctx.addRect(box); ctx.setFillColor(color.copy(alpha: 0.08) ?? color); ctx.fillPath()
                ctx.addRect(box); ctx.setStrokeColor(color.copy(alpha: 0.35) ?? color); ctx.setLineWidth(lw * 0.6); ctx.strokePath()
                let arm = min(min(box.width, box.height) * 0.28, lw * 22)
                ctx.setStrokeColor(color); ctx.setLineWidth(lw * 1.4)
                for (cx, cy, sx, sy) in [(box.minX, box.minY, 1.0, 1.0), (box.maxX, box.minY, -1.0, 1.0),
                                         (box.minX, box.maxY, 1.0, -1.0), (box.maxX, box.maxY, -1.0, -1.0)] {
                    ctx.move(to: CGPoint(x: cx + arm * CGFloat(sx), y: cy))
                    ctx.addLine(to: CGPoint(x: cx, y: cy))
                    ctx.addLine(to: CGPoint(x: cx, y: cy + arm * CGFloat(sy)))
                }
                ctx.strokePath()
            }
        }
        // ---- label (full | min | off) ----
        if label == .off { continue }
        let minMode = label == .min
        let name = d.cls < names.count ? names[d.cls] : "class\(d.cls)"
        let idPrefix = d.trackId.map { "#\($0) " } ?? ""
        let text = minMode ? idPrefix + name : "\(idPrefix)\(name)  \(String(format: "%.2f", d.score))"
        let fontSize = minMode ? baseFont * 0.85 : baseFont
        let font = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, fontSize, nil)
        let attr = NSAttributedString(string: text, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): labelTextColor(on: color)])
        let line = CTLineCreateWithAttributedString(attr)
        let tw = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        let padX = fontSize * 0.5, chipH = fontSize + 6, chipW = tw + padX * 2
        var chipY = box.maxY - lw / 2
        if chipY + chipH > CGFloat(h) { chipY = box.maxY - chipH }
        let chip = CGRect(x: box.minX - lw / 2, y: chipY, width: chipW, height: chipH)
        let chipPath = CGPath(roundedRect: chip, cornerWidth: chipH * 0.28, cornerHeight: chipH * 0.28, transform: nil)
        ctx.addPath(chipPath); ctx.setFillColor(color.copy(alpha: minMode ? 0.55 : 0.72) ?? color); ctx.fillPath()
        ctx.textPosition = CGPoint(x: chip.minX + padX, y: chipY + (chipH - fontSize) / 2 + fontSize * 0.2)
        CTLineDraw(line, ctx)
    }
    drawPoses()
    return ctx.makeImage()
}
