// Shared detection-overlay renderer with selectable styles.
// solid / neon / hud mirror the Mac Kit's Annotate styles (same 10-color class
// palette); chip is the classic label-plate look; minimal is a thin monochrome.
import SwiftUI
import UIKit
import YOLOMasterKit

enum BoxStyle: String, CaseIterable, Identifiable {
    case solid, neon, hud, chip, minimal
    var id: String { rawValue }
    var label: String {
        switch self {
        case .solid: return "Solid"
        case .neon: return "Neon"
        case .hud: return "HUD"
        case .chip: return "Chip"
        case .minimal: return "Minimal"
        }
    }
}

/// COCO-80 fallback when a model carries no usable names metadata.
let cocoNames = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

/// Same palette as mac/Sources/YOLOMasterKit/Annotate.swift and the C++ runner.
private let palette: [Color] = [
    Color(red: 0.98, green: 0.26, blue: 0.30), Color(red: 0.20, green: 0.71, blue: 0.98),
    Color(red: 0.16, green: 0.85, blue: 0.52), Color(red: 0.99, green: 0.79, blue: 0.12),
    Color(red: 0.72, green: 0.40, blue: 0.98), Color(red: 0.99, green: 0.55, blue: 0.18),
    Color(red: 0.10, green: 0.83, blue: 0.80), Color(red: 0.98, green: 0.36, blue: 0.66),
    Color(red: 0.55, green: 0.82, blue: 0.28), Color(red: 0.40, green: 0.52, blue: 0.98),
]
private let darkText: Set<Int> = [2, 3, 6, 8]   // light palette entries -> black label text

extension BoxStyle {
    /// Nearest Kit `annotate` style for baked exports (chip/minimal have no
    /// direct Kit equivalent).
    var kitStyle: YOLOMasterKit.BoxStyle {
        switch self {
        case .neon: return .neon
        case .hud, .minimal: return .hud
        default: return .solid
        }
    }
}

enum DetOverlay {
    /// `boxes: false` (segmentation masks-only mode) keeps the per-style labels but
    /// skips the box geometry - the mask carries the shape.
    /// `mirrorWidth`: when set (the frame's width in frame pixels), every box is
    /// flipped left-to-right inside the frame. The front camera's preview is a
    /// mirror image while its frames are not, so its boxes must flip to land on
    /// what the user sees. Only positions flip - label text stays readable.
    /// `labels: false` drops the label too (a pose model in skeleton-only mode: with
    /// `boxes: false` nothing but the skeleton is left).
    /// `pose`: how the skeletons of a pose model are drawn - which keypoints, from what
    /// visibility, at what size (the Kit's `PoseDisplay`, also used for saved photos).
    /// `boxScale`: the "box" lever of Photo's tuning panel. Every box line is that many
    /// times as thick and every label that many times as large; the boxes' shape (corner
    /// radius, the HUD style's bracket arms) and the skeleton, which has its own size in
    /// `pose`, are not touched. At 1 - what Live always passes - everything is drawn exactly
    /// as it was before the lever existed.
    static func draw(_ ctx: GraphicsContext, _ dets: [Detection],
                     scale: CGFloat, ox: CGFloat, oy: CGFloat,
                     style: BoxStyle = .chip, names: [String] = cocoNames,
                     boxes: Bool = true, labels: Bool = true, mirrorWidth: CGFloat? = nil,
                     pose: PoseDisplay = PoseDisplay(), boxScale: CGFloat = 1) {
        let k = boxScale
        for d in dets where boxes || labels {
            let left = mirrorWidth.map { $0 - d.rect.maxX } ?? d.rect.minX
            let r = CGRect(x: left * scale + ox, y: d.rect.minY * scale + oy,
                           width: d.rect.width * scale, height: d.rect.height * scale)
            let ci = ((d.cls % palette.count) + palette.count) % palette.count
            let col = palette[ci]
            let name = d.cls < names.count ? names[d.cls] : "\(d.cls)"
            let title = "\(name) \(Int(d.score * 100))%"
            let corner = min(min(r.width, r.height) * 0.14, 10)

            switch style {
            case .solid:
                if boxes {
                    ctx.stroke(Path(roundedRect: r, cornerRadius: corner),
                               with: .color(col), lineWidth: 2.4 * k)
                }
                if labels { chipLabel(ctx, title, at: r, color: col, dark: darkText.contains(ci), scale: k) }

            case .neon:
                if boxes {
                    var glow = ctx
                    glow.addFilter(.shadow(color: col.opacity(0.95), radius: 8 * k))
                    glow.stroke(Path(roundedRect: r, cornerRadius: corner),
                                with: .color(col), lineWidth: 2.6 * k)
                    ctx.stroke(Path(roundedRect: r, cornerRadius: corner),
                               with: .color(col), lineWidth: 1.2 * k)
                }
                if labels {
                    var t = ctx
                    t.addFilter(.shadow(color: col.opacity(0.9), radius: 5 * k))
                    let txt = t.resolve(Text(title).font(labelFont(k)).foregroundStyle(col))
                    t.draw(txt, at: CGPoint(x: r.minX + 2, y: max(r.minY - 9 * k, 6 * k)), anchor: .leading)
                }

            case .hud:
                if boxes {
                    ctx.fill(Path(r), with: .color(col.opacity(0.08)))
                    ctx.stroke(Path(r), with: .color(col.opacity(0.35)), lineWidth: 1 * k)
                    let arm = min(min(r.width, r.height) * 0.28, 26)
                    var p = Path()
                    for (cx, cy, sx, sy): (CGFloat, CGFloat, CGFloat, CGFloat) in
                        [(r.minX, r.minY, 1, 1), (r.maxX, r.minY, -1, 1),
                         (r.minX, r.maxY, 1, -1), (r.maxX, r.maxY, -1, -1)] {
                        p.move(to: CGPoint(x: cx + arm * sx, y: cy))
                        p.addLine(to: CGPoint(x: cx, y: cy))
                        p.addLine(to: CGPoint(x: cx, y: cy + arm * sy))
                    }
                    ctx.stroke(p, with: .color(col), lineWidth: 2.8 * k)
                }
                if labels {
                    let txt = ctx.resolve(Text(title.uppercased())
                        .font(.system(size: 9 * k, weight: .bold, design: .monospaced))
                        .foregroundStyle(col))
                    ctx.draw(txt, at: CGPoint(x: r.minX + 3, y: max(r.minY - 9 * k, 6 * k)), anchor: .leading)
                }

            case .chip:
                if boxes {
                    ctx.stroke(Path(roundedRect: r, cornerRadius: 3),
                               with: .color(col), lineWidth: 2 * k)
                }
                if labels { chipLabel(ctx, title, at: r, color: col, dark: darkText.contains(ci), scale: k) }

            case .minimal:
                if boxes {
                    ctx.stroke(Path(r), with: .color(.white.opacity(0.9)), lineWidth: 1 * k)
                }
                if labels {
                    let txt = ctx.resolve(Text(title)
                        .font(.system(size: 9 * k, weight: .medium))
                        .foregroundStyle(.white.opacity(0.9)))
                    ctx.draw(txt, at: CGPoint(x: r.minX + 2, y: max(r.minY - 8 * k, 5 * k)), anchor: .leading)
                }
            }
        }
        // pose skeletons last, over every box and label (no-op without keypoints)
        guard pose.drawsSkeleton else { return }
        for d in dets where !d.keypoints.isEmpty {
            drawPose(ctx, d.keypoints, scale: scale, ox: ox, oy: oy, mirrorWidth: mirrorWidth,
                     display: pose)
        }
    }

    /// One pose skeleton: limbs coloured by body part, then the joints on top. The topology and
    /// colours are the Kit's (`PoseSkeleton`), and which keypoints are drawn and how large is
    /// the Kit's too (`PoseDisplay`), so the live overlay and a saved photo draw the same thing.
    /// `mirrorWidth` flips x like the boxes (front camera).
    private static func drawPose(_ ctx: GraphicsContext, _ kpts: [Keypoint],
                                 scale: CGFloat, ox: CGFloat, oy: CGFloat, mirrorWidth: CGFloat?,
                                 display: PoseDisplay) {
        func pt(_ k: Keypoint) -> CGPoint {
            let x = mirrorWidth.map { $0 - k.x } ?? k.x
            return CGPoint(x: x * scale + ox, y: k.y * scale + oy)
        }
        let n = kpts.count
        let size = display.size
        for limb in PoseSkeleton.limbs(forKeypointCount: n) {
            let a = kpts[limb.a], b = kpts[limb.b]
            guard display.showsLimb(from: limb.a, to: limb.b, count: n),
                  display.shows(keypoint: limb.a, count: n, conf: a.conf),
                  display.shows(keypoint: limb.b, count: n, conf: b.conf) else { continue }
            // feet, face and hands of the whole-body layout are drawn at about half size
            let fine = PoseSkeleton.isFine(keypoint: max(limb.a, limb.b), count: kpts.count)
            var line = Path()
            line.move(to: pt(a))
            line.addLine(to: pt(b))
            ctx.stroke(line, with: .color(Color(cgColor: PoseSkeleton.color(part: limb.part))),
                       style: StrokeStyle(lineWidth: (fine ? 1.5 : 2.5) * size, lineCap: .round))
        }
        for (i, k) in kpts.enumerated() where display.shows(keypoint: i, count: n, conf: k.conf) {
            let fine = PoseSkeleton.isFine(keypoint: i, count: kpts.count)
            let r: CGFloat = (fine ? 2 : 3.5) * size
            let c = pt(k)
            let dot = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r))
            let part = PoseSkeleton.part(ofKeypoint: i, count: kpts.count)
            ctx.fill(dot, with: .color(Color(cgColor: PoseSkeleton.color(part: part))))
            ctx.stroke(dot, with: .color(.white.opacity(0.9)), lineWidth: (fine ? 0.5 : 1) * size)
        }
    }

    /// The font of the chip, solid and neon labels at `scale` times its size. At 1 it is the
    /// text style itself, as it always was; at any other scale it is the system font at that
    /// style's current point size (so it still follows the user's text size) times the scale.
    private static func labelFont(_ scale: CGFloat) -> Font {
        if scale == 1 { return .caption2.bold() }
        return .system(size: UIFont.preferredFont(forTextStyle: .caption2).pointSize * scale, weight: .bold)
    }

    /// `scale` enlarges the text and, with it, the plate's padding and corners (`boxScale`).
    private static func chipLabel(_ ctx: GraphicsContext, _ title: String,
                                  at r: CGRect, color: Color, dark: Bool, scale k: CGFloat = 1) {
        let t = ctx.resolve(Text(title).font(labelFont(k))
            .foregroundStyle(dark ? Color.black : Color.white))
        let sz = t.measure(in: CGSize(width: 320 * max(k, 1), height: 40 * max(k, 1)))
        let chip = CGRect(x: r.minX, y: max(r.minY - sz.height - 4 * k, 2),
                          width: sz.width + 8 * k, height: sz.height + 3 * k)
        ctx.fill(Path(roundedRect: chip, cornerRadius: 3 * k), with: .color(color))
        ctx.draw(t, at: CGPoint(x: chip.minX + 4 * k, y: chip.midY), anchor: .leading)
    }
}
