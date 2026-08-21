// Shared detection-overlay renderer with selectable styles.
// solid / neon / hud mirror the Mac Kit's Annotate styles (same 10-color class
// palette); chip is the classic label-plate look; minimal is a thin monochrome.
import SwiftUI
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

/// Same palette as mac/Sources/YOLOMasterKit/Annotate.swift and the C++ runner.
private let palette: [Color] = [
    Color(red: 0.98, green: 0.26, blue: 0.30), Color(red: 0.20, green: 0.71, blue: 0.98),
    Color(red: 0.16, green: 0.85, blue: 0.52), Color(red: 0.99, green: 0.79, blue: 0.12),
    Color(red: 0.72, green: 0.40, blue: 0.98), Color(red: 0.99, green: 0.55, blue: 0.18),
    Color(red: 0.10, green: 0.83, blue: 0.80), Color(red: 0.98, green: 0.36, blue: 0.66),
    Color(red: 0.55, green: 0.82, blue: 0.28), Color(red: 0.40, green: 0.52, blue: 0.98),
]
private let darkText: Set<Int> = [2, 3, 6, 8]   // light palette entries -> black label text

enum DetOverlay {
    static func draw(_ ctx: GraphicsContext, _ dets: [Detection],
                     scale: CGFloat, ox: CGFloat, oy: CGFloat,
                     style: BoxStyle = .chip) {
        for d in dets {
            let r = CGRect(x: d.rect.minX * scale + ox, y: d.rect.minY * scale + oy,
                           width: d.rect.width * scale, height: d.rect.height * scale)
            let ci = ((d.cls % palette.count) + palette.count) % palette.count
            let col = palette[ci]
            let name = d.cls < cocoNames.count ? cocoNames[d.cls] : "\(d.cls)"
            let title = "\(name) \(Int(d.score * 100))%"
            let corner = min(min(r.width, r.height) * 0.14, 10)

            switch style {
            case .solid:
                ctx.stroke(Path(roundedRect: r, cornerRadius: corner),
                           with: .color(col), lineWidth: 2.4)
                chipLabel(ctx, title, at: r, color: col, dark: darkText.contains(ci))

            case .neon:
                var glow = ctx
                glow.addFilter(.shadow(color: col.opacity(0.95), radius: 8))
                glow.stroke(Path(roundedRect: r, cornerRadius: corner),
                            with: .color(col), lineWidth: 2.6)
                ctx.stroke(Path(roundedRect: r, cornerRadius: corner),
                           with: .color(col), lineWidth: 1.2)
                var t = ctx
                t.addFilter(.shadow(color: col.opacity(0.9), radius: 5))
                let txt = t.resolve(Text(title).font(.caption2.bold()).foregroundStyle(col))
                t.draw(txt, at: CGPoint(x: r.minX + 2, y: max(r.minY - 9, 6)), anchor: .leading)

            case .hud:
                ctx.fill(Path(r), with: .color(col.opacity(0.08)))
                ctx.stroke(Path(r), with: .color(col.opacity(0.35)), lineWidth: 1)
                let arm = min(min(r.width, r.height) * 0.28, 26)
                var p = Path()
                for (cx, cy, sx, sy): (CGFloat, CGFloat, CGFloat, CGFloat) in
                    [(r.minX, r.minY, 1, 1), (r.maxX, r.minY, -1, 1),
                     (r.minX, r.maxY, 1, -1), (r.maxX, r.maxY, -1, -1)] {
                    p.move(to: CGPoint(x: cx + arm * sx, y: cy))
                    p.addLine(to: CGPoint(x: cx, y: cy))
                    p.addLine(to: CGPoint(x: cx, y: cy + arm * sy))
                }
                ctx.stroke(p, with: .color(col), lineWidth: 2.8)
                let txt = ctx.resolve(Text(title.uppercased())
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(col))
                ctx.draw(txt, at: CGPoint(x: r.minX + 3, y: max(r.minY - 9, 6)), anchor: .leading)

            case .chip:
                ctx.stroke(Path(roundedRect: r, cornerRadius: 3),
                           with: .color(col), lineWidth: 2)
                chipLabel(ctx, title, at: r, color: col, dark: darkText.contains(ci))

            case .minimal:
                ctx.stroke(Path(r), with: .color(.white.opacity(0.9)), lineWidth: 1)
                let txt = ctx.resolve(Text(title)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9)))
                ctx.draw(txt, at: CGPoint(x: r.minX + 2, y: max(r.minY - 8, 5)), anchor: .leading)
            }
        }
    }

    private static func chipLabel(_ ctx: GraphicsContext, _ title: String,
                                  at r: CGRect, color: Color, dark: Bool) {
        let t = ctx.resolve(Text(title).font(.caption2.bold())
            .foregroundStyle(dark ? Color.black : Color.white))
        let sz = t.measure(in: CGSize(width: 320, height: 40))
        let chip = CGRect(x: r.minX, y: max(r.minY - sz.height - 4, 2),
                          width: sz.width + 8, height: sz.height + 3)
        ctx.fill(Path(roundedRect: chip, cornerRadius: 3), with: .color(color))
        ctx.draw(t, at: CGPoint(x: chip.minX + 4, y: chip.midY), anchor: .leading)
    }
}
