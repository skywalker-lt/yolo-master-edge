// Shared detection-overlay renderer: ONE implementation for every tab so the
// annotation style cannot drift (green 2px box, green chip, black label).
import SwiftUI
import YOLOMasterKit

enum DetOverlay {
    static func draw(_ ctx: GraphicsContext, _ dets: [Detection],
                     scale: CGFloat, ox: CGFloat, oy: CGFloat) {
        for d in dets {
            let r = CGRect(x: d.rect.minX * scale + ox, y: d.rect.minY * scale + oy,
                           width: d.rect.width * scale, height: d.rect.height * scale)
            ctx.stroke(Path(roundedRect: r, cornerRadius: 3),
                       with: .color(.green), lineWidth: 2)
            let name = d.cls < cocoNames.count ? cocoNames[d.cls] : "\(d.cls)"
            let t = ctx.resolve(Text("\(name) \(Int(d.score * 100))%")
                .font(.caption2.bold()).foregroundStyle(.black))
            let sz = t.measure(in: CGSize(width: 320, height: 40))
            let chip = CGRect(x: r.minX, y: max(r.minY - sz.height - 4, 2),
                              width: sz.width + 8, height: sz.height + 3)
            ctx.fill(Path(roundedRect: chip, cornerRadius: 3), with: .color(.green))
            ctx.draw(t, at: CGPoint(x: chip.minX + 4, y: chip.midY), anchor: .leading)
        }
    }
}
