// Shared stats HUD: FPS tachometer + per-stage latency bars + detection count.
// One component for Live and Photo so the style stays coherent.
import SwiftUI

struct StatsHUD: View {
    let fps: Double            // tachometer, 0-40 scale
    let pre: Double            // ms
    let inf: Double            // ms
    let dec: Double            // ms
    let dets: Int
    var fpsLabel: String = "FPS"

    var body: some View {
        HStack(spacing: 14) {
            Gauge(value: min(max(fps, 0), 40), in: 0...40) {
                Text(fpsLabel)
            } currentValueLabel: {
                Text("\(Int(fps.rounded()))")
                    .font(.system(.title3, design: .rounded).bold())
            }
            .gaugeStyle(.accessoryCircular)
            .tint(fps >= 25 ? .green : (fps >= 12 ? .yellow : .red))
            .frame(width: 56, height: 56)

            VStack(alignment: .leading, spacing: 5) {
                StageBar(label: "pre", ms: pre, color: .blue)
                StageBar(label: "inf", ms: inf, color: .green)
                StageBar(label: "dec", ms: dec, color: .orange)
            }

            VStack(spacing: 0) {
                Text("\(dets)")
                    .font(.system(.title2, design: .rounded).bold())
                    .foregroundStyle(.green)
                Text("dets").font(.caption2).foregroundStyle(.secondary)
            }
            .frame(width: 44)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

struct StageBar: View {
    let label: String
    let ms: Double
    let color: Color
    private let fullScale = 50.0   // bar saturates at 50ms

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2.monospaced())
                .frame(width: 24, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(color)
                        .frame(width: max(3, min(ms / fullScale, 1) * g.size.width))
                }
            }
            .frame(width: 90, height: 6)
            Text(String(format: "%4.1f", ms))
                .font(.caption2.monospaced())
                .frame(width: 36, alignment: .trailing)
        }
    }
}

/// conf / IoU tuning sliders - the Mac GUI's knobs, phone-sized.
struct TuningPanel: View {
    @Binding var conf: Double
    @Binding var iou: Double
    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Text("conf").font(.caption.monospaced()).frame(width: 34, alignment: .leading)
                Slider(value: $conf, in: 0.05...0.9)
                Text(String(format: "%.2f", conf)).font(.caption.monospaced()).frame(width: 34)
            }
            HStack {
                Text("IoU").font(.caption.monospaced()).frame(width: 34, alignment: .leading)
                Slider(value: $iou, in: 0.1...0.9)
                Text(String(format: "%.2f", iou)).font(.caption.monospaced()).frame(width: 34)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Shared mutable tuning state readable from the detection loop without
/// main-actor hops (word-sized reads; eventual consistency is fine here).
final class Tuning: ObservableObject {
    @Published var conf: Double = 0.25
    @Published var iou: Double = 0.5
}
