// Shared stats HUD: FPS tachometer (smooth color ramp), per-stage latency bars
// (green -> orange as a stage misbehaves), detection count, thermal state.
import SwiftUI

// MARK: - color ramps

private typealias RGB = (r: Double, g: Double, b: Double)
private let cRed: RGB = (0.96, 0.26, 0.21)
private let cOrange: RGB = (1.00, 0.58, 0.00)
private let cGreen: RGB = (0.20, 0.84, 0.29)
private let cPurple: RGB = (0.69, 0.32, 0.87)

private func lerp(_ a: RGB, _ b: RGB, _ t: Double) -> RGB {
    let k = min(max(t, 0), 1)
    return (a.r + (b.r - a.r) * k, a.g + (b.g - a.g) * k, a.b + (b.b - a.b) * k)
}

private func color(_ c: RGB) -> Color { Color(red: c.r, green: c.g, blue: c.b) }

/// Pure band colors - 0-10 red, 10-20 orange, 20-<30 green, and full purple
/// once the dial pegs at 30 (model outrunning the camera). Band changes
/// animate in TIME (fast gradient); values never mix bands.
func fpsColor(_ fps: Double) -> Color {
    switch fps {
    case ..<10: return color(cRed)
    case ..<20: return color(cOrange)
    case ..<29.5: return color(cGreen)
    default: return color(cPurple)
    }
}

/// Stage bars: green while legit, blending to orange past ~20ms, fully orange
/// at 35ms+ ("extraordinarily slow" for any single stage of a 30FPS budget).
func stageColor(_ ms: Double) -> Color {
    color(lerp(cGreen, cOrange, (ms - 20) / 15))
}

// MARK: - HUD

struct StatsHUD: View {
    let fps: Double            // model end-to-end FPS; dial max = 30 (camera rate)
    let pre: Double            // ms
    let inf: Double            // ms
    let dec: Double            // ms
    let dets: Int
    var fpsLabel: String = "FPS"
    var active: Bool = true    // false while no model is loaded/running: N/A + grey

    @State private var thermal = ProcessInfo.processInfo.thermalState

    var body: some View {
        HStack(spacing: 14) {
            Gauge(value: active ? min(max(fps, 0), 30) : 0, in: 0...30) {
                Text(fpsLabel)
            } currentValueLabel: {
                Text(active ? "\(Int(fps.rounded()))" : "N/A")
                    .font(.system(active ? .title3 : .footnote, design: .rounded).bold())
                    .foregroundStyle(active ? .primary : .secondary)
            }
            .gaugeStyle(.accessoryCircular)
            .tint(active ? fpsColor(fps) : Color.gray.opacity(0.5))
            .animation(.easeInOut(duration: 0.25), value: active ? fpsColor(fps) : .gray)
            .frame(width: 56, height: 56)

            VStack(alignment: .leading, spacing: 5) {
                StageBar(label: "pre", ms: pre)
                StageBar(label: "inf", ms: inf)
                StageBar(label: "dec", ms: dec)
            }

            VStack(spacing: 2) {
                VStack(spacing: 0) {
                    Text("\(dets)")
                        .font(.system(.title3, design: .rounded).bold())
                        .foregroundStyle(.green)
                    Text("dets").font(.caption2).foregroundStyle(.secondary)
                }
                VStack(spacing: 0) {
                    Image(systemName: "thermometer.medium")
                        .foregroundStyle(thermalColor)
                        .animation(.easeOut(duration: 0.3), value: thermal)
                    Text(thermalLabel).font(.caption2).foregroundStyle(.secondary)
                }
            }
            .frame(width: 48)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .onReceive(NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            thermal = ProcessInfo.processInfo.thermalState
        }
    }

    private var thermalColor: Color {
        switch thermal {
        case .nominal: return .blue
        case .fair: return .green
        case .serious: return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }

    private var thermalLabel: String {
        switch thermal {
        case .nominal: return "cool"
        case .fair: return "fair"
        case .serious: return "hot"
        case .critical: return "crit"
        @unknown default: return "?"
        }
    }
}

struct StageBar: View {
    let label: String
    let ms: Double
    private let fullScale = 50.0   // bar saturates at 50ms

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2.monospaced())
                .frame(width: 24, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(stageColor(ms))
                        .frame(width: max(3, min(ms / fullScale, 1) * g.size.width))
                }
            }
            .frame(width: 90, height: 6)
            .animation(.easeOut(duration: 0.15), value: ms)
            Text(String(format: "%4.1f", ms))
                .font(.caption2.monospaced())
                .frame(width: 36, alignment: .trailing)
        }
    }
}

/// conf / IoU tuning sliders + box style - the detection knobs, one panel.
struct TuningPanel: View {
    @Binding var conf: Double
    @Binding var iou: Double
    var style: Binding<BoxStyle>? = nil
    var body: some View {
        VStack(spacing: 6) {
            if let style {
                Picker("Style", selection: style) {
                    ForEach(BoxStyle.allCases) { s in
                        Text(s.label).lineLimit(1).fixedSize().tag(s)
                    }
                }
                .pickerStyle(.segmented)
            }
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
