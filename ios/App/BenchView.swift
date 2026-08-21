// On-device benchmark: cold median + sustained loop, per model x compute unit.
// The phone-methodology lesson from the Orin/S26 work: flagships throttle, so a
// real-time claim needs BOTH the cold median and the sustained (thermal) number.
import CoreGraphics
import SwiftUI
import YOLOMasterKit

struct BenchRow: Identifiable {
    let id = UUID()
    let model: String, compute: String
    let coldMedian: Double, coldP90: Double
    let sustainedMedian: Double?          // nil until the sustained pass runs
}

struct BenchView: View {
    @State private var models = BundledModel.discover()
    @State private var rows: [BenchRow] = []
    @State private var status = "idle"
    @State private var busy = false
    @State private var sustainedMinutes = 3.0

    var body: some View {
        NavigationStack {
            List {
                Section("Results") {
                    ForEach(rows) { r in
                        VStack(alignment: .leading) {
                            Text("\(r.model) @\(r.compute)").font(.headline)
                            Text(String(format: "cold %.2f ms (p90 %.2f)%@",
                                        r.coldMedian, r.coldP90,
                                        r.sustainedMedian.map {
                                            String(format: "   sustained %.2f ms", $0)
                                        } ?? ""))
                                .font(.caption.monospacedDigit())
                        }
                    }
                    if rows.isEmpty { Text("No results yet").foregroundStyle(.secondary) }
                }
                Section {
                    Stepper(value: $sustainedMinutes, in: 1...10, step: 1) {
                        Text("Sustained pass: \(Int(sustainedMinutes)) min")
                    }
                    Button(busy ? status : "Run cold sweep (all models x units)") { runCold() }
                        .disabled(busy || models.isEmpty)
                    Button("Run sustained (first model @ANE)") { runSustained() }
                        .disabled(busy || models.isEmpty)
                    ShareLink(item: exportText()) { Text("Share results") }
                        .disabled(rows.isEmpty)
                }
            }
            .navigationTitle("Benchmark")
        }
    }

    /// 640x640 mid-gray test image; inferOnly measures predict() wall time.
    private func testImage() -> CGImage {
        let ctx = CGContext(data: nil, width: 640, height: 640, bitsPerComponent: 8,
                            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.45, green: 0.45, blue: 0.45, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 640, height: 640))
        return ctx.makeImage()!
    }

    private func runCold() {
        busy = true; rows = []
        let img = testImage()
        Task.detached(priority: .userInitiated) {
            var out: [BenchRow] = []
            for m in models {
                for c in ComputeChoice.allCases {
                    await MainActor.run { status = "\(m.id) @\(c.rawValue)..." }
                    guard let det = try? Detector(modelURL: m.url, compute: c.mode) else { continue }
                    var ms: [Double] = []
                    for _ in 0..<10 { autoreleasepool { _ = try? det.inferOnly(img) } }
                    for _ in 0..<50 {
                        autoreleasepool {
                            if let t = try? det.inferOnly(img) { ms.append(t) }
                        }
                    }
                    guard !ms.isEmpty else { continue }
                    ms.sort()
                    out.append(BenchRow(model: m.id, compute: c.rawValue,
                                        coldMedian: ms[ms.count / 2],
                                        coldP90: ms[Int(Double(ms.count) * 0.9)],
                                        sustainedMedian: nil))
                }
            }
            let done = out
            await MainActor.run { rows = done; busy = false; status = "idle" }
        }
    }

    private func runSustained() {
        guard let m = models.first else { return }
        busy = true
        let img = testImage()
        let seconds = sustainedMinutes * 60
        Task.detached(priority: .userInitiated) {
            guard let det = try? Detector(modelURL: m.url, compute: ComputeChoice.ane.mode) else {
                await MainActor.run { busy = false }; return
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            var ms: [Double] = []
            while CFAbsoluteTimeGetCurrent() - t0 < seconds {
                autoreleasepool {
                    if let t = try? det.inferOnly(img) { ms.append(t) }
                }
                let pct = Int(100 * (CFAbsoluteTimeGetCurrent() - t0) / seconds)
                if ms.count % 50 == 0 {
                    await MainActor.run { status = "sustained \(pct)%..." }
                }
            }
            ms.sort()
            // report the LAST-quarter median: that's the thermal steady state
            let tail = Array(ms.suffix(ms.count / 4)).sorted()
            let sustained = tail.isEmpty ? ms[ms.count / 2] : tail[tail.count / 2]
            await MainActor.run {
                rows.append(BenchRow(model: m.id, compute: "ANE",
                                     coldMedian: ms[ms.count / 2],
                                     coldP90: ms[Int(Double(ms.count) * 0.9)],
                                     sustainedMedian: sustained))
                busy = false; status = "idle"
            }
        }
    }

    private func exportText() -> String {
        (["model,compute,cold_median_ms,cold_p90_ms,sustained_ms"] +
         rows.map { r in
             String(format: "%@,%@,%.2f,%.2f,%@", r.model, r.compute,
                    r.coldMedian, r.coldP90,
                    r.sustainedMedian.map { String(format: "%.2f", $0) } ?? "")
         }).joined(separator: "\n")
    }
}
