// On-device benchmark, rebuilt in the shared HUD design language: tachometer
// dials, color-ramp bars, the thermal tach, a live throttle sparkline, haptics,
// and the animated export toast - matching the Live and Photo tabs.
//
// Methodology (from the Orin/S26 work): flagships throttle, so a real-time claim
// needs BOTH a cold median and a sustained (thermal) number. Headline metric is
// inferOnly (pure model predict wall-time); each result also carries one
// forward+decode pass for a pre/inf/dec stage breakdown. The synthetic gray input
// yields ~no detections, so the decode stage reads the empty-scene floor -
// inference is the number that matters.
import CoreGraphics
import SwiftUI
import UIKit
import YOLOMasterKit

struct BenchResult: Identifiable {
    var id: String { "\(modelId)|\(compute.rawValue)" }
    let modelId: String
    let shortID: String
    let fullName: String
    let compute: ComputeChoice
    let coldMedian: Double
    let coldP90: Double
    let coldMin: Double
    let pre: Double
    let inf: Double
    let dec: Double
    var sustainedMedian: Double? = nil
    var throttlePct: Double? = nil
    var fpsEquiv: Double { coldMedian > 0 ? 1000.0 / coldMedian : 0 }
}

/// Run-control flags the detached bench loop polls each iteration (word-sized
/// reads; a data flag, eventual consistency is fine). Lets a run be PAUSED
/// (suspended, resumable) rather than only cancelled, and auto-paused when the
/// tab loses focus so benchmarking never competes with Live/Photo inference.
final class BenchControl: @unchecked Sendable {
    var paused = false
    var cancelled = false
}

struct BenchView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case sweep = "Cold Sweep", sustained = "Sustained"
        var id: String { rawValue }
    }
    enum Phase: Equatable {
        case idle
        case loadingModel(String, String)            // model shortID, unit
        case benchmarking(String, String, Int, Int)  // model, unit, done, total
        case sustained(String, String, Int, Int)     // model, unit, elapsedS, totalS
        case done
    }

    @State private var models: [BundledModel] = []
    @State private var results: [BenchResult] = []
    @State private var mode: Mode = .sweep
    @State private var phase: Phase = .idle
    @State private var running = false
    @State private var paused = false
    @State private var control = BenchControl()
    @Environment(\.scenePhase) private var scenePhase
    @State private var showSettings = false
    @State private var expandedCards: Set<String> = []
    // sustained target + settings
    @State private var selectedModel: BundledModel?
    @State private var selectedCompute: ComputeChoice = ComputeChoice.deviceDefault
    @State private var sustainedMinutes = 3.0
    @State private var iters = 50
    @State private var warmup = 10
    // live readouts
    @State private var liveMs = 0.0
    @State private var sparkSamples: [Double] = []
    @State private var sparkBaseline: Double? = nil
    @State private var thermal = ProcessInfo.processInfo.thermalState
    // export toast
    @State private var exportMsg: String? = nil
    @State private var exportOK = false
    @State private var exportGen = 0

    @State private var loopTask: Task<Void, Never>?
    private let lightHaptic = UIImpactFeedbackGenerator(style: .light)
    private let medHaptic = UIImpactFeedbackGenerator(style: .medium)
    private let notifyHaptic = UINotificationFeedbackGenerator()

    var body: some View {
        ZStack {
            if results.isEmpty && !running {
                // vertically centered, matching the Photo tab's "No photos" empty state
                ContentUnavailableView("No benchmarks yet",
                    systemImage: "speedometer",
                    description: Text("Run a cold sweep across every model and compute unit."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 10) {
                        if running { progressCard }
                        if mode == .sweep, let hero = fastest { heroCard(hero) }
                        ForEach(modelsWithResults, id: \.self) { modelCard($0) }
                    }
                    .padding(.horizontal, 10)
                    .padding(.bottom, 24)
                }
            }
            if let msg = exportMsg {
                ExportToast(success: exportOK, message: msg)
                    .transition(.scale(scale: 0.82).combined(with: .opacity))
                    .onTapGesture { withAnimation(.easeOut(duration: 0.2)) { exportMsg = nil } }
            }
        }
        .safeAreaInset(edge: .top) { topBar }
        .onAppear {
            lightHaptic.prepare(); medHaptic.prepare(); notifyHaptic.prepare()
            guard models.isEmpty else { return }
            Task.detached {
                let found = BundledModel.discover()
                await MainActor.run {
                    models = found
                    if selectedModel == nil { selectedModel = BundledModel.preferred(in: found) }
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            thermal = ProcessInfo.processInfo.thermalState
        }
        // leaving the tab (or backgrounding) auto-pauses the run so it never
        // competes with Live/Photo inference; the user resumes with play
        .onDisappear { autoPause() }
        .onChange(of: scenePhase) { _, ph in if ph != .active { autoPause() } }
    }

    private func autoPause() {
        if running && !paused { paused = true; control.paused = true }
    }

    // MARK: - top bar

    private var topBar: some View {
        VStack(spacing: 8) {
            HStack {
                Picker("Mode", selection: $mode) {
                    ForEach(Mode.allCases) { m in Text(m.rawValue).tag(m) }
                }
                .pickerStyle(.segmented)
                .disabled(running)
                Button { withAnimation { showSettings.toggle() } } label: {
                    Image(systemName: "slider.horizontal.3")
                }
                .buttonStyle(.bordered)
                if !results.isEmpty {
                    Button { exportCSV() } label: { Image(systemName: "square.and.arrow.up") }
                        .buttonStyle(.bordered)
                        .disabled(running)
                }
                if running {
                    Button { togglePause() } label: {
                        Image(systemName: paused ? "play.fill" : "pause.fill").frame(minWidth: 40)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(paused ? .accentColor : .orange)
                    Button { stop() } label: { Image(systemName: "xmark") }
                        .buttonStyle(.bordered)
                        .tint(.red)
                } else {
                    Button { start() } label: {
                        Image(systemName: "play.fill").frame(minWidth: 40)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(models.isEmpty || (mode == .sustained && selectedModel == nil))
                }
            }
            // sustained target + duration are ALWAYS visible in sustained mode
            // (not hidden behind the gear); the gear holds advanced iters/warmup
            if mode == .sustained { sustainedConfig }
            if showSettings { advancedSettings }
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 10)
    }

    private var sustainedConfig: some View {
        VStack(spacing: 8) {
            HStack {
                Menu {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(models) { m in Text(m.fullName).tag(Optional(m)) }
                    }
                } label: {
                    HStack(spacing: 3) {
                        Text(selectedModel?.shortID ?? "Model").lineLimit(1)
                        Image(systemName: "chevron.up.chevron.down").font(.caption2)
                    }
                }
                .fixedSize()
                Picker("Unit", selection: $selectedCompute) {
                    ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).tag(c) }
                }
                .pickerStyle(.segmented)
            }
            // manual duration selection: a fine stepper plus quick presets
            HStack(spacing: 8) {
                Stepper(value: $sustainedMinutes, in: 1...Double(maxSustained), step: 1) {
                    Text("Duration: \(Int(sustainedMinutes)) min")
                        .font(.caption.monospacedDigit())
                }
                ForEach([3, 5, 10, 20].filter { $0 <= maxSustained }, id: \.self) { p in
                    Button("\(p)m") { sustainedMinutes = Double(p) }
                        .font(.caption2)
                        .buttonStyle(.bordered)
                        .tint(Int(sustainedMinutes) == p ? .accentColor : .secondary)
                }
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .disabled(running)
        .onChange(of: selectedCompute) { _, _ in
            if sustainedMinutes > Double(maxSustained) { sustainedMinutes = Double(maxSustained) }
        }
    }

    // CPU inference is slow and hot - cap sustained CPU stress at 3 minutes.
    private var maxSustained: Int { selectedCompute == .cpu ? 3 : 60 }

    private var advancedSettings: some View {
        VStack(spacing: 6) {
            Stepper(value: $iters, in: 20...200, step: 10) {
                Text("Timed iters: \(iters)").font(.caption)
            }
            Stepper(value: $warmup, in: 0...50, step: 5) {
                Text("Warmup: \(warmup)").font(.caption)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .disabled(running)
    }

    // MARK: - cards

    private var progressCard: some View {
        VStack(spacing: 8) {
            if paused {
                HStack(spacing: 4) {
                    Image(systemName: "pause.circle.fill")
                    Text("Paused").font(.caption.bold())
                }
                .foregroundStyle(.orange)
                .frame(maxWidth: .infinity)
            }
            HStack(alignment: .center, spacing: 12) {
                // leading flexible column: the ProgressView fills THIS width only,
                // so it can never run under the thermal dial on the right
                phaseContent
                    .frame(maxWidth: .infinity, alignment: .leading)
                ThermalTach(level: thermalLevel, color: thermalColor)
            }
            if mode == .sustained, sparkSamples.count > 1 {
                SparklineView(samples: sparkSamples, baseline: sparkBaseline, color: .blue)
                    .frame(height: 58)
                HStack(spacing: 6) {
                    if let b = sparkBaseline {
                        Text("cold \(fmt(b))").font(.caption2).foregroundStyle(.secondary)
                    }
                    Text("→ live \(fmt(liveMs)) ms").font(.caption2.monospacedDigit())
                    Spacer()
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder private var phaseContent: some View {
        switch phase {
        case .loadingModel(let m, let u):
            HStack(spacing: 8) {
                ProgressView()
                Text("Loading \(m) to \(u)").font(.caption)
            }
        case .benchmarking(let m, let u, let d, let t):
            VStack(alignment: .leading, spacing: 4) {
                Text("Benchmarking \(m) @ \(u)").font(.caption)
                ProgressView(value: Double(d), total: Double(max(t, 1)))
                Text("\(d)/\(t)").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        case .sustained(let m, let u, let e, let tot):
            VStack(alignment: .leading, spacing: 4) {
                Text("Sustained \(m) @ \(u)").font(.caption)
                ProgressView(value: Double(e), total: Double(max(tot, 1)))
                Text("\(e / 60)m \(e % 60)s / \(tot / 60)m")
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        default:
            EmptyView()
        }
    }

    private func heroCard(_ r: BenchResult) -> some View {
        HStack(spacing: 16) {
            Gauge(value: min(max(r.coldMedian, 30), 100), in: 30...100) {
                Text("ms")
            } currentValueLabel: {
                Text("\(Int(r.coldMedian.rounded()))")
                    .font(.system(.title3, design: .rounded).bold())
            }
            .gaugeStyle(.accessoryCircular)
            .tint(msColor(r.coldMedian))
            .frame(width: 60, height: 60)
            VStack(alignment: .leading, spacing: 2) {
                Text("FASTEST").font(.caption2).foregroundStyle(.secondary)
                Text(r.fullName).font(.headline).lineLimit(1).minimumScaleFactor(0.6)
                Text("\(r.compute.rawValue)  ·  \(fmt(r.coldMedian)) ms  ·  \(Int(r.fpsEquiv)) FPS")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            ThermalTach(level: thermalLevel, color: thermalColor)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func modelCard(_ mid: String) -> some View {
        let rows = results.filter { $0.modelId == mid }.sorted { $0.coldMedian < $1.coldMedian }
        let title = rows.first?.fullName ?? mid
        let fastestUnit = rows.first?.compute
        let expanded = expandedCards.contains(mid)
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title).font(.subheadline.bold()).lineLimit(1).minimumScaleFactor(0.6)
                Spacer()
                if let f = fastestUnit {
                    Text("fastest \(f.rawValue)").font(.caption2)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(.green.opacity(0.2), in: Capsule())
                }
                Image(systemName: expanded ? "chevron.up" : "chevron.down")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            ForEach(rows) { r in
                DetailBar(icon: unitIcon(r.compute), name: r.compute.rawValue,
                          ms: r.coldMedian, fullScale: 50,
                          color: msColor(r.coldMedian), value: "\(fmt(r.coldMedian))")
                if let s = r.sustainedMedian, let tp = r.throttlePct {
                    DetailBar(icon: "flame", name: "sustained", ms: s, fullScale: 50,
                              color: msColor(s), value: "+\(Int(tp))%")
                }
            }
            if expanded {
                Divider()
                ForEach(rows) { r in
                    Text("\(r.compute.rawValue) · \(Int(r.fpsEquiv)) FPS · p90 \(fmt(r.coldP90))")
                        .font(.caption2).foregroundStyle(.secondary)
                    DetailBar(icon: "aspectratio", name: "preprocess", ms: r.pre)
                    DetailBar(icon: "cpu", name: "inference", ms: r.inf)
                    DetailBar(icon: "rectangle.dashed", name: "decode", ms: r.dec)
                    DetailBar(icon: "timer", name: "end to end", ms: r.pre + r.inf + r.dec,
                              fullScale: 100, greenUntil: 50)
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.spring(duration: 0.3)) {
                if expanded { expandedCards.remove(mid) } else { expandedCards.insert(mid) }
            }
        }
    }

    // MARK: - derived

    private var fastest: BenchResult? { results.min { $0.coldMedian < $1.coldMedian } }
    private var modelsWithResults: [String] {
        models.map(\.id).filter { id in results.contains { $0.modelId == id } }
    }
    private func fmt(_ v: Double) -> String { String(format: "%.1f", v) }
    private func unitIcon(_ c: ComputeChoice) -> String {
        switch c {
        case .ane: return "bolt.fill"
        case .gpu: return "square.stack.3d.up.fill"
        case .cpu: return "cpu"
        }
    }
    private var thermalLevel: Int {
        switch thermal {
        case .nominal: return 0
        case .fair: return 1
        case .serious: return 2
        case .critical: return 3
        @unknown default: return 0
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

    // MARK: - run control

    private func start() {
        guard !running else { return }
        control.paused = false; control.cancelled = false
        running = true; paused = false
        medHaptic.impactOccurred(); medHaptic.prepare()
        sparkSamples = []; sparkBaseline = nil; liveMs = 0
        if mode == .sweep {
            results = []; expandedCards = []
            runSweep()
        } else {
            runSustained()
        }
    }

    private func stop() {
        control.cancelled = true
        loopTask?.cancel(); loopTask = nil
        running = false; paused = false
        phase = .idle
    }

    private func togglePause() {
        paused.toggle()
        control.paused = paused
        lightHaptic.impactOccurred(); lightHaptic.prepare()
    }

    /// Suspend the detached loop while paused; returns false if cancelled.
    private func pauseGate() async -> Bool {
        while control.paused && !control.cancelled {
            try? await Task.sleep(nanoseconds: 120_000_000)
        }
        return !control.cancelled
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

    /// One forward+decode pass -> pre/inf/dec stage ms (empty-scene decode floor).
    private func stageBreakdown(_ det: Detector, _ img: CGImage) -> (Double, Double, Double) {
        var pre = 0.0, inf = 0.0, dec = 0.0
        autoreleasepool {
            let a = CFAbsoluteTimeGetCurrent()
            guard let raw = try? det.forward(img) else { return }
            let b = CFAbsoluteTimeGetCurrent()
            _ = det.decode(raw, conf: 0.25, iou: 0.5)
            let c = CFAbsoluteTimeGetCurrent()
            inf = raw.inferMs
            pre = max((b - a) * 1000 - raw.inferMs, 0)
            dec = (c - b) * 1000
        }
        return (pre, inf, dec)
    }

    private func runSweep() {
        let warmN = warmup, iterN = iters
        loopTask = Task.detached(priority: .userInitiated) {
            let img = testImage()
            let units = ComputeChoice.allCases
            let total = models.count * units.count
            var done = 0
            for m in models {
                for c in units {
                    if control.cancelled { return }
                    if !(await pauseGate()) { return }
                    await MainActor.run { phase = .loadingModel(m.fullName, c.rawValue) }
                    guard let url = try? m.compiledURL(),
                          let det = try? Detector(modelURL: url, compute: c.mode) else {
                        done += 1; continue
                    }
                    await MainActor.run { phase = .benchmarking(m.fullName, c.rawValue, done, total) }
                    for _ in 0..<warmN { autoreleasepool { _ = try? det.inferOnly(img) } }
                    var ms: [Double] = []
                    for _ in 0..<iterN {
                        if control.cancelled { return }
                        autoreleasepool { if let t = try? det.inferOnly(img) { ms.append(t) } }
                    }
                    guard !ms.isEmpty else { done += 1; continue }
                    ms.sort()
                    let (pre, inf, dec) = stageBreakdown(det, img)
                    let r = BenchResult(modelId: m.id, shortID: m.shortID, fullName: m.fullName, compute: c,
                        coldMedian: ms[ms.count / 2],
                        coldP90: ms[min(Int(Double(ms.count) * 0.9), ms.count - 1)],
                        coldMin: ms.first ?? 0, pre: pre, inf: inf, dec: dec)
                    done += 1
                    let d = done
                    await MainActor.run {
                        withAnimation(.spring(duration: 0.3)) { results.append(r) }
                        phase = .benchmarking(m.fullName, c.rawValue, d, total)
                        lightHaptic.impactOccurred(); lightHaptic.prepare()
                    }
                }
            }
            await MainActor.run {
                running = false; phase = .done
                notifyHaptic.notificationOccurred(.success); notifyHaptic.prepare()
            }
        }
    }

    private func runSustained() {
        guard let m = selectedModel else { running = false; return }
        let c = selectedCompute, warmN = warmup, iterN = iters
        let totalS = Int(sustainedMinutes * 60)
        loopTask = Task.detached(priority: .userInitiated) {
            let img = testImage()
            await MainActor.run { phase = .loadingModel(m.fullName, c.rawValue) }
            guard let url = try? m.compiledURL(),
                          let det = try? Detector(modelURL: url, compute: c.mode) else {
                await MainActor.run { running = false; phase = .idle }; return
            }
            // cold baseline
            for _ in 0..<warmN { autoreleasepool { _ = try? det.inferOnly(img) } }
            var cold: [Double] = []
            for _ in 0..<iterN { autoreleasepool { if let t = try? det.inferOnly(img) { cold.append(t) } } }
            cold.sort()
            let coldMed = cold.isEmpty ? 0 : cold[cold.count / 2]
            await MainActor.run { sparkBaseline = coldMed; sparkSamples = [] }
            // sustained loop
            var t0 = CFAbsoluteTimeGetCurrent()
            var all: [Double] = []
            var lastPub = 0.0
            while CFAbsoluteTimeGetCurrent() - t0 < Double(totalS) {
                if control.cancelled { return }
                if control.paused {
                    let ps = CFAbsoluteTimeGetCurrent()
                    if !(await pauseGate()) { return }
                    t0 += CFAbsoluteTimeGetCurrent() - ps   // exclude paused time from elapsed
                }
                autoreleasepool { if let t = try? det.inferOnly(img) { all.append(t) } }
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastPub >= 0.1 {
                    lastPub = now
                    let elapsed = Int(now - t0)
                    // smooth trend: bucket the whole run into ~100 time-windows and
                    // average each, so per-inference jitter cancels out
                    let buckets = 100
                    let bsize = max(1, all.count / buckets)
                    var pts: [Double] = []
                    var bi = 0
                    while bi < all.count {
                        let end = min(bi + bsize, all.count)
                        pts.append(all[bi..<end].reduce(0, +) / Double(end - bi))
                        bi = end
                    }
                    let live = all.suffix(30).sorted()
                    let liveMed = live.isEmpty ? 0 : live[live.count / 2]
                    await MainActor.run {
                        sparkSamples = pts; liveMs = liveMed
                        phase = .sustained(m.fullName, c.rawValue, elapsed, totalS)
                    }
                }
            }
            all.sort()
            let tail = Array(all.suffix(max(all.count / 4, 1))).sorted()
            let sust = tail.isEmpty ? coldMed : tail[tail.count / 2]
            let p90 = all.isEmpty ? 0 : all[min(Int(Double(all.count) * 0.9), all.count - 1)]
            let (pre, inf, dec) = stageBreakdown(det, img)
            let r = BenchResult(modelId: m.id, shortID: m.shortID, fullName: m.fullName, compute: c,
                coldMedian: coldMed, coldP90: p90, coldMin: all.first ?? 0,
                pre: pre, inf: inf, dec: dec,
                sustainedMedian: sust, throttlePct: coldMed > 0 ? (sust - coldMed) / coldMed * 100 : 0)
            await MainActor.run {
                results.removeAll { $0.modelId == m.id && $0.compute == c }
                withAnimation(.spring(duration: 0.3)) { results.append(r) }
                running = false; phase = .done
                notifyHaptic.notificationOccurred(.success); notifyHaptic.prepare()
            }
        }
    }

    // MARK: - export

    private func buildCSV() -> String {
        let header = "model,compute,cold_median_ms,cold_p90_ms,cold_min_ms,pre_ms,inf_ms,dec_ms,fps_equiv,sustained_ms,throttle_pct"
        let rows = results.map { r in
            String(format: "%@,%@,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%@,%@",
                   r.modelId, r.compute.rawValue, r.coldMedian, r.coldP90, r.coldMin,
                   r.pre, r.inf, r.dec, r.fpsEquiv,
                   r.sustainedMedian.map { String(format: "%.2f", $0) } ?? "",
                   r.throttlePct.map { String(format: "%.1f", $0) } ?? "")
        }
        return ([header] + rows).joined(separator: "\n")
    }

    private func exportCSV() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("benchmark.csv")
        let ok = (try? buildCSV().data(using: .utf8)?.write(to: url)) != nil
        lightHaptic.impactOccurred(); lightHaptic.prepare()
        notifyHaptic.notificationOccurred(ok ? .success : .error); notifyHaptic.prepare()
        exportGen += 1
        let gen = exportGen
        withAnimation(.spring(duration: 0.3)) {
            exportOK = ok
            exportMsg = ok ? "benchmark.csv ready" : "Export failed"
        }
        if ok { share(url) }
        Task {
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            if exportGen == gen { withAnimation(.easeOut(duration: 0.25)) { exportMsg = nil } }
        }
    }

    private func share(_ url: URL) {
        guard let scene = UIApplication.shared.connectedScenes
                .first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene,
              let root = scene.keyWindow?.rootViewController else { return }
        let av = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        av.popoverPresentationController?.sourceView = root.view
        av.popoverPresentationController?.sourceRect = CGRect(
            x: root.view.bounds.midX, y: root.view.bounds.midY, width: 0, height: 0)
        root.present(av, animated: true)
    }
}
