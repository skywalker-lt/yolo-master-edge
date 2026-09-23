// Bench mode: the app's second major mode. The sidebar holds the protocol (models, compute
// units, preprocessing device, warm-up, timed iterations, sustained minutes, dataset / accuracy
// set) and the run history; the stage becomes a dashboard that streams the run as it happens:
// a per-iteration latency chart, the per-second sustained trace with its thermal band, a
// thermometer, the stat cards, the sweep table and the accuracy per-class chart.
//
// The measurements come from the Kit's BenchRunner / AccuracyRunner (the same probe, statistics
// and protocol as the CLI's --bench / --accuracy, through the portable core), so every run also
// yields a yolomaster-bench/v1 document that can be saved or exported.
import SwiftUI
import Charts
import AppKit
import IOKit
import UniformTypeIdentifiers
import YOLOMasterKit

enum AppMode: String, CaseIterable { case inference = "Inference", bench = "Bench" }

// MARK: - data

enum BenchKind: String, CaseIterable, Codable {
    case cold = "Cold sweep", sustained = "Sustained", dataset = "Dataset", accuracy = "Accuracy"
    var icon: String {
        switch self { case .cold: return "bolt.fill"; case .sustained: return "flame.fill"
        case .dataset: return "photo.stack"; case .accuracy: return "checkmark.seal" }
    }
    var blurb: String {
        switch self {
        case .cold: return "Warm-up, then timed model-only predictions on a gray probe at the input size. The headline latency."
        case .sustained: return "A timed loop; the slowest-quarter median against the cold median is the throttle figure. Thermal state is sampled every second."
        case .dataset: return "Full pipeline over a folder of images: preprocess, model and postprocess per image at the current confidence."
        case .accuracy: return "Val protocol (conf 0.001, IoU 0.7, max_det 300) over a labelled folder, scored in process: mAP50 / mAP50-95 per class."
        }
    }
}

enum ComputeChoice: String, CaseIterable, Codable, Identifiable {
    case ane = "ANE", gpu = "GPU", cpu = "CPU"
    var id: String { rawValue }
    var mode: ComputeMode { switch self { case .ane: return .all; case .gpu: return .cpuAndGPU; case .cpu: return .cpu } }
    var ep: String { "CoreML-" + rawValue }
}

/// One (model x compute unit) cell of a run.
struct BenchCell: Identifiable, Codable {
    var id = UUID()
    let modelName: String
    let modelPath: String
    let compute: ComputeChoice
    let preproc: String
    var cold: StageStats?
    var sustained: BenchDocument.Sustained?
    var dataset: BenchDocument.Dataset?
    var accuracy: BenchDocument.Accuracy?
    var samples: [Double] = []           // the per-iteration series (model ms; dataset: total ms)
    var sampleTimes: [Double] = []       // seconds since the cell's run started, parallel to samples
    var thermal: [Int] = []              // thermal level per second (sustained) or per sample bucket
    var document: BenchDocument?
    var headlineMs: Double? { cold?.median ?? sustained?.sustained_median_ms ?? dataset?.infer_ms.median ?? accuracy?.timings["infer_ms"]?.median }
    var fps: Double { (headlineMs ?? 0) > 0 ? 1000 / headlineMs! : 0 }
}

/// A saved run (JSON-persisted, newest first).
struct BenchRecord: Identifiable, Codable {
    var id = UUID()
    var name: String
    let date: Date
    let kind: BenchKind
    let warmup: Int, iters: Int, minutes: Double
    let cells: [BenchCell]
    let hostName: String, cpuModel: String, osVersion: String
    var fastest: BenchCell? { cells.min { ($0.headlineMs ?? .infinity) < ($1.headlineMs ?? .infinity) } }
}

final class BenchStore: ObservableObject {
    @Published private(set) var records: [BenchRecord] = []
    private let url: URL
    init() {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("YOLOMaster", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        url = dir.appendingPathComponent("bench_history.json")
        if let d = try? Data(contentsOf: url), let r = try? JSONDecoder().decode([BenchRecord].self, from: d) { records = r }
    }
    func add(_ r: BenchRecord) { records.insert(r, at: 0); save() }
    func remove(_ ids: Set<UUID>) { records.removeAll { ids.contains($0.id) }; save() }
    func rename(_ id: UUID, _ name: String) { if let i = records.firstIndex(where: { $0.id == id }) { records[i].name = name; save() } }
    func clear() { records = []; save() }
    private func save() {
        let enc = JSONEncoder(); enc.dateEncodingStrategy = .iso8601
        if let d = try? enc.encode(records) { try? d.write(to: url) }
    }
    /// One row per cell of every record: what a spreadsheet wants.
    func csv() -> String {
        var out = ["run,date,kind,model,compute,preproc,warmup,iters,minutes,cold_median_ms,cold_p90_ms,cold_p99_ms,cold_min_ms,sustained_median_ms,throttle_pct,dataset_pre_ms,dataset_infer_ms,dataset_post_ms,map50,map5095,images"]
        let f = ISO8601DateFormatter()
        for r in records {
            for c in r.cells {
                func s(_ v: Double?) -> String { v.map { String(format: "%.4f", $0) } ?? "" }
                out.append([r.name.replacingOccurrences(of: ",", with: " "), f.string(from: r.date), r.kind.rawValue, c.modelName, c.compute.rawValue, c.preproc,
                            "\(r.warmup)", "\(r.iters)", "\(r.minutes)",
                            s(c.cold?.median), s(c.cold?.p90), s(c.cold?.p99), s(c.cold?.min),
                            s(c.sustained?.sustained_median_ms), s(c.sustained?.throttle_pct),
                            s(c.dataset?.pre_ms.median), s(c.dataset?.infer_ms.median), s(c.dataset?.post_ms.median),
                            s(c.accuracy?.map50), s(c.accuracy?.map5095), c.accuracy.map { "\($0.images)" } ?? ""].joined(separator: ","))
            }
        }
        return out.joined(separator: "\n") + "\n"
    }
}

// MARK: - thermal

func thermalLevel(_ s: ProcessInfo.ThermalState) -> Int {
    switch s { case .nominal: return 0; case .fair: return 1; case .serious: return 2; case .critical: return 3; @unknown default: return 0 }
}
func thermalName(_ level: Int) -> String { ["Cool", "Normal", "Hot", "Critical"][max(0, min(3, level))] }

// MARK: - battery power flow (IOKit AppleSmartBattery: instantaneous current x voltage)

struct BatterySample {
    let present: Bool
    let watts: Double          // > 0 charging, < 0 discharging, 0 when idle / no battery
    let charging: Bool
    let external: Bool         // on mains
    let percent: Int?
    static let none = BatterySample(present: false, watts: 0, charging: false, external: true, percent: nil)
}
enum BatteryReader {
    static func read() -> BatterySample {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSmartBattery"))
        guard service != 0 else { return .none }
        defer { IOObjectRelease(service) }
        var propsRef: Unmanaged<CFMutableDictionary>?
        guard IORegistryEntryCreateCFProperties(service, &propsRef, kCFAllocatorDefault, 0) == KERN_SUCCESS,
              let props = propsRef?.takeRetainedValue() as? [String: Any] else { return .none }
        let mA = (props["InstantAmperage"] as? Int64) ?? (props["Amperage"] as? Int64) ?? Int64((props["InstantAmperage"] as? Int) ?? (props["Amperage"] as? Int) ?? 0)
        let signed = mA > Int64(Int32.max) ? mA - (Int64(1) << 32) : mA     // some firmware reports a 32-bit two's complement in a 64-bit field
        let mV = Double((props["Voltage"] as? Int) ?? 0)
        let watts = Double(signed) / 1000 * mV / 1000
        let charging = (props["IsCharging"] as? Bool) ?? false
        let external = (props["ExternalConnected"] as? Bool) ?? false
        var percent: Int? = nil
        if let cur = props["CurrentCapacity"] as? Int, let mx = props["MaxCapacity"] as? Int, mx > 0 { percent = Int((Double(cur) / Double(mx) * 100).rounded()) }
        if let pct = props["CurrentCapacity"] as? Int, pct <= 100, props["MaxCapacity"] as? Int == 100 { percent = pct }
        return BatterySample(present: true, watts: watts, charging: charging, external: external, percent: percent)
    }
}
func thermalColor(_ level: Int) -> Color { [Color.green, .yellow, .orange, .red][max(0, min(3, level))] }

// MARK: - the meters (their own observable so their 10 Hz ticks re-render only the two gauges)

final class MeterModel: ObservableObject {
    @Published private(set) var thermal = thermalLevel(ProcessInfo.processInfo.thermalState)
    @Published private(set) var thermalPeak = 0
    @Published private(set) var battery = BatteryReader.read()
    private var thermalTimer: Timer?
    private var powerTimer: Timer?
    var tracking = false          // a run is on: keep the peak

    init() {
        // thermal state is coarse (ProcessInfo changes rarely): once a second
        thermalTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self else { return }
            let l = thermalLevel(ProcessInfo.processInfo.thermalState)
            if l != self.thermal { withAnimation(.easeInOut(duration: 0.8)) { self.thermal = l } }
            if self.tracking && l > self.thermalPeak { self.thermalPeak = l }
        }
        // the battery's instantaneous current at 10 Hz (an IOKit registry read, well under a millisecond)
        powerTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self else { return }
            let b = BatteryReader.read()
            if abs(b.watts - self.battery.watts) > 0.01 || b.charging != self.battery.charging || b.present != self.battery.present {
                withAnimation(.easeInOut(duration: 0.25)) { self.battery = b }
            }
        }
    }
    func startRun() { tracking = true; thermalPeak = thermal }
    func endRun() { tracking = false }
}

final class BenchModel: ObservableObject {
    // protocol
    @Published var models: [URL] = []
    @Published var selectedModels: Set<URL> = []
    @Published var computes: Set<ComputeChoice> = [.gpu]
    @Published var preproc: PreprocDevice = .gpu
    @Published var kind: BenchKind = .cold
    @Published var warmup = 10.0
    @Published var iters = 100.0
    @Published var minutes = 2.0
    @Published var datasetURL: URL?             // dataset / accuracy: the images folder
    @Published var datasetLimit = 0.0           // 0 = all
    @Published var conf = 0.25                  // dataset pass only
    @Published var iou = 0.5
    // live
    @Published private(set) var running = false
    @Published private(set) var phase = ""          // what is happening now
    @Published private(set) var progress: Double?   // 0...1 when known
    @Published private(set) var liveSamples: [(t: Double, ms: Double)] = []   // t = seconds since the cell's run started
    @Published private(set) var liveStart = Date()
    @Published private(set) var liveSeconds: [(t: Double, med: Double, thermal: Int)] = []
    @Published private(set) var liveCell: BenchCell?
    let meters = MeterModel()
    private(set) var runPowerW: [Double] = []      // battery watts sampled once per second during a run
    @Published private(set) var cells: [BenchCell] = []     // the current / last run
    @Published private(set) var lastRecord: BenchRecord?
    @Published var note = ""
    let store = BenchStore()

    private let queue = DispatchQueue(label: "com.yolomaster.bench", qos: .userInitiated)
    private var cancelFlag = false
    private let cancelLock = NSLock()
    private var powerLogTimer: Timer?
    private var pendingSamples: [(t: Double, ms: Double)] = []
    private var cellStart = Date()
    private var cellGen = 0                 // bumped per cell: samples still in flight from the previous cell are dropped
    private var flushScheduled = false
    private var detectors: [String: Detector] = [:]

    init() {
        // once a second during a run: log the battery draw (not published; the record keeps it)
        powerLogTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self, self.running, self.meters.battery.present else { return }
            self.runPowerW.append(self.meters.battery.watts)
        }
    }

    func addModel(_ url: URL) {
        if !models.contains(url) { models.append(url) }
        selectedModels.insert(url)
    }
    func removeModel(_ url: URL) { models.removeAll { $0 == url }; selectedModels.remove(url) }

    private var isCancelled: Bool { cancelLock.lock(); defer { cancelLock.unlock() }; return cancelFlag }
    func cancel() { cancelLock.lock(); cancelFlag = true; cancelLock.unlock() }

    private func detector(_ url: URL, _ c: ComputeChoice) throws -> Detector {
        let key = url.path + "|" + c.rawValue
        if let d = detectors[key] { d.preprocDevice = preproc; return d }
        let d = try Detector(modelURL: url, compute: c.mode)
        d.preprocDevice = preproc
        detectors[key] = d
        return d
    }

    // streaming: samples are batched onto the main thread at ~20 Hz so the chart never starves the run
    private func push(_ ms: Double) {
        let t = Date().timeIntervalSince(cellStart), gen = cellGen
        DispatchQueue.main.async { [weak self] in
            guard let self, gen == self.cellGen else { return }
            self.pendingSamples.append((t, ms))
            if !self.flushScheduled {
                self.flushScheduled = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    if gen == self.cellGen { self.liveSamples.append(contentsOf: self.pendingSamples) }
                    self.pendingSamples.removeAll(); self.flushScheduled = false
                }
            }
        }
    }
    private func main(_ f: @escaping () -> Void) { DispatchQueue.main.async(execute: f) }

    func run() {
        guard !running else { return }
        let targets = models.filter { selectedModels.contains($0) }
        guard !targets.isEmpty else { note = "Add and select at least one model."; return }
        guard !computes.isEmpty else { note = "Select at least one compute unit."; return }
        if (kind == .dataset || kind == .accuracy) && datasetURL == nil { note = "Choose the images folder first."; return }
        cancelLock.lock(); cancelFlag = false; cancelLock.unlock()
        running = true; cells = []; liveSamples = []; liveSeconds = []; liveCell = nil; progress = nil; note = ""
        runPowerW = []; meters.startRun()
        let kind = self.kind, warm = Int(warmup), iters = Int(iters), minutes = self.minutes
        let computes = ComputeChoice.allCases.filter { self.computes.contains($0) }
        let dataset = datasetURL, limit = Int(datasetLimit), conf = Float(self.conf), iou = CGFloat(self.iou)
        queue.async { [weak self] in
            guard let self else { return }
            var done: [BenchCell] = []
            outer: for url in targets {
                for c in computes {
                    if self.isCancelled { break outer }
                    let name = url.deletingPathExtension().lastPathComponent
                    self.main { self.phase = "Loading \(name) on \(c.rawValue)…" }
                    let det: Detector
                    do { det = try self.detector(url, c) } catch {
                        self.main { self.note = "\(name) on \(c.rawValue): \(error.localizedDescription)" }
                        continue
                    }
                    var cell = BenchCell(modelName: name, modelPath: url.path, compute: c, preproc: det.effectivePreprocDevice.rawValue)
                    // new cell: new clock and generation; the chart and any in-flight samples of the previous cell are dropped
                    self.cellStart = Date()
                    let cellStart = self.cellStart
                    let gen = self.cellGen + 1
                    self.cellGen = gen
                    self.main { self.liveCell = cell; self.liveStart = cellStart; self.liveSamples = []; self.liveSeconds = []; self.pendingSamples = [] }
                    var card = BenchEnvironment.model(det); card.ep_note = "preproc=\(det.effectivePreprocDevice.rawValue)"
                    var doc = BenchDocument(timestamp: YMCore.timestampUTC(), tool: "macos", model: card, environment: BenchEnvironment.collect(),
                                            protocol: .init(mode: kind == .sustained ? "sustained" : "cold", conf: conf, iou: Float(iou), max_det: 300,
                                                            multi_label: true, slicing: "off", tile_size: 0, warmup: warm, iters: iters, minutes: minutes,
                                                            probe: "gray114", probe_mode: "infer_only", dataset: dataset?.lastPathComponent ?? "",
                                                            image_count: 0, image_list_sha256: YMCore.imageListSha256([])))
                    var thermalTrack: [Int] = []
                    let t0 = Date(); var lastSec = -1
                    let sampleThermal: () -> Void = {
                        let sec = Int(Date().timeIntervalSince(t0))
                        if sec != lastSec { lastSec = sec; thermalTrack.append(thermalLevel(ProcessInfo.processInfo.thermalState)) }
                    }
                    switch kind {
                    case .cold:
                        self.main { self.phase = "\(name) · \(c.rawValue): warm-up \(warm), then \(iters) timed iterations" }
                        let cold = BenchRunner.coldSweep(det, warmup: warm, iters: iters, cancel: { self.isCancelled }) { i, ms in
                            cell.samples.append(ms); cell.sampleTimes.append(Date().timeIntervalSince(cellStart)); self.push(ms); sampleThermal()
                            if i % 5 == 0 { let p = Double(i + 1) / Double(max(iters, 1)); self.main { self.progress = p } }
                        }
                        cell.cold = cold.infer_ms; doc.cold = cold
                    case .sustained:
                        self.main { self.phase = "\(name) · \(c.rawValue): sustained \(String(format: "%.1f", minutes)) min" }
                        let su = BenchRunner.sustainedLoop(det, warmup: warm, minutes: minutes, coldIters: iters,
                                                           cancel: { self.isCancelled },
                                                           tick: { elapsed, med in
                                                               let th = thermalLevel(ProcessInfo.processInfo.thermalState)
                                                               thermalTrack.append(th)
                                                               self.main { self.liveSeconds.append((elapsed, med, th)); self.progress = min(1, elapsed / (minutes * 60)) }
                                                           },
                                                           onSample: { _, ms in
                                                               cell.samples.append(ms); cell.sampleTimes.append(Date().timeIntervalSince(cellStart)); self.push(ms)
                                                           })
                        cell.sustained = su; doc.sustained = su
                        var coldStats = StageStats(Array(cell.samples.prefix(max(iters, 1))))
                        coldStats.n = min(cell.samples.count, max(iters, 1))
                        cell.cold = coldStats
                    case .dataset:
                        guard let ds = dataset else { break }
                        var files = listImages(ds)
                        if limit > 0 && files.count > limit { files = Array(files.prefix(limit)) }
                        self.main { self.phase = "\(name) · \(c.rawValue): \(files.count) images at conf \(String(format: "%.2f", conf))" }
                        if let probe = BenchRunner.probeImage(det.imgsz) { for _ in 0..<warm { _ = try? det.inferOnly(probe) } }
                        var samples = BenchSamples()
                        let ts = Date()
                        for (i, f) in files.enumerated() {
                            if self.isCancelled { break }
                            autoreleasepool {
                                guard let cg = loadCGImage(f), let r = try? det.detect(cg, conf: conf, iou: iou) else { return }
                                samples.add(r); cell.samples.append(r.inferMs); cell.sampleTimes.append(Date().timeIntervalSince(cellStart)); self.push(r.inferMs); sampleThermal()
                            }
                            if i % 5 == 0 { let p = Double(i + 1) / Double(files.count); self.main { self.progress = p } }
                        }
                        let d = samples.dataset(wallS: Date().timeIntervalSince(ts))
                        cell.dataset = d; doc.dataset = d
                        doc.protocol.image_count = files.count; doc.protocol.image_list_sha256 = YMCore.imageListSha256(files.map { $0.path })
                        cell.cold = d.infer_ms
                    case .accuracy:
                        guard let ds = dataset else { break }
                        var files = listImages(ds)
                        if limit > 0 && files.count > limit { files = Array(files.prefix(limit)) }
                        let sibling = ds.deletingLastPathComponent().appendingPathComponent("labels")
                        let labels = FileManager.default.fileExists(atPath: sibling.path) ? sibling.path : "auto"
                        self.main { self.phase = "\(name) · \(c.rawValue): scoring \(files.count) images (val protocol)" }
                        if let probe = BenchRunner.probeImage(det.imgsz) { for _ in 0..<warm { _ = try? det.inferOnly(probe) } }
                        let o = AccuracyRunner.run(det, images: files, labels: labels,
                                                   progress: { done, total in
                                                       if done % 5 == 0 { let p = Double(done) / Double(total); self.main { self.progress = p } }
                                                   },
                                                   onInfer: { _, ms in
                                                       cell.samples.append(ms); cell.sampleTimes.append(Date().timeIntervalSince(cellStart)); self.push(ms); sampleThermal()
                                                   })
                        cell.accuracy = o.document(); doc.accuracy = o.document()
                        doc.protocol.image_count = files.count; doc.protocol.image_list_sha256 = YMCore.imageListSha256(files.map { $0.path })
                        cell.cold = o.inferMs
                    }
                    cell.thermal = thermalTrack
                    cell.document = doc
                    done.append(cell)
                    let snapshot = done
                    self.main { self.cells = snapshot; self.liveCell = cell; self.progress = nil }
                }
            }
            let cancelled = self.isCancelled
            let env = BenchEnvironment.collect()
            let record = BenchRecord(name: BenchModel.defaultName(kind, done), date: Date(), kind: kind, warmup: warm, iters: iters, minutes: minutes,
                                     cells: done, hostName: env.host, cpuModel: env.cpu_model, osVersion: env.os)
            self.main {
                self.running = false; self.progress = nil; self.meters.endRun()
                self.phase = cancelled ? "Stopped." : "Done."
                if !done.isEmpty { self.store.add(record); self.lastRecord = record }
            }
        }
    }

    private static func defaultName(_ kind: BenchKind, _ cells: [BenchCell]) -> String {
        let f = DateFormatter(); f.dateFormat = "MMM d HH:mm"
        let m = Set(cells.map(\.modelName)).sorted().joined(separator: "+")
        return "\(kind.rawValue) · \(m.isEmpty ? "-" : m) · \(f.string(from: Date()))"
    }

    // ---- export ----
    func saveJSON(_ cell: BenchCell) {
        guard let doc = cell.document else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "bench-\(cell.modelName)-\(cell.compute.rawValue).json"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try doc.json().write(to: url); note = "Saved \(url.lastPathComponent)" } catch { note = "Save failed: \(error.localizedDescription)" }
    }
    func exportCSV() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "yolomaster-bench-history.csv"
        panel.allowedContentTypes = [.commaSeparatedText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try store.csv().write(to: url, atomically: true, encoding: .utf8); note = "Exported \(store.records.count) runs" }
        catch { note = "Export failed: \(error.localizedDescription)" }
    }
}

// MARK: - sidebar

struct BenchSidebar: View {
    @ObservedObject var bench: BenchModel
    @ObservedObject var store: BenchStore
    @Binding var selectedRecord: UUID?
    let brand: Color

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                box("Models", "cube.box.fill") {
                    if bench.models.isEmpty {
                        Text("No models yet. Add a .mlpackage / .mlmodelc; the Inference mode's model is added automatically.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    ForEach(bench.models, id: \.self) { u in
                        HStack(spacing: 8) {
                            Toggle(isOn: Binding(get: { bench.selectedModels.contains(u) },
                                                 set: { if $0 { bench.selectedModels.insert(u) } else { bench.selectedModels.remove(u) } })) {
                                Text(u.deletingPathExtension().lastPathComponent).font(.callout).lineLimit(1).truncationMode(.middle)
                            }.toggleStyle(.checkbox)
                            Spacer(minLength: 4)
                            Button { bench.removeModel(u) } label: { Image(systemName: "minus.circle") }.buttonStyle(.borderless).foregroundStyle(.secondary)
                        }
                    }
                    Button { addModel() } label: { Label("Add model…", systemImage: "plus") }.controlSize(.small).disabled(bench.running)
                }
                box("Compute", "cpu") {
                    HStack(spacing: 10) {
                        ForEach(ComputeChoice.allCases) { c in
                            Toggle(c.rawValue, isOn: Binding(get: { bench.computes.contains(c) },
                                                             set: { if $0 { bench.computes.insert(c) } else { bench.computes.remove(c) } }))
                                .toggleStyle(.checkbox)
                        }
                    }.disabled(bench.running)
                    row("Preprocess") {
                        Picker("", selection: $bench.preproc) { Text("GPU (Metal)").tag(PreprocDevice.gpu); Text("CPU").tag(PreprocDevice.cpu) }
                            .pickerStyle(.segmented).labelsHidden()
                    }.disabled(bench.running)
                    Text("ANE = all compute units (Core ML decides), GPU = CPU and GPU, CPU only. Each selected model runs on each selected unit.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                box("Protocol", "list.bullet.clipboard") {
                    Picker("", selection: $bench.kind) {
                        ForEach(BenchKind.allCases, id: \.self) { Label($0.rawValue, systemImage: $0.icon).tag($0) }
                    }.pickerStyle(.menu).labelsHidden().disabled(bench.running)
                    Text(bench.kind.blurb).font(.caption2).foregroundStyle(.secondary)
                    intRow("Warm-up iterations", $bench.warmup, 0...100, step: 1)
                    if bench.kind == .cold || bench.kind == .sustained {
                        intRow(bench.kind == .cold ? "Timed iterations" : "Cold baseline iterations", $bench.iters, 10...2000, step: 10)
                    }
                    if bench.kind == .sustained { slider("Minutes", $bench.minutes, 0.5...30) }
                    if bench.kind == .dataset || bench.kind == .accuracy {
                        Button { pickDataset() } label: {
                            HStack {
                                Image(systemName: "folder").foregroundStyle(bench.datasetURL == nil ? .secondary : brand)
                                Text(bench.datasetURL?.lastPathComponent ?? "Choose images folder…").lineLimit(1).truncationMode(.middle)
                                Spacer()
                            }
                        }.buttonStyle(.bordered).controlSize(.small)
                        intRow("Image limit (0 = all)", $bench.datasetLimit, 0...5000, step: 50)
                        if bench.kind == .accuracy {
                            Text("Labels are read from the sibling labels/ folder (the ultralytics layout) or next to each image.")
                                .font(.caption2).foregroundStyle(.secondary)
                        } else {
                            slider("Confidence", $bench.conf, 0.05...0.95); slider("IoU", $bench.iou, 0.1...0.9)
                        }
                    }
                }
                if !store.records.isEmpty {
                    box("History", "clock.arrow.circlepath") {
                        ForEach(store.records) { r in
                            HStack(spacing: 8) {
                                Image(systemName: r.kind.icon).foregroundStyle(selectedRecord == r.id ? brand : .secondary).frame(width: 16)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(r.name).font(.caption).lineLimit(1).truncationMode(.middle)
                                    Text(r.fastest.map { String(format: "%.2f ms · %@ · %@", $0.headlineMs ?? 0, $0.compute.rawValue, r.kind.rawValue) } ?? r.kind.rawValue)
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                                Spacer(minLength: 4)
                            }
                            .contentShape(Rectangle())
                            .padding(6)
                            .background(RoundedRectangle(cornerRadius: 6).fill(selectedRecord == r.id ? brand.opacity(0.12) : .clear))
                            .onTapGesture { selectedRecord = selectedRecord == r.id ? nil : r.id }
                            .contextMenu {
                                Button("Delete") { store.remove([r.id]); if selectedRecord == r.id { selectedRecord = nil } }
                            }
                        }
                        HStack {
                            Button { bench.exportCSV() } label: { Label("Export CSV", systemImage: "square.and.arrow.up") }
                            Spacer()
                            Button(role: .destructive) { store.clear(); selectedRecord = nil } label: { Label("Clear", systemImage: "trash") }
                        }.controlSize(.small)
                    }
                }
            }
        }
    }

    private func addModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = true; panel.allowsMultipleSelection = true
        panel.treatsFilePackagesAsDirectories = false
        panel.message = "Choose Core ML models (.mlpackage / .mlmodelc)"
        guard panel.runModal() == .OK else { return }
        for u in panel.urls where ["mlpackage", "mlmodelc", "mlmodel"].contains(u.pathExtension.lowercased()) { bench.addModel(u) }
    }
    private func pickDataset() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.allowsMultipleSelection = false
        panel.message = bench.kind == .accuracy ? "Choose the images folder of a labelled set (labels/ next to it)" : "Choose an images folder"
        guard panel.runModal() == .OK, let u = panel.url else { return }
        bench.datasetURL = u
    }

    // small helpers (the Inference sidebar has private twins)
    private func box<C: View>(_ title: String, _ icon: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon).font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.leading, 2)
            VStack(alignment: .leading, spacing: 12) { content() }
                .frame(maxWidth: .infinity, alignment: .leading).padding(16)
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
        }
    }
    private func row<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.callout); content() }
    }
    private func slider(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack { Text(title).font(.callout); Spacer()
                Text(String(format: "%.2f", value.wrappedValue)).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule()) }
            Slider(value: value, in: range)
        }.disabled(bench.running)
    }
    private func intRow(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double) -> some View {
        let rounded = Binding<Double>(get: { value.wrappedValue }, set: { value.wrappedValue = ($0 / step).rounded() * step })
        return VStack(alignment: .leading, spacing: 3) {
            HStack { Text(title).font(.callout); Spacer()
                Text("\(Int(value.wrappedValue))").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule()) }
            Slider(value: rounded, in: range)
        }.disabled(bench.running)
    }
}

// MARK: - dashboard

struct BenchDashboard: View {
    @ObservedObject var bench: BenchModel
    @ObservedObject var store: BenchStore
    let selectedRecord: UUID?
    let brand: Color

    private var shownCells: [BenchCell] {
        if let id = selectedRecord, let r = store.records.first(where: { $0.id == id }) { return r.cells }
        return bench.cells
    }
    private var shownRecord: BenchRecord? {
        if let id = selectedRecord { return store.records.first { $0.id == id } }
        return bench.lastRecord
    }

    /// The protocol being shown: the record's for a history record, else the live selection.
    private var shownKind: BenchKind { selectedRecord != nil ? (shownRecord?.kind ?? bench.kind) : (bench.running || bench.lastRecord == nil ? bench.kind : bench.lastRecord!.kind) }
    /// Cell colours: one per (model, unit), stable across the charts and the table.
    private func cellColor(_ index: Int) -> Color {
        let palette: [Color] = [brand, .orange, .green, .purple, .pink, .teal, .indigo, .brown]
        return palette[index % palette.count]
    }

    var body: some View {
        VStack(spacing: 14) {
            header
            HStack(alignment: .top, spacing: 14) {
                VStack(spacing: 14) {
                    statCards
                    // the main chart takes whatever height the stage has; the secondary charts keep a fixed band
                    switch shownKind {
                    case .sustained:
                        timeChart.frame(minHeight: 220, maxHeight: .infinity)
                        sustainedChart.frame(height: 180)
                        if shownCells.count > 1 { comparisonChart.frame(height: 150) }
                    case .dataset:
                        timeChart.frame(minHeight: 220, maxHeight: .infinity)
                        if shownCells.count > 1 { comparisonChart.frame(height: 150) }
                    case .cold:
                        histogramChart.frame(minHeight: 220, maxHeight: .infinity)
                        if shownCells.count > 1 || (bench.running && !bench.cells.isEmpty) { comparisonChart.frame(height: 170) }
                    case .accuracy:
                        if let acc = (selectedRecord == nil ? bench.liveCell?.accuracy ?? shownCells.first?.accuracy : shownCells.first?.accuracy) {
                            accuracyChart(acc).frame(minHeight: 220, maxHeight: .infinity)
                        } else {
                            accuracyPending.frame(minHeight: 220, maxHeight: .infinity)
                        }
                        if shownCells.count > 1 { comparisonChart.frame(height: 170) }
                    }
                }
                ThermometerView(meters: bench.meters, running: bench.running).frame(width: 96)
                PowerMeterView(meters: bench.meters).frame(width: 96)
            }
            if !shownCells.isEmpty {
                resultsTable
            } else if !bench.running {
                VStack(spacing: 6) {
                    Image(systemName: "gauge.with.dots.needle.67percent").font(.system(size: 40)).foregroundStyle(.tertiary)
                    Text("Pick models, compute units and a protocol in the sidebar, then Run.").font(.callout).foregroundStyle(.secondary)
                    Text("Every run streams here as it happens and is kept in History.").font(.caption).foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, minHeight: 160)
            }
        }
        .padding(16)
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(selectedRecord != nil ? (shownRecord?.name ?? "Run") : (bench.running ? "Running" : (bench.lastRecord?.name ?? "Benchmark")))
                    .font(.title3.weight(.semibold)).lineLimit(1)
                Text(bench.running ? bench.phase : (bench.note.isEmpty ? (shownRecord.map { "\($0.cpuModel) · \($0.osVersion)" } ?? bench.phase) : bench.note))
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            if let p = bench.progress, bench.running { ProgressView(value: p).frame(width: 160) }
            if bench.running {
                Button(role: .destructive) { bench.cancel() } label: { Label("Stop", systemImage: "stop.fill") }
            } else {
                Button { bench.run() } label: { Label("Run", systemImage: "play.fill") }
                    .buttonStyle(.borderedProminent).tint(brand).keyboardShortcut(.return, modifiers: .command)
                    .disabled(bench.selectedModels.isEmpty || bench.computes.isEmpty)
            }
        }
    }

    private var liveStats: StageStats? {
        if bench.running || selectedRecord == nil { return bench.liveSamples.count > 1 ? StageStats(bench.liveSamples.map(\.ms)) : shownCells.first?.cold }
        return shownCells.first?.cold
    }
    private var statCards: some View {
        let s = liveStats
        let cell = selectedRecord == nil ? bench.liveCell : shownCells.first
        return HStack(spacing: 10) {
            card("Median", s.map { String(format: "%.2f ms", $0.median) } ?? "-", s.map { String(format: "%.1f fps", $0.median > 0 ? 1000 / $0.median : 0) } ?? "")
            card("p90 / p99", s.map { String(format: "%.2f / %.2f", $0.p90, $0.p99) } ?? "-", "ms")
            card("Min / max", s.map { String(format: "%.2f / %.2f", $0.min, $0.max) } ?? "-", "ms")
            card("Samples", s.map { "\($0.n)" } ?? "-", cell.map { "\($0.modelName) · \($0.compute.rawValue)" } ?? "")
            if let su = cell?.sustained {
                card("Throttle", String(format: "%+.1f%%", su.throttle_pct), String(format: "%.2f -> %.2f ms", su.cold_median_ms, su.sustained_median_ms))
            } else if let a = cell?.accuracy {
                card("mAP50-95", String(format: "%.4f", a.map5095), String(format: "mAP50 %.4f · %d images", a.map50, a.images))
            }
        }
    }
    private func card(_ title: String, _ value: String, _ sub: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.title3, design: .rounded).weight(.semibold).monospacedDigit()).lineLimit(1).minimumScaleFactor(0.7)
            Text(sub).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// One series of (seconds since the cell started, model ms).
    private struct Series { let name: String; let color: Color; let points: [(t: Double, ms: Double)] }
    /// While a run streams: the current cell, a rolling last-minute window. Once it is done (and for
    /// history records): every cell's full series overlaid from its own 0 s, one colour per cell.
    private var seriesToShow: (series: [Series], tStart: Double, tEnd: Double) {
        if bench.running, let c = bench.liveCell {
            let pts = bench.liveSamples
            let tEnd = max(pts.map(\.t).max() ?? 0, 1)
            // once the window rolls its width is exactly windowSeconds, so the slot width (and hence the
            // slot edges) stays constant from one redraw to the next
            let tStart = max(0, tEnd - BenchDashboard.windowSeconds)
            return ([Series(name: "\(c.modelName) · \(c.compute.rawValue)", color: brand, points: pts)],
                    tStart, tStart > 0 ? tStart + BenchDashboard.windowSeconds : max(tEnd, 1))
        }
        let cells = shownCells
        let series = cells.enumerated().map { k, c in
            Series(name: "\(c.modelName) · \(c.compute.rawValue)", color: cellColor(k),
                   points: c.samples.enumerated().map { ($0.offset < c.sampleTimes.count ? c.sampleTimes[$0.offset] : Double($0.offset), $0.element) })
        }
        let tEnd = max(series.flatMap { $0.points.map(\.t) }.max() ?? 0, 1)
        return (series, 0, tEnd)
    }
    private static let windowSeconds = 30.0
    /// Percentile-bounded y range so a single outlier never compresses the trace (values beyond it are clamped).
    private static func yRange(_ values: [Double]) -> ClosedRange<Double> {
        guard values.count > 1 else { let v = values.first ?? 0; return (v - 0.5)...(v + 0.5) }
        let sorted = values.sorted()
        let lo = sorted[Int(Double(sorted.count) * 0.01)], hi = sorted[min(Int(Double(sorted.count) * 0.99), sorted.count - 1)]
        let pad = max((hi - lo) * 0.10, 0.05)
        return (lo - pad)...(hi + pad)
    }
    private struct Bucket: Identifiable { let id: Int; let series: String; let t, lo, hi, mean, trend: Double }
    /// Decimate one series into time-aligned slots (min / max band + mean) and smooth the means.
    /// Slots are anchored to absolute time (slot k covers [k * width, (k + 1) * width)), so a rolling
    /// window never re-partitions the samples it already showed; the 8-slot moving average is warmed
    /// up on the slots preceding `tStart` and only slots inside the window are returned.
    private static func decimate(_ pts: [(t: Double, ms: Double)], name: String, from tStart: Double, to tEnd: Double,
                                 into range: ClosedRange<Double>, live: Bool) -> [Bucket] {
        guard !pts.isEmpty else { return [] }
        // live: a constant slot width (the rolling window's) even while the axis is still growing, so
        // slot edges never move; finished runs: 400 slots across the whole run
        let width = live ? BenchDashboard.windowSeconds / 400 : max((tEnd - tStart) / 400, 0.01)
        let smooth = 8
        let leadIn = tStart - Double(smooth) * width
        let clamp: (Double) -> Double = { min(max($0, range.lowerBound), range.upperBound) }
        // group by slot (the samples arrive in time order)
        var slots: [(k: Int, lo: Double, hi: Double, sum: Double, n: Int)] = []
        for p in pts where p.t >= leadIn {
            let k = Int((p.t / width).rounded(.down))
            if let last = slots.last, last.k == k {
                slots[slots.count - 1] = (k, min(last.lo, p.ms), max(last.hi, p.ms), last.sum + p.ms, last.n + 1)
            } else {
                slots.append((k, p.ms, p.ms, p.ms, 1))
            }
        }
        var out: [Bucket] = []; out.reserveCapacity(slots.count)
        var window: [Double] = []
        for sl in slots {
            let mean = sl.sum / Double(sl.n)
            window.append(mean); if window.count > smooth { window.removeFirst() }
            let t = Double(sl.k) * width
            if t + width < tStart { continue }   // lead-in slot: it only warmed the average up
            out.append(Bucket(id: sl.k, series: name, t: t, lo: clamp(sl.lo), hi: clamp(sl.hi), mean: clamp(mean),
                              trend: clamp(window.reduce(0, +) / Double(window.count))))
        }
        return out
    }

    private var timeChart: some View {
        let shown = seriesToShow
        let visible = shown.series.flatMap { s in (shown.tStart > 0 ? s.points.filter { $0.t >= shown.tStart } : s.points).map(\.ms) }
        let range = BenchDashboard.yRange(visible)
        let med = visible.isEmpty ? 0 : StageStats(visible).median
        let buckets = shown.series.map { BenchDashboard.decimate($0.points, name: $0.name, from: shown.tStart, to: shown.tEnd, into: range, live: bench.running) }
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(shownKind == .dataset ? "Model time per image" : "Model time per iteration").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                if !visible.isEmpty {
                    Text(String(format: "%@%.0f s · %d samples · median %.2f ms · y = p1..p99", bench.running ? "last " : "full run ",
                                shown.tEnd - shown.tStart, visible.count, med))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            Chart {
                ForEach(Array(buckets.enumerated()), id: \.offset) { k, bs in
                    ForEach(bs) { b in
                        AreaMark(x: .value("s", b.t), yStart: .value("min", b.lo), yEnd: .value("max", b.hi), series: .value("series", b.series + " band"))
                            .foregroundStyle(shown.series[k].color.opacity(0.16))
                        LineMark(x: .value("s", b.t), y: .value("ms", b.trend), series: .value("series", b.series))
                            .foregroundStyle(by: .value("series", b.series)).lineStyle(StrokeStyle(lineWidth: 2)).interpolationMethod(.monotone)
                    }
                }
                if !visible.isEmpty && shown.series.count == 1 {
                    RuleMark(y: .value("median", med)).foregroundStyle(.secondary.opacity(0.6)).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                }
            }
            .chartForegroundStyleScale(domain: shown.series.map(\.name), range: shown.series.map(\.color))
            .chartYAxisLabel("ms").chartXAxisLabel("seconds")
            .chartXScale(domain: shown.tStart...max(shown.tEnd, shown.tStart + 1))
            .chartYScale(domain: range)
            .chartLegend(shown.series.count > 1 ? .visible : .hidden)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// Cold sweep: the latency distribution of the timed iterations (fills up live), with the
    /// median / p90 / p99 marked. After a sweep every cell's distribution is overlaid.
    private var histogramChart: some View {
        let shown = seriesToShow
        let all = shown.series.flatMap { $0.points.map(\.ms) }
        let range = BenchDashboard.yRange(all)
        let binCount = 40
        let width = (range.upperBound - range.lowerBound) / Double(binCount)
        struct Bin: Identifiable { let id: String; let series: String; let x: Double; let n: Int }
        var bins: [Bin] = []
        for s in shown.series {
            var counts = [Int](repeating: 0, count: binCount)
            for v in s.points.map(\.ms) {
                let k = min(max(Int((v - range.lowerBound) / width), 0), binCount - 1)
                counts[k] += 1
            }
            for (k, n) in counts.enumerated() where n > 0 { bins.append(Bin(id: "\(s.name)#\(k)", series: s.name, x: range.lowerBound + (Double(k) + 0.5) * width, n: n)) }
        }
        let stats = all.count > 1 ? StageStats(all) : nil
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Latency distribution of the timed iterations").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                if let st = stats { Text(String(format: "%d samples · median %.2f · p90 %.2f · p99 %.2f ms", st.n, st.median, st.p90, st.p99)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary) }
            }
            Chart {
                ForEach(bins) { b in
                    BarMark(x: .value("ms", b.x), y: .value("count", b.n), width: .fixed(6))
                        .foregroundStyle(by: .value("series", b.series)).opacity(shown.series.count > 1 ? 0.7 : 0.9)
                }
                if let st = stats, shown.series.count == 1 {
                    RuleMark(x: .value("median", st.median)).foregroundStyle(.primary).lineStyle(StrokeStyle(lineWidth: 1.5)).annotation(position: .top, alignment: .leading) { Text("median").font(.caption2) }
                    RuleMark(x: .value("p90", st.p90)).foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3])).annotation(position: .top, alignment: .leading) { Text("p90").font(.caption2).foregroundStyle(.secondary) }
                    RuleMark(x: .value("p99", st.p99)).foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1, dash: [2, 3])).annotation(position: .top, alignment: .leading) { Text("p99").font(.caption2).foregroundStyle(.secondary) }
                }
            }
            .chartForegroundStyleScale(domain: shown.series.map(\.name), range: shown.series.map(\.color))
            .chartXAxisLabel("ms").chartYAxisLabel("iterations")
            .chartXScale(domain: range)
            .chartLegend(shown.series.count > 1 ? .visible : .hidden)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// Cells side by side: median with the p90 whisker (cold / sustained / dataset) or mAP50-95 (accuracy).
    private var comparisonChart: some View {
        let cells = bench.running ? bench.cells : shownCells
        let accuracyMode = shownKind == .accuracy
        struct Row: Identifiable { let id: UUID; let name: String; let value: Double; let hi: Double }
        let rows = cells.map { c -> Row in
            if accuracyMode { return Row(id: c.id, name: "\(c.modelName) · \(c.compute.rawValue)", value: c.accuracy?.map5095 ?? 0, hi: c.accuracy?.map50 ?? 0) }
            return Row(id: c.id, name: "\(c.modelName) · \(c.compute.rawValue)", value: c.cold?.median ?? 0, hi: c.cold?.p90 ?? 0)
        }
        return VStack(alignment: .leading, spacing: 6) {
            Text(accuracyMode ? "Cells side by side: mAP50-95 (bar) and mAP50 (tick)" : "Cells side by side: median (bar) and p90 (tick)")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Chart {
                ForEach(Array(rows.enumerated()), id: \.offset) { k, r in
                    BarMark(x: .value("value", r.value), y: .value("cell", r.name)).foregroundStyle(cellColor(k).opacity(0.85))
                        .annotation(position: .trailing) { Text(accuracyMode ? String(format: "%.4f", r.value) : String(format: "%.2f ms", r.value)).font(.caption2.monospacedDigit()) }
                    if r.hi > 0 { PointMark(x: .value("hi", r.hi), y: .value("cell", r.name)).symbol(.diamond).foregroundStyle(.primary).symbolSize(30) }
                }
            }
            .chartXAxisLabel(accuracyMode ? "mAP" : "ms")
            .chartXScale(domain: accuracyMode ? 0...1 : 0...max((rows.map(\.hi).max() ?? 1) * 1.15, 0.1))
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    private var accuracyPending: some View {
        VStack(spacing: 8) {
            if bench.running {
                ProgressView(value: bench.progress ?? 0).frame(width: 260)
                Text(bench.phase).font(.caption).foregroundStyle(.secondary)
                if bench.liveSamples.count > 1 {
                    let st = StageStats(bench.liveSamples.map(\.ms))
                    Text(String(format: "%d images scored · model median %.2f ms", st.n, st.median)).font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                }
            } else {
                Image(systemName: "checkmark.seal").font(.system(size: 30)).foregroundStyle(.tertiary)
                Text("The per-class AP chart appears once the accuracy pass has scored the set.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// Sustained: one median per second, coloured by the thermal state of that second.
    private var sustainedChart: some View {
        let points: [(t: Double, med: Double, thermal: Int)] = {
            if bench.running || selectedRecord == nil, !bench.liveSeconds.isEmpty { return bench.liveSeconds }
            if let c = shownCells.first, let su = c.sustained {
                return su.sparkline.enumerated().map { (Double($0.offset), $0.element, $0.offset < c.thermal.count ? c.thermal[$0.offset] : 0) }
            }
            return []
        }()
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Sustained: per-second median and thermal state").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                ForEach(0..<4, id: \.self) { l in HStack(spacing: 3) { Circle().fill(thermalColor(l)).frame(width: 7, height: 7); Text(thermalName(l)).font(.caption2).foregroundStyle(.secondary) } }
            }
            Chart {
                ForEach(Array(points.enumerated()), id: \.offset) { _, p in
                    LineMark(x: .value("s", p.t), y: .value("ms", p.med)).foregroundStyle(.secondary)
                    PointMark(x: .value("s", p.t), y: .value("ms", p.med)).foregroundStyle(thermalColor(p.thermal)).symbolSize(18)
                }
            }
            .chartYAxisLabel("ms").chartXAxisLabel("seconds")
            .chartXScale(domain: 0...max(points.map(\.t).max() ?? 1, 1))
            .chartYScale(domain: {   // every second is one median: show them all, padded, never outside the plot
                let ys = points.map(\.med); let lo = ys.min() ?? 0, hi = ys.max() ?? 1; let pad = max((hi - lo) * 0.12, 0.05)
                return (lo - pad)...(hi + pad)
            }())
            .chartLegend(.hidden)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    private var resultsTable: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Results").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Text("floor-rank percentiles · yolomaster-bench/v1").font(.caption2).foregroundStyle(.tertiary)
            }
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("Model").font(.caption.weight(.semibold)); Text("Unit").font(.caption.weight(.semibold)); Text("Pre").font(.caption.weight(.semibold))
                    Text("Median ms").font(.caption.weight(.semibold)); Text("p90").font(.caption.weight(.semibold)); Text("p99").font(.caption.weight(.semibold))
                    Text("Min").font(.caption.weight(.semibold)); Text("FPS").font(.caption.weight(.semibold)); Text("Extra").font(.caption.weight(.semibold)); Text("").font(.caption)
                }
                ForEach(shownCells) { c in
                    GridRow {
                        Text(c.modelName).font(.caption).lineLimit(1)
                        Text(c.compute.rawValue).font(.caption)
                        Text(c.preproc).font(.caption)
                        Text(c.cold.map { String(format: "%.2f", $0.median) } ?? "-").font(.caption.monospacedDigit())
                        Text(c.cold.map { String(format: "%.2f", $0.p90) } ?? "-").font(.caption.monospacedDigit())
                        Text(c.cold.map { String(format: "%.2f", $0.p99) } ?? "-").font(.caption.monospacedDigit())
                        Text(c.cold.map { String(format: "%.2f", $0.min) } ?? "-").font(.caption.monospacedDigit())
                        Text(String(format: "%.1f", c.fps)).font(.caption.monospacedDigit())
                        Text(extra(c)).font(.caption).lineLimit(1)
                        Button { bench.saveJSON(c) } label: { Image(systemName: "square.and.arrow.down") }.buttonStyle(.borderless).help("Save the yolomaster-bench/v1 JSON")
                            .disabled(c.document == nil)
                    }
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
    private func extra(_ c: BenchCell) -> String {
        if let su = c.sustained { return String(format: "throttle %+.1f%% over %.0fs, peak %@", su.throttle_pct, su.duration_s, thermalName(c.thermal.max() ?? 0)) }
        if let d = c.dataset { return String(format: "pre %.2f · post %.2f ms · %d frames", d.pre_ms.median, d.post_ms.median, d.frames) }
        if let a = c.accuracy { return String(format: "mAP50 %.4f · mAP50-95 %.4f · %d images", a.map50, a.map5095, a.images) }
        return ""
    }

    private func accuracyChart(_ a: BenchDocument.Accuracy) -> some View {
        let rows = a.per_class.sorted { $0.ap5095 > $1.ap5095 }
        return VStack(alignment: .leading, spacing: 6) {
            Text("Accuracy per class (AP50-95, \(rows.count) classes)").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Chart {
                ForEach(rows, id: \.class_id) { c in
                    BarMark(x: .value("class", "\(c.class_id)"), y: .value("AP", c.ap5095)).foregroundStyle(brand.opacity(0.85))
                }
                RuleMark(y: .value("mAP", a.map5095)).foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
            .chartYScale(domain: 0...1).chartXAxis { AxisMarks(values: .automatic(desiredCount: 20)) }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
}

/// A thermometer: four zones, the bulb filled to the current thermal state, the peak of the run marked.
struct ThermometerView: View {
    @ObservedObject var meters: MeterModel
    let running: Bool
    var body: some View {
        let level = meters.thermal
        return VStack(spacing: 8) {
            Text("Thermal").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            GeometryReader { g in
                let h = g.size.height, w: CGFloat = 22
                let fill = h * CGFloat(level + 1) / 4
                ZStack(alignment: .bottom) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: w)
                    Capsule().fill(LinearGradient(colors: [.green, .yellow, .orange, .red], startPoint: .bottom, endPoint: .top))
                        .frame(width: w).mask(alignment: .bottom) { Rectangle().frame(height: max(w, fill)) }
                        .animation(.easeInOut(duration: 0.8), value: level)
                    ForEach(1..<4, id: \.self) { k in
                        Rectangle().fill(Color.primary.opacity(0.25)).frame(width: w + 10, height: 1).offset(y: -h * CGFloat(k) / 4)
                    }
                    if running || meters.thermalPeak > 0 {
                        Rectangle().fill(Color.primary).frame(width: w + 14, height: 2)
                            .offset(y: -h * CGFloat(meters.thermalPeak + 1) / 4 + 1)
                            .animation(.easeInOut(duration: 0.8), value: meters.thermalPeak)
                    }
                }.frame(maxWidth: .infinity)
            }
            Text(thermalName(level)).font(.caption.weight(.semibold)).foregroundStyle(thermalColor(level))
            if running || meters.thermalPeak > 0 {
                Text("peak \(thermalName(meters.thermalPeak))").font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

}

/// Battery power flow: the bar grows downward from the zero line while discharging (watts drawn
/// from the battery) and upward while charging; a Mac on mains with a full battery sits at zero.
struct PowerMeterView: View {
    @ObservedObject var meters: MeterModel
    var body: some View {
        let b = meters.battery
        let w = b.watts
        let scale = 100.0                       // full bar = 100 W either way
        return VStack(spacing: 8) {
            Text("Power").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            GeometryReader { g in
                let h = g.size.height, barW: CGFloat = 22
                let half = h / 2
                let len = min(half, half * CGFloat(abs(w)) / scale)
                ZStack(alignment: .center) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: barW)
                    Rectangle().fill(Color.primary.opacity(0.35)).frame(width: barW + 10, height: 1)       // zero line
                    if b.present {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(w < 0 ? LinearGradient(colors: [.orange, .red], startPoint: .top, endPoint: .bottom)
                                        : LinearGradient(colors: [.green, .mint], startPoint: .bottom, endPoint: .top))
                            .frame(width: barW - 4, height: max(2, len))
                            .offset(y: w < 0 ? len / 2 : -len / 2)
                            .animation(.easeInOut(duration: 0.25), value: w)
                    }
                    ForEach([-50.0, 50.0], id: \.self) { mark in
                        Rectangle().fill(Color.primary.opacity(0.18)).frame(width: barW + 6, height: 1).offset(y: -half * CGFloat(mark) / scale)
                    }
                }.frame(maxWidth: .infinity)
            }
            if b.present {
                Text(String(format: "%@%.1f W", w < 0 ? "-" : "+", abs(w))).font(.caption.weight(.semibold).monospacedDigit())
                    .foregroundStyle(w < -0.05 ? Color.orange : (w > 0.05 ? Color.green : Color.secondary))
                    .contentTransition(.numericText())
                Text(b.charging ? "Charging" : "On battery").font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
            } else {
                Text("no battery").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text("desktop Mac").font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
}

