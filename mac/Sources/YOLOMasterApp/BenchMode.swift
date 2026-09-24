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
import IOKit.ps
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

enum PowerState: String { case pluggedIn = "Plugged in", charging = "Charging", onBattery = "On battery" }

struct BatterySample {
    let present: Bool
    let watts: Double          // > 0 into the battery, < 0 out of it, 0 when idle / no battery
    let state: PowerState      // from the IOKit Power Sources API, not inferred from the sign of the current
    let percent: Int?
    static let none = BatterySample(present: false, watts: 0, state: .pluggedIn, percent: nil)
}
enum BatteryReader {
    /// State and charge come from IOPowerSources (kIOPSPowerSourceStateKey / kIOPSIsChargingKey); the
    /// instantaneous power comes from the AppleSmartBattery registry entry (current x voltage), which
    /// the Power Sources API does not expose.
    static func read() -> BatterySample {
        var state: PowerState? = nil
        var percent: Int? = nil
        if let blob = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
           let list = IOPSCopyPowerSourcesList(blob)?.takeRetainedValue() as? [CFTypeRef] {
            for ps in list {
                guard let d = IOPSGetPowerSourceDescription(blob, ps)?.takeUnretainedValue() as? [String: Any],
                      (d[kIOPSTypeKey as String] as? String) == kIOPSInternalBatteryType else { continue }
                let onAC = (d[kIOPSPowerSourceStateKey as String] as? String) == kIOPSACPowerValue
                let charging = (d[kIOPSIsChargingKey as String] as? Bool) ?? false
                state = onAC ? (charging ? .charging : .pluggedIn) : .onBattery
                if let cur = d[kIOPSCurrentCapacityKey as String] as? Int, let mx = d[kIOPSMaxCapacityKey as String] as? Int, mx > 0 {
                    percent = Int((Double(cur) / Double(mx) * 100).rounded())
                }
                break
            }
        }
        guard let state else { return .none }          // no internal battery: a desktop Mac
        var watts = 0.0
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSmartBattery"))
        if service != 0 {
            defer { IOObjectRelease(service) }
            var propsRef: Unmanaged<CFMutableDictionary>?
            if IORegistryEntryCreateCFProperties(service, &propsRef, kCFAllocatorDefault, 0) == KERN_SUCCESS,
               let props = propsRef?.takeRetainedValue() as? [String: Any] {
                // InstantAmperage is the live value (Amperage is a rolling average that lags by a minute). A
                // discharge is negative and the registry may hand it over as an unsigned 64-bit or a 32-bit
                // two's-complement pattern, so decode the bit pattern instead of trusting a plain Int cast.
                func milliamps(_ key: String) -> Int64? {
                    guard let n = props[key] as? NSNumber else { return nil }
                    var v: Int64
                    if let i = n as? Int64 { v = i } else { v = Int64(bitPattern: n.uint64Value) }
                    if v > Int64(Int32.max) && v <= Int64(UInt32.max) { v -= Int64(1) << 32 }
                    return v
                }
                let mA = milliamps("InstantAmperage") ?? milliamps("Amperage") ?? 0
                let mV = Double((props["Voltage"] as? Int) ?? 0)
                watts = Double(mA) / 1000 * mV / 1000
            }
        }
        return BatterySample(present: true, watts: watts, state: state, percent: percent)
    }
}
func thermalColor(_ level: Int) -> Color { [Color.blue, .green, .orange, .red][max(0, min(3, level))] }

// MARK: - die temperature (AppleSMC user client: the same keys smctemp / iStat read)

/// Reads the CPU / GPU die temperature sensors through the AppleSMC IOKit user client. ProcessInfo's
/// thermalState only says whether macOS is already under thermal PRESSURE (throttling); a machine
/// whose fans keep up stays "nominal" however hot the die is. The SMC exposes the sensors themselves:
/// on Apple silicon the CPU dies are the Tp / Te / Tf keys and the GPU dies the Tg keys, all "flt "
/// values in degrees Celsius. The key set is discovered once (the SMC lists its keys by index) and
/// then read on every tick; the hottest sensor is the reported temperature.
final class SMCTemperature {
    // SMCKeyInfoData is 12 bytes in C (9 + 3 padding); Swift lays fields out at size, not stride, so
    // the padding is explicit to keep the 80-byte SMCKeyData_t layout the kernel expects
    private struct KeyInfo { var dataSize: UInt32 = 0; var dataType: UInt32 = 0; var dataAttributes: UInt8 = 0; var pad: (UInt8, UInt8, UInt8) = (0, 0, 0) }
    private struct KeyData {          // the 80-byte SMCKeyData_t of the AppleSMC user client
        var key: UInt32 = 0
        var vers: (UInt8, UInt8, UInt8, UInt8, UInt16) = (0, 0, 0, 0, 0)
        var pLimit: (UInt16, UInt16, UInt32, UInt32, UInt32) = (0, 0, 0, 0, 0)
        var keyInfo = KeyInfo()
        var result: UInt8 = 0
        var status: UInt8 = 0
        var data8: UInt8 = 0
        var data32: UInt32 = 0
        var bytes: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                    UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8) =
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    }
    private static let kSMCHandleYPCEvent: UInt32 = 2
    private static let kSMCReadKey: UInt8 = 5, kSMCGetKeyFromIndex: UInt8 = 8, kSMCGetKeyInfo: UInt8 = 9

    private var conn: io_connect_t = 0
    private var sensors: [(name: String, key: UInt32, info: KeyInfo)] = []   // key info cached: one IOKit call per sensor per read
    var sensorNames: [String] { sensors.map(\.name) }
    var available: Bool { conn != 0 && !sensors.isEmpty }

    init() {
        guard MemoryLayout<KeyData>.size == 80 else { return }   // layout guard: never talk to the SMC with a wrong struct
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"))
        guard service != 0 else { return }
        defer { IOObjectRelease(service) }
        guard IOServiceOpen(service, mach_task_self_, 0, &conn) == KERN_SUCCESS else { conn = 0; return }
        discover()
    }
    deinit { if conn != 0 { IOServiceClose(conn) } }

    private static func code(_ s: String) -> UInt32 { s.utf8.reduce(0) { ($0 << 8) | UInt32($1) } }
    private static func name(_ c: UInt32) -> String {
        String(bytes: [UInt8(c >> 24 & 0xff), UInt8(c >> 16 & 0xff), UInt8(c >> 8 & 0xff), UInt8(c & 0xff)], encoding: .ascii) ?? "????"
    }
    private func call(_ input: inout KeyData) -> KeyData? {
        var output = KeyData()
        var outSize = MemoryLayout<KeyData>.stride
        let rc = withUnsafePointer(to: &input) { ip in
            IOConnectCallStructMethod(conn, SMCTemperature.kSMCHandleYPCEvent, ip, MemoryLayout<KeyData>.stride, &output, &outSize)
        }
        return rc == KERN_SUCCESS && output.result == 0 ? output : nil
    }
    private func keyInfo(_ key: UInt32) -> KeyInfo? {
        var q = KeyData(); q.key = key; q.data8 = SMCTemperature.kSMCGetKeyInfo
        return call(&q)?.keyInfo
    }
    private func readFloat(_ key: UInt32, _ info: KeyInfo) -> Double? {
        var q = KeyData(); q.key = key; q.keyInfo = info; q.data8 = SMCTemperature.kSMCReadKey
        guard let r = call(&q) else { return nil }
        let b = withUnsafeBytes(of: r.bytes) { Array($0.prefix(Int(info.dataSize))) }
        switch SMCTemperature.name(info.dataType) {
        case "flt ": guard b.count >= 4 else { return nil }; return Double(Float(bitPattern: UInt32(b[0]) | UInt32(b[1]) << 8 | UInt32(b[2]) << 16 | UInt32(b[3]) << 24))
        case "sp78": guard b.count >= 2 else { return nil }; return Double(Int16(bitPattern: UInt16(b[0]) << 8 | UInt16(b[1]))) / 256   // Intel Macs
        case "ui8 ": return b.first.map(Double.init)
        case "ui16": guard b.count >= 2 else { return nil }; return Double(UInt16(b[0]) << 8 | UInt16(b[1]))
        default: return nil
        }
    }
    /// Enumerate every key once; keep the die sensors (Tp / Te / Tf CPU, Tg GPU, Tc CPU on Intel) that
    /// read a plausible temperature.
    private func discover() {
        guard let cnt = keyInfo(SMCTemperature.code("#KEY")), let n = readUInt32(SMCTemperature.code("#KEY"), cnt), n > 0, n < 20000 else { return }
        var found: [(String, UInt32, KeyInfo)] = []
        for i in 0..<n {
            var q = KeyData(); q.data8 = SMCTemperature.kSMCGetKeyFromIndex; q.data32 = UInt32(i)
            guard let r = call(&q) else { continue }
            let nm = SMCTemperature.name(r.key)
            guard nm.hasPrefix("Tp") || nm.hasPrefix("Te") || nm.hasPrefix("Tf") || nm.hasPrefix("Tg") || nm.hasPrefix("Tc") else { continue }
            guard let info = keyInfo(r.key), let v = readFloat(r.key, info), v > 5, v < 130 else { continue }
            found.append((nm, r.key, info))
        }
        sensors = found
    }
    private func readUInt32(_ key: UInt32, _ info: KeyInfo) -> Int? {
        var q = KeyData(); q.key = key; q.keyInfo = info; q.data8 = SMCTemperature.kSMCReadKey
        guard let r = call(&q) else { return nil }
        let b = withUnsafeBytes(of: r.bytes) { Array($0.prefix(4)) }
        return Int(UInt32(b[0]) << 24 | UInt32(b[1]) << 16 | UInt32(b[2]) << 8 | UInt32(b[3]))
    }
    /// The die temperature right now, in degrees Celsius: the mean of the die sensors (what the
    /// temperature apps report as "CPU / GPU temperature"), not the single hottest spot, which on a
    /// many-sensor SoC is a permanent outlier. nil when nothing could be read.
    func temperature() -> Double? {
        var sum = 0.0, n = 0
        for s in sensors {
            guard let v = readFloat(s.key, s.info), v > 5, v < 130 else { continue }
            sum += v; n += 1
        }
        return n > 0 ? sum / Double(n) : nil
    }
}

/// Die temperature to the four meter levels: Cool below 50 C, Normal to 80, Hot to 110, Critical above.
func thermalLevel(celsius: Double) -> Int { celsius < 50 ? 0 : (celsius < 80 ? 1 : (celsius < 110 ? 2 : 3)) }

// MARK: - the meters (their own observable so their 10 Hz ticks re-render only the two gauges)

final class MeterModel: ObservableObject {
    @Published private(set) var thermal = thermalLevel(ProcessInfo.processInfo.thermalState)
    @Published private(set) var thermalPeak = 0
    @Published private(set) var celsius: Double? = nil        // hottest die sensor (nil: no SMC sensor, pressure state only)
    @Published private(set) var celsiusPeak: Double? = nil
    @Published private(set) var battery = BatteryReader.read()
    let smc = SMCTemperature()
    private var thermalTimer: Timer?
    private var powerTimer: Timer?
    private let smcQueue = DispatchQueue(label: "com.yolomaster.smc", qos: .utility)
    private var smcBusy = false
    var tracking = false          // a run is on: keep the peak

    init() {
        // die temperature once a second: the SMC reads (one IOKit call per sensor) run off the main
        // thread and the result is published in one animated step
        thermalTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self, !self.smcBusy else { return }
            self.smcBusy = true
            let pressure = thermalLevel(ProcessInfo.processInfo.thermalState)
            self.smcQueue.async {
                let c = self.smc.available ? self.smc.temperature() : nil
                DispatchQueue.main.async {
                    self.smcBusy = false
                    let l = c.map { thermalLevel(celsius: $0) } ?? pressure
                    if let c, abs((self.celsius ?? -1) - c) >= 0.5 { self.celsius = c }
                    if l != self.thermal { self.thermal = l }
                    if self.tracking {
                        if l > self.thermalPeak { self.thermalPeak = l }
                        if let c { self.celsiusPeak = max(self.celsiusPeak ?? c, c) }
                    }
                }
            }
        }
        // the battery's instantaneous current at 10 Hz (an IOKit registry read, well under a millisecond)
        powerTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self else { return }
            let b = BatteryReader.read()
            if abs(b.watts - self.battery.watts) > 0.01 || b.state != self.battery.state || b.present != self.battery.present {
                self.battery = b
            }
        }
    }
    func startRun() { tracking = true; thermalPeak = thermal; celsiusPeak = celsius }
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
                        if sec != lastSec { lastSec = sec; thermalTrack.append(self.meters.thermal) }
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
                                                               let th = self.meters.thermal
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
            card("Median", s.map { String(format: "%.2f ms  ·  %.1f fps", $0.median, $0.median > 0 ? 1000 / $0.median : 0) } ?? "-")
            card("p90 / p99 ms", s.map { String(format: "%.2f / %.2f", $0.p90, $0.p99) } ?? "-")
            card("Min / max ms", s.map { String(format: "%.2f / %.2f", $0.min, $0.max) } ?? "-")
            card("Samples", s.map { "\($0.n)" } ?? "-")
            if let su = cell?.sustained {
                card("Throttle", String(format: "%+.1f%%", su.throttle_pct))
            } else if let a = cell?.accuracy {
                card("mAP50-95", String(format: "%.4f", a.map5095))
            }
        }
    }
    private func card(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.title3, design: .rounded).weight(.semibold).monospacedDigit()).lineLimit(1).minimumScaleFactor(0.7)
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
            Text(shownKind == .dataset ? "Model time per image" : "Model time per iteration").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
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
            Text("Latency distribution of the timed iterations").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
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
            Text("Results").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                GridRow {
                    ForEach(["Model", "Unit", "Pre", "Median ms", "p90", "p99", "Min", "FPS"], id: \.self) { h in
                        Text(h).font(.caption.weight(.semibold)).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("Extra").font(.caption.weight(.semibold)).frame(maxWidth: .infinity, alignment: .leading).gridCellColumns(2)
                    Text("").font(.caption)
                }
                ForEach(shownCells) { c in
                    GridRow {
                        ForEach(Array(rowValues(c).enumerated()), id: \.offset) { _, v in
                            cellText(v)
                        }
                        Text(extra(c)).font(.callout).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading).gridCellColumns(2)
                        Button { bench.saveJSON(c) } label: { Image(systemName: "square.and.arrow.down") }.buttonStyle(.borderless).help("Save the yolomaster-bench/v1 JSON")
                            .disabled(c.document == nil)
                    }
                }
            }
            .frame(maxWidth: .infinity)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
    private func rowValues(_ c: BenchCell) -> [String] {
        func f2(_ v: Double?) -> String { v.map { String(format: "%.2f", $0) } ?? "-" }
        return [c.modelName, c.compute.rawValue, c.preproc, f2(c.cold?.median), f2(c.cold?.p90), f2(c.cold?.p99), f2(c.cold?.min),
                String(format: "%.1f", c.fps)]
    }
    private func cellText(_ v: String) -> some View {
        Text(v).font(.callout.monospacedDigit()).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
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
        // fill: the die temperature on a 30..110 C scale when a sensor exists, else the pressure level
        let frac: CGFloat = meters.celsius.map { CGFloat(min(max(($0 - 30) / 90, 0.04), 1)) } ?? CGFloat(level + 1) / 4   // 30..120 C
        return VStack(spacing: 8) {
            Text(thermalName(level)).font(.caption.weight(.semibold)).foregroundStyle(thermalColor(level)).lineLimit(1)
            GeometryReader { g in
                let h = g.size.height, w: CGFloat = 22
                let fill = h * frac
                ZStack(alignment: .bottom) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: w)
                    Capsule().fill(LinearGradient(stops: [                                                // 30..120 C on the tube
                            .init(color: .blue, location: 0.0), .init(color: .blue, location: 0.17),        // < 50 C
                            .init(color: .green, location: 0.27), .init(color: .green, location: 0.50),     // 50 - 80 C
                            .init(color: .yellow, location: 0.60), .init(color: .orange, location: 0.83),   // 80 - 110 C
                            .init(color: .red, location: 0.90), .init(color: .red, location: 1.0)           // >= 110 C
                        ], startPoint: .bottom, endPoint: .top))
                        .frame(width: w).mask(alignment: .bottom) { Rectangle().frame(height: max(w, fill)) }
                        .animation(.easeInOut(duration: 0.6), value: frac)
                }.frame(maxWidth: .infinity)
            }
            Text(meters.celsius.map { String(format: "%.0f °C", $0) } ?? thermalName(level))
                .font(.caption.weight(.semibold).monospacedDigit()).foregroundStyle(thermalColor(level))
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
            Text(b.present ? b.state.rawValue : "Power").font(.caption.weight(.semibold)).foregroundStyle(.secondary).lineLimit(1)
            GeometryReader { g in
                let h = g.size.height, barW: CGFloat = 22
                let half = h / 2
                let len = min(half, half * CGFloat(abs(w)) / scale)
                ZStack(alignment: .center) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: barW)
                    Rectangle().fill(Color.primary.opacity(0.35)).frame(width: barW + 10, height: 1)       // zero line
                    if b.present {
                        Rectangle()
                            .fill(w < 0 ? LinearGradient(colors: [.orange, .red], startPoint: .top, endPoint: .bottom)
                                        : LinearGradient(colors: [.green, .mint], startPoint: .bottom, endPoint: .top))
                            .frame(width: barW, height: max(2, len))
                            .offset(y: w < 0 ? len / 2 : -len / 2)
                            .frame(width: barW, height: h)
                            .clipShape(Capsule())                 // the tube's outline crops the bar
                            .animation(.easeInOut(duration: 0.25), value: w)
                    }
                }.frame(maxWidth: .infinity)
            }
            if b.present {
                Text(String(format: "%@%.1f W", w < 0 ? "-" : "+", abs(w))).font(.caption.weight(.semibold).monospacedDigit())
                    .foregroundStyle(w < -0.05 ? Color.orange : (w > 0.05 ? Color.green : Color.secondary))
            } else {
                Text("no battery").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
}

