// Batch photo detection: up to 100 images in a run, gallery (3-up grid) or pager view,
// macOS-style phase progress (Loading/iCloud -> Inference), per-image stage
// averages + img/s dial. The conf / IoU levers filter the boxes kept from each photo's pass
// again - no re-runs, except for the masks of a large segmentation run.
// Photos are ADDED to the run (only the new ones are inferred) and leave it through
// Select in the gallery, which also saves a chosen set.
import CoreML
import CoreTransferable
import Photos
import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers
import YOLOMasterKit

/// A run holds at most this many photos.
private let photoLimit = 100

/// A picked file larger than this is not imported. Decimal megabytes: the unit Photos and
/// Files show sizes in, so "50 MB" here is the "50 MB" the user reads there.
/// (20 MB until 1.1.0 build 2: that left out a 12 MP ProRAW shot and most panoramas.)
/// The limit is on the FILE, not on the picture's resolution - see `decodedPhoto` for what
/// becomes of a high-resolution photo.
private let photoByteLimit = 50_000_000

private func megabytes(_ bytes: Int) -> String {
    String(format: "%.1f MB", Double(bytes) / 1_000_000)
}

/// The conf lever's lower end (`TuningPanel`). Every photo keeps the boxes its pass found
/// above this score, before NMS, and the levers filter THOSE again (`Detector.nms`) - the use
/// the Kit's `candidates` is made for. A few hundred small values per photo, so every photo
/// of a run of any size keeps them.
/// Up to 1.1.0 build 2 the levers decoded the model's whole output tensor again, and those
/// tensors (3 to 7 MB each) were kept only while the run had at most 40 photos, 20 for a
/// segmentation model. In a larger run none was kept and the levers did nothing at all, in
/// the gallery and in the pager alike, with nothing to say so.
private let candidateFloor: Float = 0.05

/// At most this many boxes are kept per photo, the highest-scoring ones. NMS keeps 300 at
/// most, so only a photo with thousands of overlapping boxes above 5% could tell the
/// difference; the cap is there for memory (a whole-body pose box carries 133 keypoints).
private let candidateCap = 3000

/// A segmentation model's masks are rendered from the prototypes of its pass (about 3 MB a
/// photo). Runs of up to this many photos keep them, and the levers redo the masks from
/// them at once. In a larger run none is kept: there the boxes still follow the levers at
/// once, and the masks are redone by running the photos again when the lever has come to
/// rest (`remaskByPass`).
private let protoLimit = 40

/// The image the run works on, from a picked photo's bytes: upright (EXIF applied), at most
/// 2048 pixels on its longer side, and DECODED here, while photos are loading.
/// ImageIO otherwise hands back an image that is decoded the first time something reads its
/// pixels. That was the model's forward pass for every photo not yet on screen - 100 to 500 ms
/// of decoding counted as preprocessing (a 505 ms "pre" beside a 13 ms inference, seen on
/// device) - and the main thread for the photos on screen. `ShouldCacheImmediately` asks for
/// the decode now; reading the pixels once makes sure of it whatever ImageIO does with the
/// flag for a thumbnail (it is the same read the Kit's preprocessing does).
///
/// A HIGH-RESOLUTION photo is treated like any other: the model sees the WHOLE picture,
/// scaled into its own input size (`Detector.imgsz`, letterboxed), however many pixels the
/// file holds. It is never cut into tiles that are run one by one - the Kit can do that
/// (`forwardPadded`), this app does not ask for it - so a 48 MP photo costs one pass, like a
/// 1 MP one, and its time on the HUD is the model's. The 2048-pixel copy made here is what
/// that scaling starts from, and what is shown and saved; the file's bytes are let go once
/// it is made, so a 50 MB file does not stay in memory.
private func decodedPhoto(_ data: Data) -> CGImage? {
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
              kCGImageSourceCreateThumbnailWithTransform: true,   // EXIF
              kCGImageSourceCreateThumbnailFromImageAlways: true,
              kCGImageSourceThumbnailMaxPixelSize: 2048,
              kCGImageSourceShouldCacheImmediately: true,
          ] as CFDictionary) else { return nil }
    autoreleasepool { _ = img.dataProvider?.data }
    return img
}

/// Untimed passes on a freshly loaded model, so that Core ML's one-off set-up is not counted
/// as the model's speed. Core ML finishes setting a model up inside its first predictions:
/// 9 s on the GPU for the first one (seen on device, it made the first photo read 9,112 ms),
/// and with ONE warm-up pass a photo still came out at 3,457 ms on Auto. So: at least three
/// passes, and more (up to six) while a pass is still far above the fastest seen. A two-stage
/// pose model also runs its pose network once, whether or not the photo has a person in it
/// (`inferOnly`). A pass that fails ends the warm-up without a report: the timed pass on the
/// same photo fails the same way, and that one is reported.
private func warmUp(_ det: Detector, on image: CGImage) {
    var fastest = Double.infinity
    for pass in 0..<6 {
        var ms = -1.0
        autoreleasepool {
            if let raw = try? det.forward(image) { ms = raw.inferMs }
        }
        if ms < 0 { break }
        if pass >= 2, ms < fastest * 2 { break }        // settled
        fastest = min(fastest, ms)
    }
    if det.isTopDown {
        autoreleasepool { _ = try? det.inferOnly(image) }
    }
}

/// What a pick did not bring in, for the message box: files over the size limit by name and
/// size, and how many could not be read at all.
private func importMessage(tooBig: [String], unreadable: Int) -> String {
    var parts: [String] = []
    if !tooBig.isEmpty {
        let listed = tooBig.prefix(3).joined(separator: ", ")
        let more = tooBig.count > 3 ? " and \(tooBig.count - 3) more" : ""
        parts.append("Not imported, over \(photoByteLimit / 1_000_000) MB: \(listed)\(more).")
    }
    if unreadable == 1 {
        parts.append("1 photo could not be read.")
    } else if unreadable > 1 {
        parts.append("\(unreadable) photos could not be read.")
    }
    return parts.joined(separator: " ")
}

/// A picked photo: its encoded bytes plus the file name the picker hands over.
/// Loading through a FILE representation is what exposes the original name
/// (IMG_4231.HEIC) with no photo-library permission: the system picker runs
/// out of process and the app only ever receives the items the user chose.
private struct PickedPhoto: Transferable {
    let data: Data?          // nil: the file is over `photoByteLimit` and was not read
    let fileName: String
    let byteCount: Int

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .image) { received in
            let name = received.file.lastPathComponent
            // size first: a file over the limit is refused without being read into memory
            let size = (try? received.file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            if size > photoByteLimit {
                return PickedPhoto(data: nil, fileName: name, byteCount: size)
            }
            // the received file is temporary - read it before returning
            let data = try Data(contentsOf: received.file)
            return PickedPhoto(data: data, fileName: name, byteCount: data.count)
        }
    }
}

/// One photo of the run with everything known about it. `id` never changes while the photo
/// is in the run, so its result, the selection and its grid cell stay attached to it when
/// other photos are added or removed.
private struct Shot: Identifiable {
    let id: Int
    let image: CGImage
    let name: String
    let type: String
    let pick: String?                       // the picker's identifier of this photo (nil: it gave none)
    var dets: [Detection] = []
    var mask: CGImage? = nil                // segmentation composite
    var cands: [Detection]? = nil           // the boxes of its pass before NMS: what the levers filter
    var protos: Detector.RawOutput? = nil   // segmentation, small runs: what its masks are rendered from
    var passConf: Float = 0                 // the levers as they stood for its pass: `mask` is of these
    var passIoU: CGFloat = 0
    var pre = 0.0                           // stage times of its pass, ms
    var inf = 0.0
    var dec = 0.0
    var maskMs = 0.0
    var wall = 0.0                          // seconds its pass took in the run loop
    var done = false                        // has a result from the model that is loaded now
    var stale = false                       // has one, but its mask is of other thresholds: gets a new pass

    /// The run loop owes this photo a pass.
    var waiting: Bool { !done || stale }
}

/// What one pass of the run loop needs, read on the main actor right before the pass.
private struct Job: @unchecked Sendable {
    let id: Int
    let image: CGImage
    let conf: Float
    let iou: CGFloat
    let keepProtos: Bool
}

/// What one pass of the run loop brings back.
private struct Pass: @unchecked Sendable {
    let dets: [Detection]
    let mask: CGImage?
    let pool: [Detection]               // the boxes above `candidateFloor`, before NMS
    let protos: Detector.RawOutput?     // nil unless the job asked for them
    let pre: Double                     // stage times, ms
    let inf: Double
    let dec: Double
    let maskMs: Double
}

/// The question put to the user before something is done with the selected photos.
private enum SelectionPrompt {
    case save(Int)
    case remove(Int)

    var title: String {
        switch self {
        case .save(let n): return n == 1 ? "Save 1 image to Photos?" : "Save \(n) images to Photos?"
        case .remove(let n): return n == 1 ? "Remove 1 photo from this run?" : "Remove \(n) photos from this run?"
        }
    }
    var message: String {
        switch self {
        case .save: return "The annotated images are added to your photo library."
        case .remove: return "The photos stay in your photo library. To have one back in the run, add it again."
        }
    }
}

/// Why a save to the photo library did not happen, in words for the user.
private struct SaveFailure {
    let message: String
    let offerSettings: Bool     // the fix is a switch in the app's page of the Settings app

    static let denied = SaveFailure(
        message: "This app is not allowed to add to your photo library, so nothing was saved. "
            + "Turn on Photos access for it in Settings, then save again.",
        offerSettings: true)
    static let restricted = SaveFailure(
        message: "Adding to the photo library is restricted on this \(deviceName), for example by "
            + "Screen Time or a device management profile. Nothing was saved.",
        offerSettings: false)

    /// The reason behind an error of the save itself. The photo library's own codes for a
    /// refused or restricted access and for a full disk are spelled out; anything else is
    /// shown as the system words it, with its code.
    static func describing(_ error: Error) -> SaveFailure {
        let ns = error as NSError
        if ns.domain == "PHPhotosErrorDomain" {
            switch ns.code {
            case 3311: return denied
            case 3310: return restricted
            case 3305:
                return SaveFailure(
                    message: "There is not enough free storage on this \(deviceName) to save the images.",
                    offerSettings: false)
            default: break
            }
        }
        return SaveFailure(
            message: "The photo library reported: \(ns.localizedDescription) (\(ns.domain) \(ns.code))",
            offerSettings: false)
    }
}

struct PhotoTestView: View {
    enum ViewMode { case gallery, pager }
    enum Phase: Equatable { case idle, loading, loadingModel, inferring }

    @State private var models: [BundledModel] = []
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = ComputeChoice.deviceDefault
    @AppStorage("allowCPU") private var allowCPU = false
    @State private var style: BoxStyle = .chip
    @State private var picks: [PhotosPickerItem] = []
    @State private var loadingPicks: Set<String> = []    // identifiers of the photos being loaded
    @State private var shots: [Shot] = []            // the run, in the order the photos were added
    @State private var nextShotID = 0
    @State private var page = 0
    @State private var viewMode: ViewMode = .gallery
    @State private var selecting = false             // gallery: taps choose photos
    @State private var selected: Set<Int> = []       // ids of the chosen photos
    @State private var conf = 0.25
    @State private var iou = 0.5
    @State private var showTuning = false
    @State private var showHUD = true
    @State private var isRunning = false             // the run loop is alive
    @State private var phase: Phase = .idle
    @State private var progress = 0
    @State private var progressTotal = 0
    @State private var errorText = ""                // a run that failed
    @State private var importNote = ""               // photos the last pick did not bring in
    @State private var lastDet: Detector?            // the loaded model ...
    @State private var detModel: BundledModel?       // ... and what it was loaded for
    @State private var detCompute: ComputeChoice?
    @State private var segOverlay: SegOverlay = .both
    @State private var isSeg = false
    @State private var pose = PoseDisplay()          // pose models: what of the skeleton is drawn
    @State private var poseKeypoints = 0             // keypoints per person of a pose model (17, 133), 0 for any other
    @State private var nmsFree = false               // end2end model: IoU lever hidden
    @State private var classNames: [String] = cocoNames
    @State private var retuneGen = 0                 // drops stale async mask renders
    @State private var maskTask: Task<Void, Never>?  // the levers' mask re-render that is alive
    @State private var settleTask: Task<Void, Never>?    // waits for the levers to rest (`settle`)
    @State private var boxScale: CGFloat = 1         // the "box" lever: box thickness and label size
    @State private var exporting = false
    @State private var exportMsg: String?            // non-nil -> "saved" popup
    @State private var exportToastGen = 0            // auto-dismiss generation
    @State private var saveProblem: SaveFailure?     // non-nil -> alert with the reason
    @State private var asking = false                // the "save / remove these?" alert is up
    @State private var prompt = SelectionPrompt.save(0)  // what it asks; kept while it closes

    private let lightHaptic = UIImpactFeedbackGenerator(style: .light)
    private let notifyHaptic = UINotificationFeedbackGenerator()

    private let gridCols = [GridItem(.flexible(), spacing: 4),
                            GridItem(.flexible(), spacing: 4),
                            GridItem(.flexible(), spacing: 4)]
    /// The gallery's columns on a wide screen (iPhone Duo, inner display): as many as fit
    /// at 120 pt or more - six held open like a book, five turned - so a cell is about the
    /// size it has on a phone. Three columns there made each photo 280 pt wide.
    private let wideGridCols = [GridItem(.adaptive(minimum: 120), spacing: 4)]
    @Environment(\.horizontalSizeClass) private var widthClass
    @Environment(\.topEdgeGap) private var topGap   // iPhone Duo: keeps the top bar off the screen's edge
    @Environment(\.hudBottomGap) private var hudGap // 20 pt over a tab bar; lower beside the Duo's tab strip
    @Environment(\.duoOuterDisplay) private var narrow   // iPhone Duo, outer display: 350 pt bars

    /// Photos are loading or a run is going: the pickers and the add button wait.
    private var busy: Bool { isRunning || phase != .idle }

    var body: some View {
        NavigationStack {
            // Like Live: the photos fill the screen and the bars float over them on a
            // translucent material. The top controls and the bottom group (messages, progress,
            // tuning panel, stats HUD) are safe-area insets, so the gallery scrolls under them
            // and comes to rest clear of them, and the pager's photo lies under them the way
            // the camera preview does.
            // The bars are exactly as wide as the tab bar and as the same bars in Live
            // (`barSideInset`), so nothing shifts sideways when switching between the two tabs.
            // The photos keep their own, narrower margin.
            ZStack { content }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .toolbar(.hidden, for: .navigationBar)
                .safeAreaInset(edge: .top) { topBar.padding(.top, topGap) }   // 0 except on an iPhone Duo
                .safeAreaInset(edge: .bottom) { bottomBars }
                .onAppear {
                    lightHaptic.prepare()
                    notifyHaptic.prepare()
                    // every time the tab appears (not only the first): a model imported or
                    // deleted in Settings must enter or leave the menu. File listing only.
                    Task.detached {
                        let found = BundledModel.discover()
                        await MainActor.run {
                            if found != models { models = found }
                            let keep = BundledModel.reselect(selectedModel, in: found)
                            if keep != selectedModel { selectedModel = keep }
                        }
                    }
                }
                .onChange(of: picks) { _, items in syncPicks(items) }
                .onChange(of: selectedModel) { _, _ in rerunAll() }
                .onChange(of: compute) { _, _ in rerunAll() }
                .onChange(of: allowCPU) { _, on in if !on && compute == .cpu { compute = ComputeChoice.deviceDefault } }
                .onChange(of: conf) { _, _ in retune() }
                .onChange(of: iou) { _, _ in retune() }
                .overlay {
                    if let msg = exportMsg {
                        ExportToast(success: true, message: msg)
                            .transition(.scale(scale: 0.82).combined(with: .opacity))
                            .onTapGesture {
                                withAnimation(.easeOut(duration: 0.2)) { exportMsg = nil }
                            }
                    }
                }
                // A save that did not happen says WHY, and stays until it is read: an alert,
                // not the toast with the red cross. When the reason is the app's Photos
                // permission, the alert leads to the switch.
                .alert("Could not save to Photos",
                       isPresented: Binding(get: { saveProblem != nil },
                                            set: { if !$0 { saveProblem = nil } }),
                       presenting: saveProblem) { problem in
                    if problem.offerSettings {
                        Button("Open Settings") {
                            if let url = URL(string: UIApplication.openSettingsURLString) {
                                UIApplication.shared.open(url)
                            }
                        }
                    }
                    Button("OK", role: .cancel) { }
                } message: { problem in
                    Text(problem.message)
                }
        }
    }

    /// Pager: THIS image's stage times + its own capability FPS.
    /// Gallery: batch averages + overall FPS + detail rows.
    /// Everything is worked out from the photos that have a result, so the numbers follow
    /// the run as photos are added to it and removed from it.
    private var hud: some View {
        let done = shots.filter { $0.done }
        let n = Double(max(done.count, 1))
        let avgPre = done.reduce(0.0) { $0 + $1.pre } / n
        let avgInf = done.reduce(0.0) { $0 + $1.inf } / n
        let avgDec = done.reduce(0.0) { $0 + $1.dec } / n
        let avgMask = done.reduce(0.0) { $0 + $1.maskMs } / n
        let wall = done.reduce(0.0) { $0 + $1.wall }
        let totalDets = done.reduce(0) { $0 + $1.dets.count }
        let throughput = wall > 0 ? Double(done.count) / wall : 0
        var shown: Shot? = nil      // the pager's photo, once it has a result
        if viewMode == .pager, shots.indices.contains(page), shots[page].done { shown = shots[page] }
        let perImage = shown != nil
        let p = shown?.pre ?? avgPre
        let i = shown?.inf ?? avgInf
        let d = shown?.dec ?? avgDec
        let mk = shown?.maskMs ?? avgMask
        let e2e = p + i + d + (isSeg ? mk : 0)
        var extras: [(String, String)] = []
        if let s = shown {
            extras = [("image", "\(page + 1)/\(shots.count)"),
                      ("size", "\(s.image.width)x\(s.image.height)"),
                      ("file", s.name),
                      ("type", s.type),
                      ("e2e", String(format: "%.1f ms", e2e)),
                      ("dets", "\(s.dets.count)")]
        } else if !done.isEmpty {
            extras = [("images", "\(shots.count)"),
                      ("wall", String(format: "%.2f s", wall)),
                      ("total dets", "\(totalDets)"),
                      ("avg e2e", String(format: "%.1f ms", e2e))]
        }
        // the green number: this photo's detections in the pager, the run's total in the gallery
        var count = totalDets
        if viewMode == .pager { count = shown?.dets.count ?? 0 }
        // pager dial = this image's END-TO-END ms (0-30 scale, band colors);
        // gallery dial = batch FPS
        return StatsHUD(fps: perImage ? e2e : throughput, pre: p, inf: i, dec: d,
                        dets: count,
                        fpsLabel: perImage ? "ms" : "FPS", active: !done.isEmpty,
                        extras: extras, mode: perImage ? .ms : .fps, fullWidth: true,
                        mask: isSeg ? mk : nil)
            .frame(maxWidth: .infinity)
            .barColumn()   // as wide as the tab bar, like Live's HUD; a phone's width on a wide screen
    }

    // MARK: pieces

    @ViewBuilder private var content: some View {
        if shots.isEmpty {
            ContentUnavailableView("No photos", systemImage: "photo.on.rectangle.angled",
                                   description: Text("Pick up to 100 images"))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            // The gallery is ALWAYS there, and the pager lies over it while a photo is open.
            // The two used to replace each other, each with a transition, and a gallery that
            // was being faded in was confined to the safe area until its transition ended:
            // for a quarter of a second the strips behind the top bar and the stats HUD were
            // black, then the photos appeared there. Now leaving a photo only takes the pager
            // away, and the gallery under it is already laid out behind the bars (and is
            // where it was scrolled to).
            ZStack {
                gallery
                    .allowsHitTesting(viewMode == .gallery)
                    .accessibilityHidden(viewMode != .gallery)
                if viewMode == .pager {
                    pager
                        .zIndex(1)      // on top of the gallery also while it fades out
                        .transition(.scale(scale: 1.08).combined(with: .opacity))
                }
            }
        }
    }

    private var gallery: some View {
        ScrollView {
            LazyVGrid(columns: widthClass == .regular ? wideGridCols : gridCols, spacing: 4) {
                ForEach(shots) { shot in
                    galleryCell(shot)
                }
            }
            .padding(.horizontal, 8)
        }
    }

    /// No page dots. Back to the gallery: pinch in on a photo at its resting size, or swipe
    /// down with one finger (both in `ZoomableScrollView`).
    /// The pager itself is the whole screen, so a photo that is zoomed in passes behind the
    /// bars. At REST the photo is fitted into what the bars leave free - centred on the whole
    /// screen, its lower edge lay under the stats HUD. The reader is laid out in the safe
    /// area, between the top bar and the bottom group, and tells each page how far that area
    /// is from the edges of the screen; it follows the bars when the HUD is expanded or
    /// hidden or the tuning panel opens. The background covers the gallery underneath.
    private var pager: some View {
        GeometryReader { free in
            let clear = free.safeAreaInsets
            TabView(selection: $page) {
                ForEach(Array(shots.enumerated()), id: \.element.id) { i, shot in
                    ZoomableScrollView(onExit: {
                        withAnimation(.spring(duration: 0.35)) { viewMode = .gallery }
                    }) {
                        annotated(shot, clear: clear)
                    }
                    .ignoresSafeArea()
                    .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .ignoresSafeArea()
            .background { Color(.systemBackground).ignoresSafeArea() }
        }
    }

    private func galleryCell(_ shot: Shot) -> some View {
        GeometryReader { geo in
            let img = shot.image
            let side = geo.size.width
            // The photo FILLS the square cell and is cropped about its centre. The boxes go
            // through that same mapping: the scale of the shorter side, and the offset that
            // centres the longer one (negative - that part is cut off). With scale = cell
            // width / image width and no offset, the boxes of a portrait photo sat below
            // their objects and those of a landscape photo came out too small.
            let scale = max(side / CGFloat(img.width), side / CGFloat(img.height))
            let ox = (side - CGFloat(img.width) * scale) / 2
            let oy = (side - CGFloat(img.height) * scale) / 2
            let picked = selecting && selected.contains(shot.id)
            ZStack(alignment: .topLeading) {
                Image(decorative: img, scale: 1).resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: side, height: side)
                    .clipped()
                // mask composite shares the photo's aspect: identical modifiers
                // guarantee pixel alignment under the .fill crop
                if segOverlay != .boxes, let mask = shot.mask {
                    Image(decorative: mask, scale: 1).resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: side, height: side)
                        .clipped()
                }
                Canvas { ctx, _ in
                    DetOverlay.draw(ctx, shot.dets, scale: scale, ox: ox, oy: oy,
                                    style: style, names: classNames,
                                    boxes: !(isSeg && segOverlay == .masks) && !skeletonOnly,
                                    labels: !skeletonOnly, pose: pose, boxScale: boxScale)
                }
            }
            .overlay(alignment: .bottomTrailing) {
                if selecting { selectionBadge(on: picked) }
            }
            .overlay {
                if picked {
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color.accentColor, lineWidth: 3)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
            .onTapGesture { tapCell(shot) }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    /// The mark in the corner of a gallery cell while photos are being selected: an empty
    /// ring, or a filled one with a tick.
    private func selectionBadge(on: Bool) -> some View {
        ZStack {
            Circle().fill(on ? Color.accentColor : Color.black.opacity(0.3))
            Circle().strokeBorder(Color.white, lineWidth: 1.5)
            if on {
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(Color.white)
            }
        }
        .frame(width: 22, height: 22)
        .padding(6)
    }

    /// A tap on a gallery cell: opens the photo in the pager, or - while selecting - puts it
    /// in or takes it out of the selection.
    private func tapCell(_ shot: Shot) {
        if selecting {
            if selected.contains(shot.id) {
                selected.remove(shot.id)
            } else {
                selected.insert(shot.id)
            }
        } else {
            page = shots.firstIndex(where: { $0.id == shot.id }) ?? 0
            withAnimation(.spring(duration: 0.35)) { viewMode = .pager }
        }
    }

    /// A message in a box of the bars' material, as Live shows a model that failed to load:
    /// the bars float over the photos now, and bare red text on a photo cannot be read.
    private func noticeBox(_ text: String) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(.red)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            .barColumn()
    }

    /// What floats over the bottom of the photos, above the tab bar. The stats HUD is last
    /// and keeps the place it has in Live (20 pt above the tab bar, `hudBottomGap`); the rest
    /// stacks on it.
    private var bottomBars: some View {
        VStack(spacing: 8) {
            if !importNote.isEmpty {
                noticeBox(importNote)
                    .onTapGesture { importNote = "" }      // read: a tap puts it away
            }
            if !errorText.isEmpty { noticeBox(errorText) }
            if phase != .idle { progressBar }
            if showTuning {
                let traits = panelTraits
                TuningPanel(conf: $conf, iou: traits.nmsFree ? nil : $iou, style: $style,
                            hudVisible: $showHUD,
                            segOverlay: traits.segment ? $segOverlay : nil,
                            pose: traits.keypoints > 0 ? $pose : nil,
                            wholeBody: traits.keypoints == 133,
                            boxScale: $boxScale)
                    .barColumn()
            }
            if showHUD { hud.padding(.bottom, hudGap) }
        }
    }

    private var progressBar: some View {
        HStack(spacing: 10) {
            if phase == .loadingModel {
                ProgressView()
                Text("Loading the model (\(compute.rawValue))")
                    .font(.caption)
                Spacer(minLength: 0)
            } else {
                Text(phase == .loading ? "Loading / iCloud" : "Inference")
                    .font(.caption)
                ProgressView(value: Double(progress), total: Double(max(progressTotal, 1)))
                    .progressViewStyle(.linear)
                Text("\(progress)/\(progressTotal)")
                    .font(.caption.monospacedDigit())
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .barColumn()
    }

    /// Label of an icon button in the top bar. Every button gets the same box, so the bar is
    /// exactly as tall as Live's (its buttons pin their labels to 20 pt too) and the buttons
    /// are equal in size.
    ///
    /// The box is narrower than Live's 22 pt and the glyph smaller, and the buttons sit 6 pt
    /// apart. That was sized for FOUR buttons beside the two pickers (select, save, tuning,
    /// add), which the gallery's bar held up to 1.1.0 build 2: at the tab bar's width four
    /// Live-sized buttons do not fit. Measured on an iPhone Air (420 pt wide): model chip 98
    /// + 8 + compute picker 77 = 183 pt, which leaves 179 pt inside the bar; four 46 pt
    /// buttons 8 pt apart need 208, four 38 pt buttons 6 pt apart need 170. (A bordered button
    /// is its label plus 12 pt on each side.) The bar now holds three - select, tuning, add in
    /// the gallery; save, tuning, add in the pager - and they keep this size, so the buttons
    /// that remain are where and as large as they were.
    ///
    /// On the outer display of an iPhone Duo (`narrow`) the bar is 350 pt wide and needs
    /// about 366 like this: there the box is 10 pt (a 34 pt button), the buttons are 4 pt
    /// apart, the bar's parts 6 pt, and its side padding is 6 pt.
    private func barIcon(_ name: String) -> some View {
        Image(systemName: name)
            .font(.system(size: 15))
            .frame(width: barIconWidth, height: 20)
    }

    private var barIconWidth: CGFloat { narrow ? 10 : 14 }

    /// The save button's label: the icon, or a spinner while images are being saved.
    @ViewBuilder private func saveLabel(_ icon: String) -> some View {
        if exporting {
            ProgressView()
                .scaleEffect(0.8)
                .frame(width: barIconWidth, height: 20)
        } else {
            barIcon(icon)
        }
    }

    /// The bar over the top of the photos: the controls, or - while photos are being
    /// selected in the gallery - what can be done with the selection.
    @ViewBuilder private var topBar: some View {
        if selecting { selectionBar } else { controls }
    }

    private var controls: some View {
        HStack(spacing: narrow ? 6 : nil) {
            Menu {
                Picker("Model", selection: $selectedModel) {
                    ForEach(models) { m in
                        // the model's type as an icon beside its name
                        Label { Text(m.fullName) } icon: { ModelIcon.image(m.kind) }
                            .tag(Optional(m))
                    }
                }
            } label: {
                ModelChip(model: selectedModel)
            }
            .fixedSize()
            .disabled(busy)
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.available(allowCPU: allowCPU)) { c in
                    Text(c.rawValue).lineLimit(1).fixedSize().tag(c)
                }
            }
            .fixedSize()
            .disabled(busy)
            Spacer(minLength: 0)        // pickers left, buttons right: the bar spans the full width
            // the buttons as one group, 6 pt apart (see `barIcon` for the arithmetic)
            HStack(spacing: narrow ? 4 : 6) {
                if !shots.isEmpty {
                    // gallery only: choose photos to save or to take out of the run.
                    // (The pager is left by a swipe down or a pinch; a photo is opened by
                    // tapping it - there is no gallery / pager button.)
                    if viewMode == .gallery {
                        Button {
                            lightHaptic.impactOccurred()
                            lightHaptic.prepare()
                            selected = []
                            withAnimation(.easeOut(duration: 0.2)) { selecting = true }
                        } label: {
                            barIcon("checkmark.circle")
                        }
                        .buttonStyle(.bordered)
                    } else {
                        // pager: save THIS image. The gallery has no save button of its own
                        // (it had one that saved the whole run at a tap, beside Select, up to
                        // 1.1.0 build 2): there a set of photos - or all of them, with Select
                        // all - is saved through Select, which asks first.
                        Button {
                            lightHaptic.impactOccurred()
                            lightHaptic.prepare()
                            exportImages(exportTargets)
                        } label: {
                            saveLabel("square.and.arrow.down")
                        }
                        .buttonStyle(.bordered)
                        .disabled(exporting || busy)
                    }
                }
                Button {
                    withAnimation { showTuning.toggle() }
                } label: {
                    barIcon("slider.horizontal.3")
                }
                .buttonStyle(.bordered)
                // The picker runs out of process, and the app needs no photo-library
                // permission to receive the chosen items. It is given the photo library
                // only so that its items carry an identifier: with those in `picks`, the
                // picker opens with the photos of the run already ticked. The identifier is
                // never used to read the library (that would need read access, which this
                // app does not ask for) - it is only compared, in `syncPicks`.
                PhotosPicker(selection: $picks,
                             maxSelectionCount: max(photoLimit - shots.filter { $0.pick == nil }.count, 1),
                             matching: .images,
                             photoLibrary: .shared()) {
                    barIcon("photo.badge.plus")
                }
                .buttonStyle(.borderedProminent)
                .disabled(busy)
            }
        }
        .padding(.horizontal, narrow ? 6 : 8)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        // as wide as the tab bar and as Live's top bar (`barSideInset`)
        .barColumn()
    }

    /// The top bar while selecting: leave, how many are chosen, Select all / Deselect, save
    /// the chosen photos (annotated, like every export), take them out of the run. Saving and
    /// removing ask first, in an alert that says how many photos it is about. Same box and
    /// button height as `controls`, so the bar does not change size.
    private var selectionBar: some View {
        let all = !shots.isEmpty && selected.count == shots.count
        return HStack(spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.2)) { selecting = false }
                selected = []
            } label: {
                barIcon("xmark")
            }
            .buttonStyle(.bordered)
            Text(selected.isEmpty ? "Select photos" : "\(selected.count) selected")
                .font(.subheadline.monospacedDigit())
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .padding(.leading, 4)
            Spacer(minLength: 0)
            Button {
                selected = all ? [] : Set(shots.map { $0.id })
            } label: {
                Text(all ? "Deselect" : "Select all")
                    .font(.subheadline)
                    .frame(height: 20)
            }
            .buttonStyle(.bordered)
            Button {
                lightHaptic.impactOccurred()
                lightHaptic.prepare()
                prompt = .save(selected.count)
                asking = true
            } label: {
                saveLabel("square.and.arrow.down")
            }
            .buttonStyle(.bordered)
            .disabled(selected.isEmpty || exporting || busy)
            Button(role: .destructive) {
                lightHaptic.impactOccurred()
                lightHaptic.prepare()
                prompt = .remove(selected.count)
                asking = true
            } label: {
                barIcon("trash")
            }
            .buttonStyle(.bordered)
            .disabled(selected.isEmpty)
        }
        .padding(.horizontal, narrow ? 6 : 8)       // as in `controls`
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .barColumn()
        .alert(Text(prompt.title), isPresented: $asking) {
            switch prompt {
            case .save:
                Button("Save") { exportImages(selected) }
            case .remove:
                Button("Remove", role: .destructive) { removeSelected() }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text(prompt.message)
        }
    }

    /// One page of the pager. The view it is hosted in covers the whole screen; `clear` is how
    /// much of each edge the bars take. The photo is fitted and centred inside what is left,
    /// so nothing covers it at rest, and because the padding belongs to the zoomed view, every
    /// part of a zoomed photo can still be brought out from behind the bars.
    /// The sides count too: on an iPhone Duo held open like a book the tabs are in a strip at
    /// the side, 84 pt wide, and the bars are centred in what it leaves. Centred on the whole
    /// screen, the photo stood 42 pt to the side of the bars above and below it. On a phone
    /// held upright the side insets are 0 and the margin is the 8 pt it always was.
    private func annotated(_ shot: Shot, clear: EdgeInsets) -> some View {
        GeometryReader { geo in
            let img = shot.image
            let scale = min(geo.size.width / CGFloat(img.width),
                            geo.size.height / CGFloat(img.height))
            let w = CGFloat(img.width) * scale
            let h = CGFloat(img.height) * scale
            ZStack(alignment: .topLeading) {
                Image(decorative: img, scale: 1).resizable()
                    .frame(width: w, height: h)
                if let mask = shot.mask, segOverlay != .boxes {
                    Image(decorative: mask, scale: 1).resizable()
                        .frame(width: w, height: h)
                }
                Canvas { ctx, _ in
                    DetOverlay.draw(ctx, shot.dets, scale: scale, ox: 0, oy: 0, style: style,
                                    names: classNames,
                                    boxes: !(isSeg && segOverlay == .masks) && !skeletonOnly,
                                    labels: !skeletonOnly, pose: pose, boxScale: boxScale)
                }
                .frame(width: w, height: h)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(EdgeInsets(top: clear.top, leading: clear.leading + 8,
                            bottom: clear.bottom, trailing: clear.trailing + 8))
    }

    // MARK: pipeline

    /// The picker's selection IS the set of picked photos in the run: it opens with them
    /// ticked, a photo ticked there joins the run (and only it is inferred), a photo
    /// unticked there leaves it. Photos are matched by the identifier the picker gives each
    /// item, so the same photo cannot be in the run twice.
    /// Called for every change of `picks`, the picker's as well as this view's own tidying
    /// (`tidyPicks`); when selection and run already agree it does nothing.
    private func syncPicks(_ items: [PhotosPickerItem]) {
        let chosen = Set(items.compactMap { $0.itemIdentifier })
        let gone = Set(shots.filter { shot in
            guard let pick = shot.pick else { return false }     // not from an identified item: stays
            return !chosen.contains(pick)
        }.map { $0.id })
        if !gone.isEmpty {
            withAnimation(.easeOut(duration: 0.2)) { dropShots(gone) }
        }
        let known = Set(shots.compactMap { $0.pick }).union(loadingPicks)   // in the run, or on the way
        let fresh = items.filter { item in
            guard let pick = item.itemIdentifier else { return true }
            return !known.contains(pick)
        }
        load(fresh)
    }

    /// `picks` brought back to the photos that are in the run: an item that was refused
    /// (over the size limit), could not be read or came with no identifier is taken out, and
    /// so is one whose photo was removed with Select. The picker then ticks exactly what is
    /// in the run the next time it opens.
    private func tidyPicks() {
        let inRun = Set(shots.compactMap { $0.pick }).union(loadingPicks)   // not those still loading
        let kept = picks.filter { item in
            guard let pick = item.itemIdentifier else { return false }
            return inRun.contains(pick)
        }
        if kept.count != picks.count { picks = kept }
    }

    /// Photos out of the run, by id. Nothing happens to them in the photo library.
    private func dropShots(_ ids: Set<Int>) {
        shots.removeAll { ids.contains($0.id) }
        selected.subtract(ids)
        page = min(page, max(shots.count - 1, 0))
        if shots.isEmpty {
            importNote = ""
            selecting = false
        }
    }

    /// Loads the photos newly ticked in the picker and adds them to the run.
    /// A file over `photoByteLimit` is not imported, and the user is told which.
    private func load(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        let take = Array(items.prefix(max(photoLimit - shots.count, 0)))
        guard !take.isEmpty else {
            tidyPicks()         // the run is full: what was ticked beyond it is not kept ticked
            return
        }
        let taking = Set(take.compactMap { $0.itemIdentifier })
        loadingPicks.formUnion(taking)
        phase = .loading
        progress = 0
        progressTotal = take.count
        importNote = ""
        let firstID = nextShotID
        nextShotID += take.count
        let firstNumber = shots.count + 1
        // off the main actor: each photo is decoded here (`decodedPhoto`), and that must not
        // hold up the interface. State is touched only inside `MainActor.run`.
        Task.detached(priority: .userInitiated) {
            var added: [Shot] = []
            var tooBig: [String] = []
            var unreadable = 0
            for (idx, item) in take.enumerated() {
                var name = "IMG_\(firstNumber + idx)"
                var type = item.supportedContentTypes.first?
                    .preferredFilenameExtension?.uppercased() ?? "-"
                // File representation first: it carries the picked file's own
                // name (IMG_4231.HEIC) and needs NO photo-library permission.
                // If the picker cannot hand an item over as a file, fall back
                // to its plain bytes and keep the positional name.
                var data: Data? = nil
                var bytes = 0
                if let picked = try? await item.loadTransferable(type: PickedPhoto.self) {
                    data = picked.data
                    bytes = picked.byteCount
                    let full = picked.fileName
                    let stem = (full as NSString).deletingPathExtension
                    if !stem.isEmpty { name = stem }
                    let ext = (full as NSString).pathExtension
                    if !ext.isEmpty { type = ext.uppercased() }
                } else if let plain = try? await item.loadTransferable(type: Data.self) {
                    data = plain
                    bytes = plain.count
                }
                if bytes > photoByteLimit {
                    tooBig.append("\(name) (\(megabytes(bytes)))")
                } else if let data, let img = decodedPhoto(data) {
                    added.append(Shot(id: firstID + idx, image: img, name: name, type: type,
                                      pick: item.itemIdentifier))
                } else {
                    unreadable += 1
                }
                let done = idx + 1
                await MainActor.run { progress = done }
            }
            let newShots = added
            let note = importMessage(tooBig: tooBig, unreadable: unreadable)
            await MainActor.run {
                let wasEmpty = shots.isEmpty
                let firstNew = shots.count
                shots.append(contentsOf: newShots)
                loadingPicks.subtract(taking)
                if !isRunning { phase = .idle }     // a run that started meanwhile owns the bar
                importNote = note
                if !newShots.isEmpty {
                    if wasEmpty {
                        page = 0
                        viewMode = newShots.count > 1 ? .gallery : .pager
                    } else if viewMode == .pager {
                        page = firstNew         // the pager goes to the first new photo
                    }
                }
                tidyPicks()                     // what did not make it is not ticked next time
                runPending()                    // only the new photos have no result yet
            }
        }
    }

    /// The model or the compute unit changed: every photo needs a result from the new one.
    /// What the levers work on goes at once, so they cannot filter the previous model's
    /// boxes over what the new run is filling in.
    private func rerunAll() {
        guard !shots.isEmpty else { return }
        var next = shots
        for k in next.indices {
            next[k].done = false
            next[k].stale = false
            next[k].cands = nil
            next[k].protos = nil
        }
        shots = next
        runPending()        // a loop that is running notices the change itself and restarts
    }

    /// Runs the model on every photo that has no result yet: all of them after a model or
    /// compute change, only the new ones after photos were added (the loaded model is kept
    /// and used again). One loop at a time. Before each pass it asks, on the main actor, for
    /// the next photo without a result - so a photo removed meanwhile is simply never asked
    /// for - and it stops as soon as the model on screen is no longer the one it holds.
    private func runPending() {
        guard !isRunning, let m = selectedModel, shots.contains(where: { $0.waiting }) else { return }
        let choice = compute
        let loaded: Detector? = (detModel == m && detCompute == choice) ? lastDet : nil
        isRunning = true
        errorText = ""
        progress = 0
        progressTotal = shots.filter { $0.waiting }.count
        phase = loaded == nil ? .loadingModel : .inferring
        let warm = shots.first(where: { $0.waiting })?.image    // for the warm-up passes below
        // the user is waiting for this: not the default priority, where work that only has
        // to get done some time may be scheduled ahead of it
        Task.detached(priority: .userInitiated) {
            var ready = loaded != nil       // false until the model of this run is loaded
            do {
                let det: Detector
                if let loaded {
                    det = loaded
                } else {
                    det = try m.makeDetector(compute: choice.mode)
                    // untimed passes on the first waiting photo, while the bar still says
                    // "Loading the model": Core ML's set-up is not the model's speed (`warmUp`)
                    if let warm { warmUp(det, on: warm) }
                    await MainActor.run { adopt(det, model: m, compute: choice) }
                }
                ready = true
                var mark = CFAbsoluteTimeGetCurrent()
                while true {
                    let job: Job? = await MainActor.run { nextJob(model: m, compute: choice, det: det) }
                    guard let job else { break }
                    let pass: Pass = try autoreleasepool {
                        let a = CFAbsoluteTimeGetCurrent()
                        let raw = try det.forward(job.image)
                        let b = CFAbsoluteTimeGetCurrent()
                        let d = det.decode(raw, conf: job.conf, iou: det.nmsThreshold(lever: job.iou))
                        let c = CFAbsoluteTimeGetCurrent()
                        // masks render for EVERY image
                        let mk = det.isSegment
                            ? det.maskOverlay(d, raw, maxSide: 1024) : nil
                        let e = CFAbsoluteTimeGetCurrent()
                        // for the levers, after the timed stages: the boxes down to the conf
                        // lever's lower end, and the prototypes when the run is small enough
                        let pool = Array(det.candidates(raw, confFloor: candidateFloor).prefix(candidateCap))
                        return Pass(dets: d, mask: mk, pool: pool,
                                    protos: job.keepProtos ? raw.maskOnly() : nil,
                                    pre: (b - a) * 1000 - raw.inferMs, inf: raw.inferMs,
                                    dec: (c - b) * 1000, maskMs: (e - c) * 1000)
                    }
                    // this photo's share of the run's wall time: from the end of the previous
                    // pass to the end of this one, so the shares add up to the whole loop
                    let now = CFAbsoluteTimeGetCurrent()
                    let wall = now - mark
                    mark = now
                    await MainActor.run { take(pass, of: job, wall: wall) }
                }
                await MainActor.run { finishRun() }
            } catch {
                let text = error.localizedDescription
                let loadFailed = !ready
                await MainActor.run { failRun(text, loadFailed: loadFailed) }
            }
        }
    }

    /// A pass has come back: the photo gets its result and what the levers need of it.
    private func take(_ pass: Pass, of job: Job, wall: Double) {
        progress += 1
        // by id, not by position: the photo may have moved up, or be gone
        guard let k = shots.firstIndex(where: { $0.id == job.id }) else { return }
        var s = shots[k]
        s.dets = pass.dets
        s.mask = pass.mask
        s.cands = pass.pool
        s.protos = pass.protos
        s.passConf = job.conf
        s.passIoU = job.iou
        s.pre = pass.pre
        s.inf = pass.inf
        s.dec = pass.dec
        s.maskMs = pass.maskMs
        s.wall = wall
        s.done = true
        s.stale = false
        shots[k] = s
        // a lever moved while this photo was with the model: its boxes are filtered again
        // like every other photo's, and its mask is redone
        if job.conf != Float(conf) || job.iou != CGFloat(iou) { retune() }
    }

    /// A model has just been loaded for the run. What is known about it is set here - before
    /// its first result is on screen - so the IoU lever, the mask and pose controls and the
    /// class names are this model's from the start. Nothing drawn may come from the model
    /// that was loaded before: every photo loses its result and waits for a new one.
    private func adopt(_ det: Detector, model: BundledModel, compute choice: ComputeChoice) {
        lastDet = det
        detModel = model
        detCompute = choice
        nmsFree = det.end2end
        isSeg = det.isSegment
        poseKeypoints = det.isPose ? det.keypointCount : 0
        classNames = det.classNames
        var next = shots
        for k in next.indices {
            next[k].dets = []
            next[k].mask = nil
            next[k].cands = nil
            next[k].protos = nil
            next[k].done = false
            next[k].stale = false
        }
        shots = next
        progress = 0
        progressTotal = next.count
        phase = .inferring
    }

    /// The next photo that is owed a pass, with the thresholds as they are right now; nil
    /// when there is none, or when another model or unit has been chosen since this loop
    /// started. The photo that is open in the pager goes first: it is the one being looked at.
    private func nextJob(model: BundledModel, compute choice: ComputeChoice, det: Detector) -> Job? {
        guard selectedModel == model, compute == choice else { return nil }
        var open: Shot? = nil
        if viewMode == .pager, shots.indices.contains(page), shots[page].waiting { open = shots[page] }
        guard let shot = open ?? shots.first(where: { $0.waiting }) else { return nil }
        // mask prototypes are kept on modest runs only (`protoLimit`): 100 of them would be
        // some 300 MB on top of the photos. Past the limit NONE is kept, also not those kept
        // so far, so every mask of the run is redone the same way.
        let keep = det.isSegment && shots.count <= protoLimit
        if !keep, shots.contains(where: { $0.protos != nil }) {
            var next = shots
            for k in next.indices { next[k].protos = nil }
            shots = next
        }
        return Job(id: shot.id, image: shot.image, conf: Float(conf), iou: CGFloat(iou), keepProtos: keep)
    }

    private func finishRun() {
        isRunning = false
        phase = .idle
        // the model or unit changed while the loop ran: it stopped early, start over with it
        runPending()
    }

    private func failRun(_ text: String, loadFailed: Bool) {
        errorText = "ERROR: \(text)"
        isRunning = false
        phase = .idle
        guard loadFailed else { return }
        // the chosen model did not load: what is on screen belongs to another model
        lastDet = nil
        detModel = nil
        detCompute = nil
        isSeg = false
        poseKeypoints = 0
        var next = shots
        for k in next.indices {
            next[k].dets = []
            next[k].mask = nil
            next[k].cands = nil
            next[k].protos = nil
            next[k].done = false
            next[k].stale = false
        }
        shots = next
    }

    /// Which controls the tuning panel offers. They are the SELECTED model's from the moment
    /// it is picked. `isSeg`, `nmsFree` and `poseKeypoints` belong to the model that is
    /// LOADED and decide how the results on screen are drawn; they change only when the new
    /// model has been loaded (`adopt`) - seconds later, or not at all while there is no photo
    /// to run it on - and the panel used to wait for them. While the loaded model is not the
    /// selected one, the panel goes by what the selected model's files say (`tuningTraits`);
    /// the drawing is untouched and stays the loaded model's.
    private var panelTraits: (segment: Bool, nmsFree: Bool, keypoints: Int) {
        if let picked = selectedModel, lastDet == nil || detModel != picked {
            return picked.tuningTraits
        }
        return (segment: isSeg, nmsFree: nmsFree, keypoints: poseKeypoints)
    }

    /// A pose model set to show the skeleton only: no person boxes, no labels - in the
    /// gallery, the pager and the exported images alike.
    private var skeletonOnly: Bool { poseKeypoints > 0 && pose.overlay == .skeleton }

    /// What the save button of the normal top bar saves: the photo that is open. The button
    /// is in the pager only; in the gallery photos are saved through Select.
    private var exportTargets: Set<Int> {
        guard viewMode == .pager, shots.indices.contains(page) else { return [] }
        return [shots[page].id]
    }

    /// Takes the selected photos out of the run. They are not touched in the photo library
    /// (the app could not: its access is add-only); to have one back, add it again.
    private func removeSelected() {
        guard !selected.isEmpty else { return }
        let gone = selected
        withAnimation(.easeOut(duration: 0.2)) {
            dropShots(gone)
            selected = []
            selecting = false
        }
        tidyPicks()         // they are not ticked the next time the picker opens
    }

    /// Export annotated images (as rendered: mask layer + boxes per the current
    /// overlay mode and style) to the photo library, then pop the result: a tick when they
    /// are saved, an alert with the reason when they are not.
    private func exportImages(_ ids: Set<Int>) {
        let chosen = shots.filter { ids.contains($0.id) }      // in the order of the gallery
        guard !exporting, !chosen.isEmpty else { return }
        exporting = true
        let drawMask = isSeg && segOverlay != .boxes
        let drawBoxes = !(isSeg && segOverlay == .masks) && !skeletonOnly
        let labelMode: LabelMode = skeletonOnly ? .off : .full
        let poseNow = pose
        let names = classNames
        let kitStyle = style.kitStyle
        let boxNow = boxScale       // the saved image follows the "box" lever like the screen
        Task.detached(priority: .userInitiated) {
            var uis: [UIImage] = []
            for s in chosen {
                autoreleasepool {
                    let cg = composeExport(s.image, dets: s.dets, mask: s.mask,
                                           drawMask: drawMask, drawBoxes: drawBoxes,
                                           names: names, style: kitStyle,
                                           label: labelMode, pose: poseNow, boxScale: boxNow)
                    uis.append(UIImage(cgImage: cg))
                }
            }
            var status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
            if status == .notDetermined {
                status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            }
            var failure: SaveFailure? = nil
            if status == .denied {
                failure = SaveFailure.denied
            } else if status == .restricted {
                failure = SaveFailure.restricted
            } else {
                do {
                    let toSave = uis
                    // Never send an EMPTY change request: one that creates nothing is not an
                    // add-only request - PhotoKit asks for read-write access for it, and with
                    // no NSPhotoLibraryUsageDescription in the bundle iOS aborts the app.
                    if !toSave.isEmpty {
                        try await PHPhotoLibrary.shared().performChanges {
                            for ui in toSave {
                                PHAssetChangeRequest.creationRequestForAsset(from: ui)
                            }
                        }
                    }
                } catch {
                    failure = SaveFailure.describing(error)
                }
            }
            let n = uis.count
            let problem = failure
            await MainActor.run { finishExport(saved: n, problem: problem) }
        }
    }

    private func finishExport(saved n: Int, problem: SaveFailure?) {
        exporting = false
        // App Store purchase-style feedback: the system success notification haptic
        // paired with the tick drawing in; the error haptic with the alert
        notifyHaptic.notificationOccurred(problem == nil ? .success : .error)
        notifyHaptic.prepare()
        if let problem {
            saveProblem = problem
            return
        }
        exportToastGen += 1
        let gen = exportToastGen
        withAnimation(.spring(duration: 0.3)) {
            exportMsg = n == 1 ? "Saved 1 image to Photos" : "Saved \(n) images to Photos"
        }
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_600_000_000)
            if exportToastGen == gen {
                withAnimation(.easeOut(duration: 0.25)) { exportMsg = nil }
            }
        }
    }

    /// A lever moved: every photo's boxes are filtered again from the boxes its pass kept
    /// (`Shot.cands`, `Detector.nms`) - at once, on the main actor, for a run of any size.
    /// A photo that is still waiting for the loaded model has none: it gets the thresholds
    /// of the moment its turn comes.
    /// Segmentation masks follow: from the prototypes where the photo holds them
    /// (`remask`), by a new pass once the lever rests where it does not (`settle`).
    private func retune() {
        guard let det = lastDet else { return }
        let confNow = Float(conf)
        let iouNow = det.nmsThreshold(lever: CGFloat(iou))
        var next = shots
        var held: [(Int, Detector.RawOutput, [Detection])] = []   // masks to render from prototypes
        var byPass = false          // masks only a new pass can redo
        var touched = false
        for k in next.indices {
            guard let pool = next[k].cands else { continue }
            let d = Detector.nms(pool, conf: confNow, iou: iouNow)
            next[k].dets = d
            touched = true
            guard det.isSegment else { continue }
            if let protos = next[k].protos {
                held.append((next[k].id, protos, d))
            } else {
                byPass = true
            }
        }
        guard touched else { return }
        shots = next
        guard det.isSegment else { return }
        remask(held, with: det)
        if byPass { settle() }
    }

    /// Renders the masks of `todo` again, off the main actor, ONE render task at a time: a
    /// new lever step cancels the task of the step before. (Each step used to start a task
    /// of its own that rendered every photo to the end, its result thrown away if it was
    /// stale - a drag queued dozens of them and the masks of the last step came seconds
    /// late.) The photo that is open in the pager is rendered first, and each mask is shown
    /// as soon as it is ready; the others wait a moment, so that while the lever is moving
    /// only that first photo is rendered per step.
    private func remask(_ todo: [(Int, Detector.RawOutput, [Detection])], with det: Detector) {
        retuneGen += 1
        let gen = retuneGen
        maskTask?.cancel()
        maskTask = nil
        guard !todo.isEmpty else { return }
        var order = todo
        if viewMode == .pager, shots.indices.contains(page) {
            let openID = shots[page].id
            if let i = order.firstIndex(where: { $0.0 == openID }) { order.swapAt(0, i) }
        }
        let list = order
        maskTask = Task.detached(priority: .userInitiated) {
            for (n, item) in list.enumerated() {
                if Task.isCancelled { return }
                let (id, protos, dets) = item
                let mk: CGImage? = autoreleasepool { det.maskOverlay(dets, protos, maxSide: 1024) }
                await MainActor.run {
                    // by id, and only while this is still the latest step and the photo
                    // still holds the pass the mask is from
                    guard gen == retuneGen, let k = shots.firstIndex(where: { $0.id == id }),
                          shots[k].protos === protos else { return }
                    shots[k].mask = mk
                }
                if n == 0 { try? await Task.sleep(nanoseconds: 80_000_000) }
            }
        }
    }

    /// Waits until the levers have been still for a third of a second, then has the masks
    /// that only the model can redo redone (`remaskByPass`). Every lever step starts the
    /// wait again, so nothing is run while a lever is being dragged.
    private func settle() {
        settleTask?.cancel()
        settleTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 350_000_000)
            guard !Task.isCancelled else { return }
            remaskByPass()
        }
    }

    /// The levers have come to rest. A segmentation photo that holds no prototypes (a run of
    /// more than `protoLimit` photos) can get the mask of the new thresholds only from the
    /// model: each such photo whose mask is of other thresholds is run again. Its boxes are
    /// right already, and it keeps what it shows until its turn comes; the stats HUD keeps
    /// counting it. A loop that is alive takes these photos as it goes.
    private func remaskByPass() {
        guard let det = lastDet, det.isSegment else { return }
        let confNow = Float(conf)
        let leverNow = CGFloat(iou)
        var next = shots
        var any = false
        for k in next.indices where next[k].done && next[k].cands != nil && next[k].protos == nil {
            if next[k].passConf != confNow || next[k].passIoU != leverNow {
                next[k].stale = true
                any = true
            }
        }
        guard any else { return }
        shots = next
        if isRunning {
            progress = 0        // the bar counts what is waiting now
            progressTotal = shots.filter { $0.waiting }.count
        } else {
            runPending()
        }
    }
}

// ExportToast / TickShape / CrossShape moved to StatsHUD.swift (shared with Bench).

/// Bake the on-screen annotation (mask composite under Kit-rendered boxes and
/// labels) into a standalone image for export.
private func composeExport(_ img: CGImage, dets: [Detection], mask: CGImage?,
                           drawMask: Bool, drawBoxes: Bool,
                           names: [String], style: YOLOMasterKit.BoxStyle,
                           label: LabelMode, pose: PoseDisplay, boxScale: CGFloat) -> CGImage {
    var out = img
    if drawMask, let mask,
       let ctx = CGContext(data: nil, width: img.width, height: img.height,
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
        ctx.interpolationQuality = .high
        let full = CGRect(x: 0, y: 0, width: img.width, height: img.height)
        ctx.draw(img, in: full)
        ctx.draw(mask, in: full)
        if let o = ctx.makeImage() { out = o }
    }
    if !dets.isEmpty {
        out = annotate(out, dets, names: names, style: style,
                       label: label, drawBoxes: drawBoxes, pose: pose, boxScale: boxScale) ?? out
    }
    return out
}

/// What a page of the pager hands to SwiftUI: it holds the zoomable scroll view and keeps it
/// on the WINDOW from top to bottom, wherever SwiftUI and the pager have put the page.
///
/// A page's photo is placed by padding it from the top and the bottom of the screen
/// (`annotated`), and that is only right while the page really is the screen. On an iPhone
/// Duo held open like a book the page that was open when the pager came up lay 17 pt too
/// high - half the height of the home-indicator inset at the bottom, the mark of a view
/// centred in the screen less that inset - so the photo's top was under the top bar and a gap
/// stood above the stats HUD. Pages reached by a swipe were right, and so was the same page
/// with the Duo turned upright, where the insets at top and bottom are equal. Why that one
/// page is put there was not found, so its frame is not trusted: this view looks where it is
/// on the window and gives the scroll view the frame that starts at the window's top edge and
/// is as tall as the window. Where the page is the screen - every page seen on an iPhone Air -
/// that frame is this view's own bounds, and nothing differs from the scroll view being the
/// page itself.
final class PagePin: UIView {
    let scroll = UIScrollView()
    private var lift: CGFloat = 0      // the scroll view's top, measured from this view's top
    private var extra: CGFloat = 0     // the scroll view's height less this view's height
    private var waiting = 0            // looks still owed while a transition scales the pager

    override init(frame: CGRect) {
        super.init(frame: frame)
        addSubview(scroll)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    override func layoutSubviews() {
        super.layoutSubviews()
        pin()
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        guard window != nil else { return }
        pin()
        // and again when things have settled: a page can be moved after its first layout
        // without anything in this view being told
        for delay in [0.3, 0.8] {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in self?.pin() }
        }
    }

    // moved without a change of size: no layout pass comes for that
    override var frame: CGRect {
        didSet { if frame.origin != oldValue.origin { pin() } }
    }
    override var center: CGPoint {
        didSet { if center != oldValue { pin() } }
    }

    /// Put the scroll view on the window from top to bottom. Called on every layout, on a
    /// move, on every SwiftUI update of the page and when a zoom ends; it changes something
    /// only when the page is not where the screen is.
    func pin() {
        if let window {
            let seen = convert(bounds, to: window)
            if abs(seen.width - bounds.width) > 0.5 || abs(seen.height - bounds.height) > 0.5 {
                // The pager is growing into place or shrinking away - its transition scales
                // it - and what is measured through that scale is not where the page will
                // rest. Keep what is known and look again shortly.
                if waiting < 15 {
                    waiting += 1
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) { [weak self] in self?.pin() }
                }
            } else {
                waiting = 0
                lift = -seen.minY
                extra = window.bounds.height - bounds.height
            }
        }
        let want = CGRect(x: 0, y: lift, width: bounds.width, height: bounds.height + extra)
        let now = scroll.frame
        if abs(now.minX - want.minX) > 0.25 || abs(now.minY - want.minY) > 0.25
            || abs(now.width - want.width) > 0.25 || abs(now.height - want.height) > 0.25 {
            scroll.frame = want
        }
        // At rest the content is exactly as large as the scroll view, so any offset is a
        // leftover that would shift the photo the same way.
        if scroll.zoomScale <= scroll.minimumZoomScale * 1.01, !scroll.isZooming,
           !scroll.isZoomBouncing, !scroll.isDragging, !scroll.isDecelerating,
           scroll.contentOffset != .zero {
            scroll.contentOffset = .zero
        }
    }
}

/// Photos-app zoom behavior via UIScrollView: pinch zoom anchored at the
/// fingers, one-finger pan ONLY while zoomed (1x swipes pass through to the
/// pager), double-tap zooms to the tapped point. Two ways back to the gallery
/// (`onExit`): pinching below minimum zoom (bounce region), and - at rest, not
/// zoomed in - dragging the photo down with one finger and letting go.
struct ZoomableScrollView<Content: View>: UIViewRepresentable {
    var onExit: (() -> Void)?
    @ViewBuilder var content: Content

    func makeCoordinator() -> Coordinator {
        let host = UIHostingController(rootView: content)
        // the pager lies under the bars and the status bar: the hosted photo is laid out in
        // the whole view, not in what the safe area leaves of it
        host.safeAreaRegions = []
        return Coordinator(host: host)
    }

    func makeUIView(context: Context) -> PagePin {
        let page = PagePin()
        let sv = page.scroll
        sv.minimumZoomScale = 1
        sv.maximumZoomScale = 5
        sv.bouncesZoom = true
        sv.showsHorizontalScrollIndicator = false
        sv.showsVerticalScrollIndicator = false
        sv.contentInsetAdjustmentBehavior = .never
        sv.backgroundColor = .clear
        sv.delegate = context.coordinator
        let hostView = context.coordinator.host.view!
        hostView.translatesAutoresizingMaskIntoConstraints = false
        hostView.backgroundColor = .clear
        sv.addSubview(hostView)
        NSLayoutConstraint.activate([
            hostView.leadingAnchor.constraint(equalTo: sv.contentLayoutGuide.leadingAnchor),
            hostView.trailingAnchor.constraint(equalTo: sv.contentLayoutGuide.trailingAnchor),
            hostView.topAnchor.constraint(equalTo: sv.contentLayoutGuide.topAnchor),
            hostView.bottomAnchor.constraint(equalTo: sv.contentLayoutGuide.bottomAnchor),
            hostView.widthAnchor.constraint(equalTo: sv.frameLayoutGuide.widthAnchor),
            hostView.heightAnchor.constraint(equalTo: sv.frameLayoutGuide.heightAnchor),
        ])
        let dt = UITapGestureRecognizer(target: context.coordinator,
                                        action: #selector(Coordinator.handleDoubleTap(_:)))
        dt.numberOfTapsRequired = 2
        sv.addGestureRecognizer(dt)
        // one finger dragging DOWN on a photo at rest: back to the gallery
        let down = UIPanGestureRecognizer(target: context.coordinator,
                                          action: #selector(Coordinator.handleExitPan(_:)))
        down.maximumNumberOfTouches = 1
        down.delegate = context.coordinator
        sv.addGestureRecognizer(down)
        return page
    }

    func updateUIView(_ page: PagePin, context: Context) {
        context.coordinator.host.rootView = content
        context.coordinator.onExit = onExit
        page.pin()
    }

    final class Coordinator: NSObject, UIScrollViewDelegate, UIGestureRecognizerDelegate {
        let host: UIHostingController<Content>
        var onExit: (() -> Void)?
        private var exitFired = false
        private var pinchFromRest = false   // the zoom in progress began on a photo at its resting size
        init(host: UIHostingController<Content>) { self.host = host }

        func viewForZooming(in scrollView: UIScrollView) -> UIView? { host.view }

        func scrollViewWillBeginZooming(_ sv: UIScrollView, with view: UIView?) {
            pinchFromRest = sv.zoomScale <= sv.minimumZoomScale * 1.01
        }

        func scrollViewDidZoom(_ sv: UIScrollView) {
            // bounce region below minimum zoom = the gallery-exit affordance, and ONLY for a
            // pinch that began on a photo at its resting size. A firm pinch on a zoomed-in
            // photo also runs through the minimum into the bounce region; that one just
            // zooms the photo out, it does not close the pager.
            if !exitFired, pinchFromRest, sv.zoomScale < sv.minimumZoomScale * 0.85,
               sv.pinchGestureRecognizer?.state == .changed {
                exitFired = true
                onExit?()
            }
        }

        func scrollViewDidEndZooming(_ sv: UIScrollView, with view: UIView?,
                                     atScale scale: CGFloat) {
            exitFired = false
            pinchFromRest = false
            (sv.superview as? PagePin)?.pin()
        }

        @objc func handleDoubleTap(_ gr: UITapGestureRecognizer) {
            guard let sv = gr.view as? UIScrollView else { return }
            if sv.zoomScale > sv.minimumZoomScale * 1.3 {
                sv.setZoomScale(sv.minimumZoomScale, animated: true)
            } else {
                let p = gr.location(in: host.view)
                let w = sv.bounds.width / 2.5, h = sv.bounds.height / 2.5
                sv.zoom(to: CGRect(x: p.x - w / 2, y: p.y - h / 2, width: w, height: h),
                        animated: true)
            }
        }

        // MARK: drag down to leave
        // This object is the delegate of the exit pan only (the scroll view is its own pan's
        // delegate, the double tap has none), so the two methods below are about that pan.

        /// The exit pan starts only on a photo at rest - zoomed in, a one-finger drag pans
        /// the photo - and only when the drag sets off downwards. A sideways drag is the
        /// pager's, an upward one is nobody's.
        func gestureRecognizerShouldBegin(_ gestureRecognizer: UIGestureRecognizer) -> Bool {
            guard let pan = gestureRecognizer as? UIPanGestureRecognizer,
                  let sv = pan.view as? UIScrollView else { return true }
            guard sv.zoomScale <= sv.minimumZoomScale * 1.01, !sv.isZooming else { return false }
            let v = pan.velocity(in: sv)
            return v.y > 0 && v.y > abs(v.x) * 1.5
        }

        /// The pager around this view turns pages with a pan of its own scroll view. It waits
        /// until the exit pan has decided: a downward drag is then the exit pan's alone, and
        /// for any other drag the exit pan fails at once (above) and the page turns as before.
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldBeRequiredToFailBy otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            otherGestureRecognizer is UIPanGestureRecognizer
                && otherGestureRecognizer.view is UIScrollView
                && otherGestureRecognizer.view !== gestureRecognizer.view
        }

        /// The photo follows the finger down and fades. Let go far enough down, or with a
        /// flick, and the pager closes; otherwise the photo goes back to its place.
        @objc func handleExitPan(_ pan: UIPanGestureRecognizer) {
            guard let view = host.view, let sv = pan.view as? UIScrollView else { return }
            let dy = max(pan.translation(in: sv).y, 0)
            switch pan.state {
            case .changed:
                view.transform = CGAffineTransform(translationX: 0, y: dy)
                view.alpha = 1 - min(dy / 500, 0.5)
            case .ended:
                let speed = pan.velocity(in: sv).y
                if dy > 110 || (dy > 40 && speed > 900) {
                    onExit?()
                    // the pager is on its way out and takes this view with it; put the photo
                    // back afterwards in case the view is kept
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                        view.transform = .identity
                        view.alpha = 1
                    }
                } else {
                    settle(view)
                }
            case .cancelled, .failed:
                settle(view)
            default:
                break
            }
        }

        private func settle(_ view: UIView) {
            UIView.animate(withDuration: 0.25, delay: 0,
                           options: [.curveEaseOut, .allowUserInteraction]) {
                view.transform = .identity
                view.alpha = 1
            }
        }
    }
}
