import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    var body: some View {
        TabView {
            LiveView()
                .tabItem { Label("Live", systemImage: "camera.viewfinder") }
            PhotoTestView()
                .tabItem { Label("Photo", systemImage: "photo") }
            BenchView()
                .tabItem { Label("Bench", systemImage: "speedometer") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
    }
}

// MARK: - Projects

/// A credited project: name, what it is, copyright line, licence, repo link.
/// Credits are TEXT only - no logos, icons or glyph tiles, for third parties
/// or for this project (trademark artwork does not belong in an App Store
/// build, and a row of stand-in icons only added clutter).
struct Ack: Identifiable {
    var id: String { repo }
    let name: String
    let blurb: String
    let copyright: String
    let license: String
    let repo: String
}

/// The two projects this app is built ON (the base) - kept apart from the
/// upstream libraries it merely acknowledges. Row names carry no "@ Company"
/// suffix (it read as an affiliation); the copyright line under each row
/// states who holds the rights.
let baseProjects: [Ack] = [
    Ack(name: "YOLO-Master",
        blurb: "The MoE YOLO detector family this app packages.",
        copyright: "Copyright 2026 Tencent", license: "AGPL-3.0",
        repo: "https://github.com/Tencent/YOLO-Master"),
    Ack(name: "YOLO-Master Edge (this app)",
        blurb: "The iOS app and the Core ML export toolchain.",
        copyright: "Copyright 2026 Thomas (Ruiheng Li), HKUST", license: "AGPL-3.0",
        repo: "https://github.com/skywalker-lt/yolo-master-edge"),
]

let acknowledgements: [Ack] = [
    Ack(name: "Ultralytics",
        blurb: "The YOLO training and inference framework YOLO-Master builds on.",
        copyright: "Copyright 2025 Ultralytics", license: "AGPL-3.0",
        repo: "https://github.com/ultralytics/ultralytics"),
    Ack(name: "Core ML",
        blurb: "Model conversion and the on-device inference runtime.",
        copyright: "Copyright 2020 to 2023 Apple Inc.", license: "BSD-3-Clause",
        repo: "https://github.com/apple/coremltools"),
]

/// Load an About-page logo from the bundled "ack" folder; nil until it is bundled
/// (xcodegen picks up Resources/ack), in which case the caller shows a placeholder.
func ackLogo(_ name: String, _ ext: String) -> UIImage? {
    let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "ack")
        ?? Bundle.main.url(forResource: name, withExtension: ext)
    return url.flatMap { UIImage(contentsOfFile: $0.path) }
}

let iosLicenseNotice = """
================================================================
YOLO-Master iOS App
Copyright (C) 2026 (Thomas) RUIHENG LI
(Author affiliated with The Hong Kong University of Science and
Technology)

YOLO-Master iOS App runs YOLO-Master models on-device using
Apple's Core ML. It shares the YOLOMasterKit inference path with
the macOS tools.

This project is licensed under the GNU Affero General Public
License, version 3 (AGPL-3.0). See
<https://www.gnu.org/licenses/> for the full terms.

It builds upon and incorporates the following third-party
software:

  * YOLO-Master - Copyright (C) 2026 Tencent. Licensed under the
    AGPL-3.0. Based on ultralytics.

  * ultralytics - Copyright (c) 2025 Ultralytics. Licensed under
    the AGPL-3.0.

  * coremltools - Copyright (c) 2020-2023, Apple Inc. Licensed
    under the BSD-3-Clause License.

Core ML model weights exported and bundled with this app are
licensed under the AGPL-3.0.

USE NOTICE (not a license term): The authors release this app for
research and personal experience only, and neither intend nor
support commercial use. This is a statement of intent and does not
impose additional restrictions beyond the AGPL-3.0.

NO WARRANTY: This program is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
================================================================
"""

// MARK: - Settings

/// App info, compute safety, history reset, and custom model import.
struct SettingsView: View {
    // Links surfaced in the About card and the Privacy section.
    static let paperURL = "https://arxiv.org/pdf/2512.23273"
    static let modelRepoURL = "https://github.com/Tencent/YOLO-Master"
    static let appRepoURL = "https://github.com/skywalker-lt/yolo-master-edge"
    /// The hosted privacy policy (PRIVACY.md at the repo root, default branch).
    /// App Review wants this same URL in App Store Connect AND a link in the app.
    static let privacyURL = "https://github.com/skywalker-lt/yolo-master-edge/blob/main/PRIVACY.md"

    @AppStorage("allowCPU") private var allowCPU = false
    @ObservedObject private var history = BenchHistory.shared
    @State private var customModels: [BundledModel] = []
    @State private var importing = false
    @State private var confirmErase = false
    @State private var importError: String?
    @State private var aboutOpen = false
    @State private var licensesOpen = false
    @State private var privacyOpen = false

    @State private var importingFolder = false   // the importer is open in folder mode
    @State private var checkingImport = false    // picked models are being copied and checked

    // model gallery: every model in the app, with its size on disk
    @State private var galleryOpen = false
    @State private var galleryModels: [BundledModel] = []
    @State private var gallerySizes: [String: Int64] = [:]
    @State private var openModels: Set<String> = []   // ids of the models whose attributes are shown

    static let modelExtensions = ["mlpackage", "mlmodelc", "mlmodel"]

    /// Content types the importer offers in document mode.
    ///
    /// A `.mlpackage` and a `.mlmodelc` are DIRECTORIES. iOS presents a directory as one
    /// selectable document only when a package type is declared for its extension, which
    /// `project.yml` now does (UTImportedTypeDeclarations, the same identifiers the Mac app
    /// uses). A type built from the bare file extension describes a FILE and never matches
    /// a directory: with only those, Files showed every package as a folder that could be
    /// opened but not chosen, so no model could be imported. `.package` admits a package
    /// iOS knows under another identifier; whatever comes back is checked by extension.
    private var modelTypes: [UTType] {
        let declared = ["com.apple.coreml.mlpackage", "com.apple.coreml.mlmodelc", "com.apple.coreml.model"]
            .map { UTType(importedAs: $0) }
        let byExtension = Self.modelExtensions.compactMap { UTType(filenameExtension: $0) }
        return declared + byExtension + [.package]
    }

    var body: some View {
        NavigationStack {
            Form {
                aboutSection
                licensesSection
                privacySection
                computeSection
                modelGallerySection
                customModelsSection
                eraseSection
            }
            .navigationTitle("Settings")
            // one importer, two modes: model documents, or (the fallback for a package that
            // Files still shows as a plain folder) a folder. Mixing `.folder` with document
            // types in one picker is unreliable on iOS, hence the separate mode.
            .fileImporter(isPresented: $importing,
                          allowedContentTypes: importingFolder ? [.folder] : modelTypes,
                          allowsMultipleSelection: !importingFolder) { importModels($0) }
            .alert("Erase all history?", isPresented: $confirmErase) {
                Button("Erase", role: .destructive) { history.clear() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This permanently deletes all \(history.runs.count) saved benchmark runs.")
            }
            .alert("Import failed", isPresented: Binding(
                get: { importError != nil }, set: { if !$0 { importError = nil } })) {
                Button("OK") { importError = nil }
            } message: {
                Text(importError ?? "")
            }
            .onAppear {
                customModels = BundledModel.customModels()
                loadGallery()
            }
        }
    }

    // MARK: About (expandable)

    private var aboutSection: some View {
        Section {
            DisclosureGroup(isExpanded: $aboutOpen) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("YOLO-Master extends the real-time YOLO detector with Mixture-of-Experts (MoE) routing. Instead of one dense network, lightweight expert branches specialize on different feature patterns, and a learned router activates only the most relevant experts for each image.")
                        .font(.footnote).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("That adds capacity where it matters, with clear gains on small and crowded objects, while keeping inference light enough to run in real time. On iPhone the model is compiled to Core ML and runs on the Neural Engine, GPU, or CPU.")
                        .font(.footnote).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Divider()
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 8) { links }
                            .frame(maxWidth: .infinity, alignment: .center)
                        VStack(spacing: 8) { links }
                            .frame(maxWidth: .infinity, alignment: .center)
                    }
                    .padding(.top, 2)
                }
                .padding(.top, 8)
            } label: {
                header
            }
        }
    }

    @ViewBuilder private var links: some View {
        linkChip("Paper", "doc.richtext", Self.paperURL)
        linkChip("Model Repo", "shippingbox", Self.modelRepoURL)
        linkChip("App Repo", "chevron.left.forwardslash.chevron.right", Self.appRepoURL)
    }

    private func linkChip(_ title: String, _ icon: String, _ urlString: String) -> some View {
        Group {
            if let url = URL(string: urlString) {
                Link(destination: url) {
                    HStack(spacing: 5) {
                        Image(systemName: icon).font(.caption2)
                        Text(title).font(.caption.weight(.semibold)).lineLimit(1)
                    }
                    .fixedSize()
                    .padding(.horizontal, 12).padding(.vertical, 7)
                    .background(Color.accentColor.opacity(0.14), in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
    }

    /// "Version 1.1.0 (2)" read from the bundle, so the About card always shows
    /// what was actually built (project.yml is the single source of truth).
    private var versionText: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "-"
        guard let build = info?["CFBundleVersion"] as? String, !build.isEmpty else {
            return "Version \(short)"
        }
        return "Version \(short) (\(build))"
    }

    private var header: some View {
        HStack(spacing: 14) {
            appMark
            VStack(alignment: .leading, spacing: 3) {
                Text("YOLO-Master").font(.title3.bold())
                Text(versionText).font(.caption).foregroundStyle(.secondary)
                Text("On-device MoE object detection with Core ML.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 4)
    }

    private var appMark: some View {
        Group {
            if let img = ackLogo("appmark", "png") {
                Image(uiImage: img).resizable().scaledToFit()
            } else {
                RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.quaternary)
                    .overlay(Image(systemName: "cube.transparent").font(.title).foregroundStyle(.secondary))
            }
        }
        .frame(width: 56, height: 56)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
            .strokeBorder(Color.primary.opacity(0.1), lineWidth: 1))
    }

    // MARK: Licenses & Acknowledgements

    private var licensesSection: some View {
        Section {
            DisclosureGroup(isExpanded: $licensesOpen) {
                // Spacing carries the structure now that the rows have no icons:
                // wide gaps between blocks (22), close lines inside a block.
                VStack(alignment: .leading, spacing: 22) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("YOLO-Master and Ultralytics are licensed under AGPL-3.0; coremltools under BSD-3-Clause. This app and the Core ML weights it bundles are distributed under AGPL-3.0, and this app complies with those terms.")
                            .font(.footnote).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)

                        Text("The authors release this app for research and personal experience only, and neither intend nor support any commercial use.")
                            .font(.footnote.bold())
                            .fixedSize(horizontal: false, vertical: true)

                        Text("This is a statement of the authors' intent. It does not add restrictions to, or otherwise limit, the rights granted under the AGPL-3.0.")
                            .font(.caption2).foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Divider()
                    ackGroup("Built on", baseProjects)

                    Divider()
                    ackGroup("Acknowledgements", acknowledgements)

                    // tertiary GROUPED background: one step off the form cell in
                    // both appearances (the plain secondary background is the
                    // cell's own color in dark mode, so the box vanished there)
                    Text(iosLicenseNotice)
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .background(RoundedRectangle(cornerRadius: 8)
                            .fill(Color(.tertiarySystemGroupedBackground)))
                }
                .padding(.top, 12)
                .padding(.bottom, 8)
            } label: {
                Label("Licenses & Acknowledgements", systemImage: "doc.text")
            }
        }
    }

    private func groupLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.caption2.weight(.bold))
            .foregroundStyle(.tertiary)
    }

    /// One titled group of credits. The label sits closer to its first row
    /// (14) than the rows sit to each other (20), so it reads as their heading.
    private func ackGroup(_ title: String, _ items: [Ack]) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            groupLabel(title)
            VStack(alignment: .leading, spacing: 20) {
                ForEach(items) { a in ackRow(a) }
            }
        }
    }

    /// Text-only credit, full width: name and what it is, then the licence
    /// chip beside the copyright line, then the repo link. No icons or logos.
    private func ackRow(_ a: Ack) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            VStack(alignment: .leading, spacing: 4) {
                Text(a.name).font(.callout.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                Text(a.blurb).font(.footnote).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(a.license)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Color.accentColor.opacity(0.14), in: Capsule())
                    .fixedSize()
                Text(a.copyright).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let url = URL(string: a.repo) {
                Link(destination: url) {
                    // manual glyph + text, not a Label: inside a Form row a
                    // Label reserves the list's icon column, which left a
                    // wide gap between the glyph and the link text
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Image(systemName: "link")
                        Text(a.repo.replacingOccurrences(of: "https://", with: ""))
                    }
                    .font(.caption)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Privacy & Security

    private var privacySection: some View {
        Section {
            DisclosureGroup(isExpanded: $privacyOpen) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("YOLO-Master runs entirely on your device.")
                        .font(.footnote.weight(.semibold))

                    VStack(alignment: .leading, spacing: 8) {
                        accessRow("camera", "Camera",
                                  "Live on-device detection. Frames are processed in memory and never stored or transmitted.")
                        accessRow("photo.on.rectangle", "Photo Library",
                                  "Reads only the images you pick for the Photo tab, and saves detection frames you explicitly export.")
                    }

                    Text("All inference runs locally on the Neural Engine, GPU, or CPU. No images, results, or usage data ever leave your phone. We do not collect, store, or transmit any personal data. The only time we receive anything from you is if you choose to contact us or report an issue yourself.")
                        .font(.footnote).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 6)
            } label: {
                Label("Privacy & Security", systemImage: "lock.shield")
            }
            // always visible (not inside the disclosure): App Review wants the
            // policy reachable from within the app, not only on the store page
            if let url = URL(string: Self.privacyURL) {
                Link(destination: url) {
                    HStack {
                        Label("Privacy Policy", systemImage: "hand.raised")
                        Spacer(minLength: 0)
                        Image(systemName: "arrow.up.right")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
            }
        }
    }

    private func accessRow(_ icon: String, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon).foregroundStyle(.secondary).frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.caption.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: Compute / Custom models / Erase

    private var computeSection: some View {
        Section {
            Toggle("Allow CPU inference", isOn: $allowCPU)
        } header: {
            Text("Compute")
        } footer: {
            Text("CPU-only inference is the slowest and hottest path, and some models do not run reliably on it in Live and Photo, so it is off by default. Turn this on to add CPU to the Live and Photo compute pickers. The Bench tab always measures CPU.")
        }
    }

    // MARK: Model gallery

    /// Every model in the app, bundled and imported. Collapsed by default; each model opens
    /// to its type and attributes. They come from the metadata Core ML's compiler stores with
    /// the model (`ModelFacts`), so nothing is loaded to show them.
    private var modelGallerySection: some View {
        Section {
            DisclosureGroup(isExpanded: $galleryOpen) {
                ForEach(galleryModels) { m in
                    DisclosureGroup(isExpanded: Binding(
                        get: { openModels.contains(m.id) },
                        set: { open in
                            if open { openModels.insert(m.id) } else { openModels.remove(m.id) }
                        })) {
                        modelAttributes(m)
                    } label: {
                        modelRowLabel(m)
                    }
                }
            } label: {
                Label("Model gallery", systemImage: "square.grid.2x2")
            }
        } header: {
            Text("Models")
        } footer: {
            Text("\(galleryModels.count) model\(galleryModels.count == 1 ? "" : "s") in the app. Open one to see what it does and how it was exported.")
        }
    }

    /// The model list and each model's size on disk, read off the main thread.
    private func loadGallery() {
        Task.detached {
            let found = BundledModel.discover()
            var sizes: [String: Int64] = [:]
            for m in found { sizes[m.id] = m.diskSize }
            let measured = sizes
            await MainActor.run {
                galleryModels = found
                gallerySizes = measured
            }
        }
    }

    /// A gallery row: the type icon, the model's name, and a one-line summary under it.
    private func modelRowLabel(_ m: BundledModel) -> some View {
        let facts = m.facts
        var parts = [facts.kind.label]
        if let precision = precisionName(facts) { parts.append(precision) }
        if let size = gallerySizes[m.id], size > 0 {
            parts.append(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))
        }
        let summary = parts.joined(separator: " · ")
        return HStack(spacing: 12) {
            Image(systemName: facts.kind.icon)
                .foregroundStyle(.secondary)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 3) {
                Text(m.fullName).font(.callout).lineLimit(1)
                Text(summary).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(.vertical, 2)
    }

    /// The attributes of one model as name / value lines. Text only, values wrap.
    private func modelAttributes(_ m: BundledModel) -> some View {
        let rows = attributeRows(m)
        return VStack(alignment: .leading, spacing: 10) {
            ForEach(rows.indices, id: \.self) { i in
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(rows[i].name).font(.caption).foregroundStyle(.secondary)
                        .frame(width: 84, alignment: .leading)
                    Text(rows[i].value).font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(.vertical, 6)
    }

    /// "FP16", "W8A16" from the exporter's stamp, else what Core ML says the weights are stored as.
    private func precisionName(_ f: ModelFacts) -> String? {
        guard let stamp = f.precision, !stamp.isEmpty else { return f.storagePrecision }
        return stamp.uppercased()
    }

    private func attributeRows(_ m: BundledModel) -> [(name: String, value: String)] {
        let f = m.facts
        var rows: [(name: String, value: String)] = []
        rows.append(("Type", f.kindIsGuess
                     ? f.kind.label + " (going by the file name: the model has not been compiled yet)"
                     : f.kind.label))
        if f.keypoints == 133 {
            rows.append(("Keypoints", "133 per person: 17 body, 6 feet, 68 face, 42 hands"))
        } else if f.keypoints > 0 {
            rows.append(("Keypoints", "\(f.keypoints) per person"))
        }
        if f.topDown {
            rows.append(("Pipeline", "Top-down: a detection model finds the people, this model runs once per person"))
        } else if !f.kindIsGuess {
            rows.append(("Pipeline", f.end2end ? "One pass, NMS-free output" : "One pass, NMS in the app"))
        }
        if f.classNames.count == 1 {
            rows.append(("Classes", f.classNames[0]))
        } else if f.classNames.count > 1 {
            let shown = f.classNames.prefix(6).joined(separator: ", ")
            let more = f.classNames.count - 6
            let tail = more > 0 ? " and \(more) more" : ""
            rows.append(("Classes", "\(f.classNames.count): " + shown + tail))
        }
        for p in f.inputs { rows.append(("Input", portText(p))) }
        for p in f.outputs { rows.append(("Output", portText(p))) }
        if let name = precisionName(f) {
            var text = name
            switch name {
            case "FP16": text = "FP16: 16-bit float weights and activations"
            case "W8A16": text = "W8A16: 8-bit weights, 16-bit float activations"
            case "W8A8": text = "W8A8: 8-bit weights and activations"
            default: break
            }
            if let stored = f.storagePrecision, stored != name { text += ". Stored as " + stored }
            rows.append(("Precision", text))
        }
        if let format = f.format {
            var text = format
            if let spec = f.specVersion { text += ", specification \(spec)" }
            if let os = f.minimumIOS { text += ", iOS " + os + " or later" }
            rows.append(("Format", text))
        }
        if let n = f.operations, n > 0 { rows.append(("Operations", "\(n)")) }
        if let size = gallerySizes[m.id], size > 0 {
            rows.append(("Size", ByteCountFormatter.string(fromByteCount: size, countStyle: .file)))
        }
        rows.append(("Origin", m.isImported ? "Imported in Settings" : "Bundled with the app"))
        if let cut = f.metadata["kpt_score"], !cut.isEmpty {
            rows.append(("Kpt score", "drawn from " + cut))
        }
        if let source = f.metadata["source"], !source.isEmpty { rows.append(("Source", source)) }
        if let about = f.summary { rows.append(("About", about)) }
        if let author = f.author { rows.append(("Author", author)) }
        if let license = f.license { rows.append(("License", license)) }
        if let version = f.version { rows.append(("Version", version)) }
        rows.append(("File", m.id))
        return rows
    }

    /// "image  1 x 3 x 640 x 640  Float16"
    private func portText(_ p: ModelPort) -> String {
        var parts = [p.name, p.shape.isEmpty ? "shape not declared" : p.shape]
        if !p.dataType.isEmpty { parts.append(p.dataType) }
        return parts.joined(separator: "   ")
    }

    private var customModelsSection: some View {
        Section {
            Button {
                importingFolder = false
                importing = true
            } label: {
                Label("Load custom Core ML model", systemImage: "plus.circle")
            }
            .disabled(checkingImport)
            Button {
                importingFolder = true
                importing = true
            } label: {
                Label("Load models from a folder", systemImage: "folder.badge.plus")
            }
            .disabled(checkingImport)
            if checkingImport {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Checking the model...")
                }
                .font(.callout).foregroundStyle(.secondary)
            }
            ForEach(customModels) { m in
                HStack {
                    Image(systemName: "cube.box").foregroundStyle(.secondary)
                    Text(m.fullName).lineLimit(1)
                    Spacer()
                    Button(role: .destructive) {
                        BundledModel.deleteCustom(m)
                        customModels = BundledModel.customModels()
                        loadGallery()
                    } label: {
                        Image(systemName: "trash").foregroundStyle(.red)
                    }
                    .buttonStyle(.borderless)
                }
            }
        } header: {
            Text("Custom models")
        } footer: {
            Text("Import a .mlpackage, .mlmodelc, or .mlmodel. If Files shows a model package as a folder, use Load models from a folder: open the package, or the folder that holds your models, and tap Open. Imported models appear in the Live, Photo, and Bench pickers the next time you open that tab. Compilation happens on first load and may take a moment.")
        }
    }

    private var eraseSection: some View {
        Section {
            Button(role: .destructive) {
                confirmErase = true
            } label: {
                Label("Erase all benchmark history", systemImage: "trash")
            }
            .disabled(history.runs.isEmpty)
        } footer: {
            Text("\(history.runs.count) saved run\(history.runs.count == 1 ? "" : "s").")
        }
    }

    // MARK: Import

    private func importModels(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let e):
            importError = e.localizedDescription
        case .success(let urls):
            // copying, compiling and loading each model takes seconds: off the main thread,
            // behind the "Checking the model" row
            checkingImport = true
            Task.detached(priority: .userInitiated) {
                let problems = SettingsView.importAll(urls)
                await MainActor.run {
                    checkingImport = false
                    if !problems.isEmpty { importError = problems.joined(separator: "\n\n") }
                    customModels = BundledModel.customModels()
                    loadGallery()
                }
            }
        }
    }

    /// Copy every picked model into the app and keep only those the app can run.
    /// Returns one message per item that was not imported.
    private static func importAll(_ urls: [URL]) -> [String] {
        let dir = BundledModel.customModelsDir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var problems: [String] = []
        for picked in urls {
            let scoped = picked.startAccessingSecurityScopedResource()
            defer { if scoped { picked.stopAccessingSecurityScopedResource() } }
            let items = modelItems(at: picked)
            if items.isEmpty {
                problems.append("\(picked.lastPathComponent) is not a Core ML model and contains none (.mlpackage, .mlmodelc, .mlmodel).")
                continue
            }
            for src in items {
                if let problem = copyModel(src, into: dir) {
                    problems.append(problem)
                    continue
                }
                // A model Core ML cannot compile, or one that is not a detector this app can
                // decode, is refused HERE with the reason. It used to import without a word
                // and then simply never produce a detection.
                let model = BundledModel(id: src.deletingPathExtension().lastPathComponent,
                                         url: dir.appendingPathComponent(src.lastPathComponent))
                if let why = model.loadProblem() {
                    BundledModel.deleteCustom(model)
                    problems.append("\(src.lastPathComponent) was not imported. \(why)")
                }
            }
        }
        return problems
    }

    /// The models a picked URL stands for: itself when it is a model; otherwise, for a plain
    /// folder chosen in folder mode, the models directly inside it.
    private static func modelItems(at url: URL) -> [URL] {
        if Self.modelExtensions.contains(url.pathExtension.lowercased()) { return [url] }
        let inside = (try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil)) ?? []
        return inside.filter { Self.modelExtensions.contains($0.pathExtension.lowercased()) }
    }

    /// Copy one model into the app's CustomModels folder. Returns a message on failure.
    /// The read is coordinated, so an item kept in iCloud Drive is downloaded first and a
    /// package is copied as one consistent directory.
    private static func copyModel(_ src: URL, into dir: URL) -> String? {
        let dest = dir.appendingPathComponent(src.lastPathComponent)
        var coordinationError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: src, options: [],
                                       error: &coordinationError) { readable in
            // replacing a model of the same name: remove the old copy AND the compiled
            // cache it produced, or the app would keep running the old model
            BundledModel.deleteCustom(BundledModel(
                id: dest.deletingPathExtension().lastPathComponent, url: dest))
            do {
                try FileManager.default.copyItem(at: readable, to: dest)
            } catch {
                copyError = error
            }
        }
        if let failure = (coordinationError as Error?) ?? copyError {
            return "\(src.lastPathComponent): \(failure.localizedDescription)"
        }
        return nil
    }
}
