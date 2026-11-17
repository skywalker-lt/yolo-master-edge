import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject private var localizer = Localizer.shared   // the tab titles follow the language

    var body: some View {
        // `screenEdgeAware`: the three tabs with a bar floating at the top are told how far
        // to keep it from the screen's edge on an iPhone Duo (0 on every other iPhone).
        TabView {
            LiveView()
                .screenEdgeAware()
                .tabItem { Label(tr("Live"), systemImage: "camera.viewfinder") }
            PhotoTestView()
                .screenEdgeAware()
                .tabItem { Label(tr("Photo"), systemImage: "photo") }
            BenchView()
                .screenEdgeAware()
                .tabItem { Label(tr("Bench"), systemImage: "speedometer") }
            SettingsView()
                .tabItem { Label(tr("Settings"), systemImage: "gearshape") }
        }
    }
}

// MARK: - Projects

/// A credited project: name, what it is, copyright line, licence, repo link.
/// Credits are TEXT only - no logos, icons or glyph tiles, for third parties
/// or for this project (trademark artwork does not belong in an App Store
/// build, and a row of stand-in icons only added clutter).
/// `blurb` is shown through `tr` (`ackRow`), and so is a name with something to translate
/// in it; project names, copyright lines and licence names stay as they are in every language.
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
        blurb: trKey("The MoE YOLO detector family this app accelerates."),
        copyright: "Copyright 2026 Tencent", license: "AGPL-3.0",
        repo: "https://github.com/Tencent/YOLO-Master"),
    Ack(name: trKey("YOLO-Master Edge (this app)"),
        blurb: trKey("The source code of this app and the Core ML export toolchain."),
        copyright: "Copyright 2026 Thomas (Ruiheng Li), HKUST", license: "AGPL-3.0",
        repo: "https://github.com/skywalker-lt/yolo-master-edge"),
]

let acknowledgements: [Ack] = [
    Ack(name: "Ultralytics",
        blurb: trKey("The YOLO training and inference framework YOLO-Master builds on."),
        copyright: "Copyright 2025 Ultralytics", license: "AGPL-3.0",
        repo: "https://github.com/ultralytics/ultralytics"),
    Ack(name: "MMPose",
        blurb: trKey("The RTMW whole-body pose model bundled with this app."),
        copyright: "Copyright 2018 to 2020 Open-MMLab", license: "Apache-2.0",
        repo: "https://github.com/open-mmlab/mmpose"),
    Ack(name: "Core ML",
        blurb: trKey("Model conversion and the on-device inference runtime."),
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

/// The notice in the box at the end of Licenses & Acknowledgements.
///
/// Every paragraph is ONE line of text and the view wraps it to the width it has. It used
/// to be broken by hand at 64 columns. The box holds about 44 characters of this font on
/// an iPhone, so every line broke a second time in its middle, and the two rules of 64
/// "=" ran onto a second line (reported with 1016.png, 2026-11-16). A backslash at the
/// end of a source line joins it to the next one, which keeps the source readable
/// without putting a line break into the text. Do not break paragraphs by hand again.
/// The words are those of the hand-broken notice; only its line breaks, the indentation
/// of the list and the two rules went. The app is named YOLO-Master Edge in it since
/// 2026-11-16 (it was "YOLO-Master iOS App"); "YOLO-Master" alone is the upstream project
/// and its models, and stays where it means those.
/// The notice is shown in ENGLISH in every language of the app: it is the notice itself,
/// not a description of it. The three paragraphs above it in Settings are translated.
let iosLicenseNotice = """
YOLO-Master Edge iOS App
Copyright (C) 2026 (Thomas) RUIHENG LI
(Author affiliated with The Hong Kong University of Science and Technology)

YOLO-Master Edge iOS App runs YOLO-Master models on-device using Apple's Core ML. \
It shares the YOLOMasterKit inference path with the macOS tools.

The source code of this app is licensed under the GNU Affero General Public License, \
version 3 (AGPL-3.0). See <https://www.gnu.org/licenses/> for the full terms.

It builds upon the following third-party projects:

* YOLO-Master - Copyright (C) 2026 Tencent. Licensed under the AGPL-3.0. \
Based on ultralytics.

* ultralytics - Copyright (c) 2025 Ultralytics. Licensed under the AGPL-3.0.

* MMPose - Copyright 2018-2020 Open-MMLab. Licensed under the Apache License 2.0. \
Source of the RTMW whole-body pose model. <https://github.com/open-mmlab/mmpose>

* coremltools - Copyright (c) 2020-2023, Apple Inc. Licensed under the \
BSD-3-Clause License.

The models bundled with this app come from YOLO-Master, ultralytics and MMPose, \
and remain under the terms of those projects.

USE NOTICE: This app, as distributed with the models it bundles, is non-commercial. \
The authors release it for research and personal experience only, and neither intend \
nor support commercial use. This notice does not change the AGPL-3.0 terms of the \
source code.

NO WARRANTY: This program is distributed in the hope that it will be useful, but \
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or \
FITNESS FOR A PARTICULAR PURPOSE.
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
    /// The form for a new issue in the app's repository (GitHub asks for a sign-in first).
    static let issueURL = "https://github.com/skywalker-lt/yolo-master-edge/issues/new"

    @AppStorage("allowCPU") private var allowCPU = false
    @ObservedObject private var history = BenchHistory.shared
    @ObservedObject private var localizer = Localizer.shared   // the language, and its picker
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
                languageSection
                computeSection
                modelGallerySection
                customModelsSection
                eraseSection
                reportSection
            }
            // a little more room under the last row (Report an issue and its footer), which
            // otherwise ends close above the tab bar when the list is scrolled to its end
            .contentMargins(.bottom, 28, for: .scrollContent)
            .navigationTitle(tr("Settings"))
            // one importer, two modes: model documents, or (the fallback for a package that
            // Files still shows as a plain folder) a folder. Mixing `.folder` with document
            // types in one picker is unreliable on iOS, hence the separate mode.
            .fileImporter(isPresented: $importing,
                          allowedContentTypes: importingFolder ? [.folder] : modelTypes,
                          allowsMultipleSelection: !importingFolder) { importModels($0) }
            .alert(tr("Erase all history?"), isPresented: $confirmErase) {
                Button(tr("Erase"), role: .destructive) { history.clear() }
                Button(tr("Cancel"), role: .cancel) {}
            } message: {
                Text(tr("This permanently deletes all {0} saved benchmark runs.", history.runs.count))
            }
            .alert(tr("Import failed"), isPresented: Binding(
                get: { importError != nil }, set: { if !$0 { importError = nil } })) {
                Button(tr("OK")) { importError = nil }
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
                    Text(tr("YOLO-Master extends the real-time YOLO detector with Mixture-of-Experts (MoE) routing. Instead of one dense network, lightweight expert branches specialize on different feature patterns, and a learned router activates only the most relevant experts for each image."))
                        .font(.footnote).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(tr("That adds capacity where it matters, with clear gains on small and crowded objects, while keeping inference light enough to run in real time. On {0} the model is compiled to Core ML and runs on the Neural Engine, GPU, or CPU.", deviceName))
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
        linkChip(tr("Paper"), "doc.richtext", Self.paperURL)
        linkChip(tr("Model Repo"), "shippingbox", Self.modelRepoURL)
        linkChip(tr("App Repo"), "chevron.left.forwardslash.chevron.right", Self.appRepoURL)
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

    /// "Version 1.2.1 (1)" read from the bundle, so the About card always shows
    /// what was actually built (project.yml is the single source of truth).
    private var versionText: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "-"
        guard let build = info?["CFBundleVersion"] as? String, !build.isEmpty else {
            return tr("Version {0}", short)
        }
        return tr("Version {0} ({1})", short, build)
    }

    private var header: some View {
        HStack(spacing: 14) {
            appMark
            VStack(alignment: .leading, spacing: 3) {
                Text("YOLO-Master Edge").font(.title3.bold())   // the APP's name; "YOLO-Master" alone is the upstream project
                Text(versionText).font(.caption).foregroundStyle(.secondary)
                Text(tr("On-device MoE multi-task detection with Core ML."))
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
                        // Only the SOURCE CODE is AGPL-3.0. The bundled models are not
                        // relicensed here: each stays under its own project's terms
                        // (the RTMW whole-body model comes from MMPose, Apache-2.0).
                        Text(tr("The source code of this app is licensed under AGPL-3.0. The models it bundles come from the projects below and remain under their terms: YOLO-Master and Ultralytics under AGPL-3.0, MMPose under Apache-2.0. coremltools is licensed under BSD-3-Clause."))
                            .font(.footnote).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)

                        Text(tr("This app is non-commercial. The authors release it for research and personal experience only, and neither intend nor support any commercial use."))
                            .font(.footnote.bold())
                            .fixedSize(horizontal: false, vertical: true)

                        Text(tr("This notice is about the app as distributed, with the models it bundles. It does not change the AGPL-3.0 terms of the source code."))
                            .font(.caption2).foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Divider()
                    ackGroup(tr("Built on"), baseProjects)

                    Divider()
                    ackGroup(tr("Acknowledgements"), acknowledgements)

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
                Label(tr("Licenses & Acknowledgements"), systemImage: "doc.text")
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
    ///
    /// The link MUST keep its `.buttonStyle(.borderless)`. All five credits are in ONE Form
    /// row (the disclosure's content), and a Form row whose buttons have the default style
    /// is itself the button: a tap anywhere in it - on any link, or on plain text - went
    /// to the row's first link, so every credit opened github.com/Tencent/YOLO-Master
    /// (reported 2026-11-16). With an explicit style each link takes only its own taps and
    /// the row takes none; the About card's chips have always had one, which is why they
    /// worked.
    private func ackRow(_ a: Ack) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            VStack(alignment: .leading, spacing: 4) {
                Text(tr(a.name)).font(.callout.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                Text(tr(a.blurb)).font(.footnote).foregroundStyle(.secondary)
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
                .buttonStyle(.borderless)   // see above: without it every credit opens the first one's repo
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Privacy & Security

    private var privacySection: some View {
        Section {
            DisclosureGroup(isExpanded: $privacyOpen) {
                VStack(alignment: .leading, spacing: 12) {
                    Text(tr("YOLO-Master Edge runs entirely on your device."))
                        .font(.footnote.weight(.semibold))

                    VStack(alignment: .leading, spacing: 8) {
                        accessRow("camera", tr("Camera"),
                                  tr("Live on-device detection. Frames are processed in memory and never stored or transmitted."))
                        accessRow("photo.on.rectangle", tr("Photo Library"),
                                  tr("Reads only the images you pick for the Photo tab, and saves detection frames you explicitly export."))
                    }
                }
                .padding(.top, 6)
            } label: {
                Label(tr("Privacy & Security"), systemImage: "lock.shield")
            }
            // always visible (not inside the disclosure): App Review wants the
            // policy reachable from within the app, not only on the store page
            if let url = URL(string: Self.privacyURL) {
                Link(destination: url) {
                    HStack {
                        Label(tr("Privacy Policy"), systemImage: "hand.raised")
                        Spacer(minLength: 0)
                        // no colour of its own: it takes the row's, like the title beside it
                        // (it was tertiary, and looked dimmer than the text)
                        Image(systemName: "arrow.up.right")
                            .font(.caption)
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

    // MARK: Language

    /// The language of the app's own text (`Localizer`): System, which follows the device,
    /// or one of the three the app has. Each language is listed by its own name, so it can
    /// be found from any other. The change shows at once, in every tab.
    private var languageSection: some View {
        Section {
            Picker(selection: Binding(get: { localizer.choice },
                                      set: { localizer.select($0) })) {
                ForEach(AppLanguage.allCases) { language in
                    Text(language.displayName).tag(language)
                }
            } label: {
                Label(tr("Language"), systemImage: "globe")
            }
        } footer: {
            Text(tr("System uses the language of this {0}. Screens that the system provides, such as the photo picker, stay in the language of the {0}.", deviceName))
        }
    }

    // MARK: Compute / Custom models / Erase

    private var computeSection: some View {
        Section {
            Toggle(tr("Allow CPU inference"), isOn: $allowCPU)
        } header: {
            Text(tr("Compute"))
        } footer: {
            Text(tr("Inference on CPU is slow and heats up the device rapidly. Some models may not run reliably on CPU."))
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
                Label(tr("Model gallery"), systemImage: "square.grid.2x2")
            }
        } header: {
            Text(tr("Models"))
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
            ModelIcon.image(facts.kind)
                .resizable().scaledToFit()
                .frame(width: 22, height: 22)
                .foregroundStyle(.secondary)
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
        // The names and the app's own wording are translated. What the MODEL says about
        // itself - class names, port names, its description, author, licence - is shown as
        // the model has it.
        rows.append((tr("Type"), f.kindIsGuess
                     ? tr("{0} (going by the file name: the model has not been compiled yet)", f.kind.label)
                     : f.kind.label))
        if f.keypoints == 133 {
            rows.append((tr("Keypoints"), tr("133 per person: 17 body, 6 feet, 68 face, 42 hands")))
        } else if f.keypoints > 0 {
            rows.append((tr("Keypoints"), tr("{0} per person", f.keypoints)))
        }
        if f.topDown {
            rows.append((tr("Pipeline"), tr("Top-down: a detection model finds the people, this model runs once per person")))
        } else if !f.kindIsGuess {
            rows.append((tr("Pipeline"), f.end2end ? tr("One pass, NMS-free output") : tr("One pass, NMS in the app")))
        }
        if f.classNames.count == 1 {
            rows.append((tr("Classes"), f.classNames[0]))
        } else if f.classNames.count > 1 {
            let shown = f.classNames.prefix(6).joined(separator: ", ")
            let more = f.classNames.count - 6
            rows.append((tr("Classes"), more > 0
                         ? tr("{0}: {1} and {2} more", f.classNames.count, shown, more)
                         : "\(f.classNames.count): " + shown))
        }
        for p in f.inputs { rows.append((tr("Input"), portText(p))) }
        for p in f.outputs { rows.append((tr("Output"), portText(p))) }
        if let name = precisionName(f) {
            var text = name
            switch name {
            case "FP16": text = tr("FP16: 16-bit float weights and activations")
            case "W8A16": text = tr("W8A16: 8-bit weights, 16-bit float activations")
            case "W8A8": text = tr("W8A8: 8-bit weights and activations")
            default: break
            }
            if let stored = f.storagePrecision, stored != name { text = tr("{0}. Stored as {1}", text, stored) }
            rows.append((tr("Precision"), text))
        }
        if let format = f.format {
            var text = format
            if let spec = f.specVersion { text = tr("{0}, specification {1}", text, spec) }
            if let os = f.minimumIOS { text = tr("{0}, iOS {1} or later", text, os) }
            rows.append((tr("Format"), text))
        }
        if let n = f.operations, n > 0 { rows.append((tr("Operations"), "\(n)")) }
        if let size = gallerySizes[m.id], size > 0 {
            rows.append((tr("Size"), ByteCountFormatter.string(fromByteCount: size, countStyle: .file)))
        }
        rows.append((tr("Origin"), m.isImported ? tr("Imported in Settings") : tr("Bundled with the app")))
        if let cut = f.metadata["kpt_score"], !cut.isEmpty {
            rows.append((tr("Kpt score"), tr("drawn from {0}", cut)))
        }
        if let source = f.metadata["source"], !source.isEmpty { rows.append((tr("Source"), source)) }
        if let about = f.summary { rows.append((tr("About"), about)) }
        if let author = f.author { rows.append((tr("Author"), author)) }
        if let license = f.license { rows.append((tr("License"), license)) }
        if let version = f.version { rows.append((tr("Version"), version)) }
        rows.append((tr("File"), m.id))
        return rows
    }

    /// "image  1 x 3 x 640 x 640  Float16"
    private func portText(_ p: ModelPort) -> String {
        var parts = [p.name, p.shape.isEmpty ? tr("shape not declared") : p.shape]
        if !p.dataType.isEmpty { parts.append(p.dataType) }
        return parts.joined(separator: "   ")
    }

    private var customModelsSection: some View {
        Section {
            Button {
                importingFolder = false
                importing = true
            } label: {
                Label(tr("Load custom Core ML model"), systemImage: "plus.circle")
            }
            .disabled(checkingImport)
            Button {
                importingFolder = true
                importing = true
            } label: {
                Label(tr("Load models from a folder"), systemImage: "folder.badge.plus")
            }
            .disabled(checkingImport)
            if checkingImport {
                HStack(spacing: 10) {
                    ProgressView()
                    Text(tr("Checking the model..."))
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
            Text(tr("Custom models"))
        } footer: {
            Text(tr("Import a .mlpackage, .mlmodelc, or .mlmodel. Use Load models from a folder if Files displays a model package as a folder."))
        }
    }

    private var eraseSection: some View {
        Section {
            Button(role: .destructive) {
                confirmErase = true
            } label: {
                // red, stated here and not left to the destructive role. With nothing to erase
                // the button is disabled, and shows it by fading.
                Label(tr("Erase all benchmark history"), systemImage: "trash")
                    .foregroundStyle(.red)
                    .opacity(history.runs.isEmpty ? 0.4 : 1)
            }
            .disabled(history.runs.isEmpty)
        } footer: {
            Text(history.runs.count == 1 ? tr("{0} saved run.", history.runs.count)
                                         : tr("{0} saved runs.", history.runs.count))
        }
    }

    /// Right under Erase: a row that opens the form for a new issue in the app's repository,
    /// in the browser. Same shape as the Privacy Policy row.
    @ViewBuilder private var reportSection: some View {
        if let url = URL(string: Self.issueURL) {
            Section {
                Link(destination: url) {
                    HStack {
                        Label(tr("Report an issue"), systemImage: "exclamationmark.bubble")
                        Spacer(minLength: 0)
                        // no colour of its own: it takes the row's, like the title beside it
                        // (it was tertiary, and looked dimmer than the text)
                        Image(systemName: "arrow.up.right")
                            .font(.caption)
                    }
                }
            } footer: {
                Text(tr("Opens a new issue on GitHub in your browser."))
            }
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
                problems.append(tr("{0} is not a Core ML model and contains none (.mlpackage, .mlmodelc, .mlmodel).", picked.lastPathComponent))
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
                    problems.append(tr("{0} was not imported. {1}", src.lastPathComponent, why))
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
