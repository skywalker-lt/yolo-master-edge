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
/// upstream libraries it merely acknowledges.
let baseProjects: [Ack] = [
    Ack(name: "YOLO-Master @ Tencent",
        blurb: "The MoE YOLO detector family this app packages.",
        copyright: "Copyright 2026 Tencent", license: "AGPL-3.0",
        repo: "https://github.com/Tencent/YOLO-Master"),
    Ack(name: "YOLO-Master Edge (this app)",
        blurb: "The iOS app and the Core ML export toolchain.",
        copyright: "Copyright 2026 Thomas (Ruiheng Li), HKUST", license: "AGPL-3.0",
        repo: "https://github.com/skywalker-lt/yolo-master-edge"),
]

// "Core ML" on its own: naming the company beside it read as an affiliation.
// The copyright line below the row still states who holds the rights.
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
YOLO-Master for iOS
Copyright (C) 2026 (Thomas) RUIHENG LI
(Author affiliated with The Hong Kong University of Science and
Technology)

YOLO-Master for iOS runs YOLO-Master models on-device using
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

    private var modelTypes: [UTType] {
        ["mlpackage", "mlmodelc", "mlmodel"].compactMap { UTType(filenameExtension: $0) }
    }

    var body: some View {
        NavigationStack {
            Form {
                aboutSection
                licensesSection
                privacySection
                computeSection
                customModelsSection
                eraseSection
            }
            .navigationTitle("Settings")
            .fileImporter(isPresented: $importing, allowedContentTypes: modelTypes,
                          allowsMultipleSelection: true) { importModels($0) }
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
            .onAppear { customModels = BundledModel.customModels() }
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
                Text("YOLO-Master for iOS").font(.title3.bold())
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
                    Label(a.repo.replacingOccurrences(of: "https://", with: ""),
                          systemImage: "link").font(.caption)
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
                    Text("YOLO-Master for iOS runs entirely on your device.")
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

    private var customModelsSection: some View {
        Section {
            Button {
                importing = true
            } label: {
                Label("Load custom Core ML model", systemImage: "plus.circle")
            }
            ForEach(customModels) { m in
                HStack {
                    Image(systemName: "cube.box").foregroundStyle(.secondary)
                    Text(m.fullName).lineLimit(1)
                    Spacer()
                    Button(role: .destructive) {
                        BundledModel.deleteCustom(m)
                        customModels = BundledModel.customModels()
                    } label: {
                        Image(systemName: "trash").foregroundStyle(.red)
                    }
                    .buttonStyle(.borderless)
                }
            }
        } header: {
            Text("Custom models")
        } footer: {
            Text("Import a .mlpackage, .mlmodelc, or .mlmodel. Imported models appear in the Live, Photo, and Bench pickers the next time you open that tab. Compilation happens on first load and may take a moment.")
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
            let dir = BundledModel.customModelsDir
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for src in urls {
                let scoped = src.startAccessingSecurityScopedResource()
                defer { if scoped { src.stopAccessingSecurityScopedResource() } }
                let dest = dir.appendingPathComponent(src.lastPathComponent)
                do {
                    try? FileManager.default.removeItem(at: dest)
                    try FileManager.default.copyItem(at: src, to: dest)
                } catch {
                    importError = error.localizedDescription
                }
            }
            customModels = BundledModel.customModels()
        }
    }
}
