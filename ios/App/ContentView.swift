import SwiftUI
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

/// App info, compute safety, history reset, and custom model import.
struct SettingsView: View {
    @AppStorage("allowCPU") private var allowCPU = false
    @ObservedObject private var history = BenchHistory.shared
    @State private var customModels: [BundledModel] = []
    @State private var importing = false
    @State private var confirmErase = false
    @State private var importError: String?

    private var modelTypes: [UTType] {
        ["mlpackage", "mlmodelc", "mlmodel"].compactMap { UTType(filenameExtension: $0) }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("About") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("YOLO-Master").font(.headline)
                        Text("On-device object detection powered by Core ML and YOLOMasterKit \u{2014} the same inference path as the macOS tools.")
                            .font(.footnote).foregroundStyle(.secondary)
                        Text("Live camera, photo-batch, and an on-device benchmark all run locally on the Neural Engine or GPU. Nothing you capture leaves the device.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 2)
                }

                Section("Licensing") {
                    Text("This app bundles YOLOMasterKit and YOLO-Master detection models. YOLO-Master builds on Ultralytics YOLO, licensed under AGPL-3.0. Core ML model weights are distributed under their respective model licenses. Full license texts are in the project repository.")
                        .font(.footnote).foregroundStyle(.secondary)
                }

                Section {
                    Toggle("Allow CPU inference", isOn: $allowCPU)
                } header: {
                    Text("Compute")
                } footer: {
                    Text("CPU-only inference can crash Live and Photo on some models, so it is off by default. Enable to expose the CPU option in the Live and Photo compute pickers. The Bench tab always measures CPU.")
                }

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
                    HStack(spacing: 6) {
                        Text("Custom models")
                        Text("BETA").font(.caption2.bold())
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.orange.opacity(0.25), in: Capsule())
                            .foregroundStyle(.orange)
                    }
                } footer: {
                    Text("Import a .mlpackage, .mlmodelc, or .mlmodel. Imported models appear in the Live, Photo, and Bench pickers the next time you open that tab. Compilation happens on first load and may take a moment.")
                }

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
