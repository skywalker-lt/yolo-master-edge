// Zoom & pan for the preview stage - images always, video when paused.
//
// One ZoomModel is shared by the image and video stages (they are mutually exclusive on
// screen). The container applies .scaleEffect + .offset AFTER layout, so wrapped content
// keeps its own fit math (the VideoStage Canvas computes ox/oy/scale from its pre-transform
// size and stays aligned with the AVPlayerLayer underneath).
//
// Interactions: trackpad pinch (cursor-anchored), drag-to-pan when zoomed (never intercepts
// clicks at 1x), ⌘+/⌘−/⌘0 (center-anchored steps; ⌘0 resets). Scroll-wheel zoom is
// deliberately absent for v1.1.0 - SwiftUI exposes no scroll event; ZoomModel is the seam an
// NSEvent monitor would feed if it's ever wanted.
import SwiftUI

final class ZoomModel: ObservableObject {
    static let maxScale: CGFloat = 8
    @Published var scale: CGFloat = 1          // 1…maxScale
    @Published var offset: CGSize = .zero      // pan, in container points
    // gesture-start snapshots
    fileprivate var baseScale: CGFloat = 1
    fileprivate var baseOffset: CGSize = .zero

    var isZoomed: Bool { scale > 1.001 }
    func reset() { scale = 1; offset = .zero; baseScale = 1; baseOffset = .zero }

    /// Zoom by `factor` about the container center, clamped; snaps home near 1x.
    func step(_ factor: CGFloat, containerSize: CGSize) {
        let s = min(max(scale * factor, 1), Self.maxScale)
        let ratio = s / scale
        scale = s
        offset = CGSize(width: offset.width * ratio, height: offset.height * ratio)
        if scale < 1.02 { reset() }
        clampOffset(containerSize)
    }

    fileprivate func clampOffset(_ size: CGSize) {
        // content is fitted at 1x, so anything beyond ±(scale-1)·size/2 is empty margin
        let mx = (scale - 1) * size.width / 2, my = (scale - 1) * size.height / 2
        offset.width = min(max(offset.width, -mx), mx)
        offset.height = min(max(offset.height, -my), my)
    }
}

/// Wrap fitted content to make it zoomable. `enabled: false` renders passthrough (used for
/// video during playback - zoom auto-resets on play via the app's onChange wiring).
struct ZoomContainer<Content: View>: View {
    @ObservedObject var zoom: ZoomModel
    var enabled: Bool = true
    @ViewBuilder let content: () -> Content

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            ZStack {
                content()
                    .frame(width: size.width, height: size.height)
                    .scaleEffect(zoom.scale)
                    .offset(zoom.offset)
                    .frame(width: size.width, height: size.height)
                    .clipped()
                // Gestures live on a transparent layer ABOVE the content: the video stage hosts
                // a raw NSView (AVPlayerLayer), and AppKit views swallow mouse events before
                // SwiftUI ancestor gestures fire. Nothing interactive sits under this layer
                // inside the container, so it can safely own click-drag/pinch.
                if enabled {
                    Color.clear
                        .contentShape(Rectangle())
                        .gesture(pan(size), isEnabled: zoom.isZoomed)   // click-and-drag to pan
                        .gesture(magnify(size))
                }
            }
                .overlay(alignment: .topTrailing) {
                    if zoom.isZoomed {
                        Text("\(Int((zoom.scale * 100).rounded()))% - ⌘0 to reset")
                            .font(.caption2).padding(.horizontal, 8).padding(.vertical, 3)
                            .background(.ultraThinMaterial, in: Capsule())
                            .padding(8)
                            .transition(.opacity)
                            .allowsHitTesting(false)
                    }
                }
                .overlay { keyboardButtons(size) }
        }
    }

    /// Cursor-anchored pinch: the point under the fingers stays put.
    private func magnify(_ size: CGSize) -> some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let s = min(max(zoom.baseScale * value.magnification, 1), ZoomModel.maxScale)
                // anchor point relative to container center
                let pc = CGSize(width: value.startLocation.x - size.width / 2,
                                height: value.startLocation.y - size.height / 2)
                let ratio = s / zoom.baseScale
                zoom.scale = s
                zoom.offset = CGSize(width: pc.width - (pc.width - zoom.baseOffset.width) * ratio,
                                     height: pc.height - (pc.height - zoom.baseOffset.height) * ratio)
                zoom.clampOffset(size)
            }
            .onEnded { _ in
                if zoom.scale < 1.02 { zoom.reset() }
                zoom.baseScale = zoom.scale
                zoom.baseOffset = zoom.offset
            }
    }

    private func pan(_ size: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { value in
                zoom.offset = CGSize(width: zoom.baseOffset.width + value.translation.width,
                                     height: zoom.baseOffset.height + value.translation.height)
                zoom.clampOffset(size)
            }
            .onEnded { _ in zoom.baseOffset = zoom.offset }
    }

    /// Hidden buttons carrying the keyboard shortcuts (fire regardless of focus, and don't
    /// collide with the arrow/space onKeyPress layer).
    private func keyboardButtons(_ size: CGSize) -> some View {
        Group {
            Button("") { if enabled { zoomStep(1.25, size) } }.keyboardShortcut("=", modifiers: .command)
            Button("") { if enabled { zoomStep(1.25, size) } }.keyboardShortcut("+", modifiers: .command)
            Button("") { if enabled { zoomStep(1 / 1.25, size) } }.keyboardShortcut("-", modifiers: .command)
            Button("") { if enabled { withAnimation(.snappy(duration: 0.2)) { zoom.reset() } } }
                .keyboardShortcut("0", modifiers: .command)
        }
        .buttonStyle(.plain).opacity(0).frame(width: 0, height: 0).allowsHitTesting(false)
    }

    private func zoomStep(_ f: CGFloat, _ size: CGSize) {
        withAnimation(.snappy(duration: 0.15)) { zoom.step(f, containerSize: size) }
        zoom.baseScale = zoom.scale
        zoom.baseOffset = zoom.offset
    }
}
