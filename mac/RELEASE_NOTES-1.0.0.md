# YOLO-Master CoreML Runner 1.0.0

On-device [YOLO-Master](https://github.com/Tencent/YOLO-Master) object **detection** and instance **segmentation** for macOS, running through Apple [Core ML](https://developer.apple.com/documentation/coreml). A native SwiftUI app — pick a model and a source (image, folder, video, or the live webcam) and it infers on-device: no command line, no cloud, nothing leaves your Mac.

This is the first public release.

## ✨ Features

- **Detection & Segmentation** — Runs both bounding-box detectors and instance-segmentation models. Masks are anti-aliased (no serrated edges), with a Masks / Boxes / Both overlay toggle.
- **Images, Video & Live Camera** — Single images, whole-folder batches, and MP4 video, plus a low-latency **live webcam** mode with a real-time FPS / ms-per-frame HUD and a mirror toggle.
- **Real-Time Tuning** — Confidence, IoU (NMS), box style, and labels redraw instantly; the forward pass is cached, so tuning never re-runs inference. Letterbox vs. stretch preprocessing is switchable.
- **Two-Phase Pipeline** — Folders and videos are inferred once with a progress bar, then browsed, scrubbed, and exported with the tuned parameters.
- **Export** — Write annotated images or an annotated MP4 with the current overlay and style.
- **Bundled Default Model** — Ships with a segmentation model, so it runs the moment you open it; load any other exported Core ML model at any time.

## 🚀 Performance

Live inference throughput (FPS) on Apple Silicon, `.mlpackage` models via the Core ML **CPU + GPU** compute unit:

| Model        | M1     | M4 Max |
| :----------- | :----- | :----- |
| v0.1-seg-N   | 27.1   | 33.0   |
| v0.1-N       | 27.6   | 29.8   |
| EsMoE-N      | 28.5   | 33.2   |
| UoMoE-N      | 29.9   | 32.3   |

Every model runs comfortably in real time even on the base M1; throughput scales with the Mac's GPU and the selected compute unit.

## 📥 Installation

1. Download `YOLO-Master-CoreML-Runner-1.0.0.zip` below and unzip it.
2. Double-click **YOLO-Master CoreML Runner.app**.

That's it. The app is **Developer-ID signed and notarized by Apple**, so it opens with a normal double-click — no "unidentified developer" warning, no right-click workaround, and nothing to trust manually. Camera access is requested on first use of Live Camera (processed entirely on-device).

## 💻 Requirements

- **macOS 14 (Sonoma) or later**
- **Apple Silicon or Intel** — the app is a universal binary
- No dependencies to install; the Core ML backend and default model are bundled

## 🙏 Acknowledgements

Built on [YOLO-Master](https://github.com/Tencent/YOLO-Master) (Tencent), [Ultralytics](https://github.com/ultralytics/ultralytics), and Apple [Core ML / coremltools](https://github.com/apple/coremltools). Licensed under AGPL-3.0 — full text and credits are in the app's **About & Licenses** page.
