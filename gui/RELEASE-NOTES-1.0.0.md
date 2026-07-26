# For Windows 10/11: YOLO-Master Windows Runner (GUI) 1.0.0

On-device [YOLO-Master](https://github.com/Tencent/YOLO-Master) object **detection** and instance **segmentation** for Windows, accelerated via **NVIDIA CUDA**, **Vulkan**, or **OpenCL**. A native Win32 + Direct3D 11 app: pick a model and a source (image, folder, video, or the live webcam) and it infers on-device — no command line, no cloud, nothing leaves your PC.

This is the first public release, and the companion to the [macOS Core ML Runner](https://github.com/skywalker-lt/yolo-master-edge/releases). The CUDA compatibility issues mentioned in that release are resolved.

## ✨ Features

- **Three Backends, One App**: [ONNX Runtime](https://onnxruntime.ai/), [ncnn](https://github.com/Tencent/ncnn), and [MNN](https://github.com/alibaba/MNN) all ship in the same binary. Switch backend from the sidebar and compare on the same image — the preprocessing and decode path is shared, so results match across all three.
- **Detection & Segmentation**: Runs both bounding-box detectors and instance-segmentation models. Masks are anti-aliased (no serrated edges), with a Masks / Boxes / Both overlay toggle.
- **Images, Video & Live Camera**: Single images, whole-folder batches, and MP4/AVI/MOV/MKV video, plus a low-latency **live webcam** mode with a real-time FPS / ms-per-frame HUD and a mirror toggle.
- **Real-Time Tuning**: Confidence, IoU (NMS), box style, and labels redraw instantly; the forward pass is cached, so tuning never re-runs inference. Letterbox vs. stretch preprocessing is also switchable.
- **Two-Phase Pipeline**: Folders and videos are inferred once with a progress bar, then browsed and scrubbed at full speed with the tuned parameters — a 30 fps clip plays back at 30 fps.
- **CPU or GPU**: One switch. ONNX runs on **CUDA**, ncnn on **Vulkan**, MNN on **OpenCL**, all in FP16 on the GPU; every backend falls back to CPU cleanly and tells you why if a GPU provider is unavailable.
- **Finder-Style Browser**: Folder batches get a thumbnail grid or list view with a resizable icon size, and macOS-Finder-style arrow-key navigation.
- **Bundled Default Model**: Ships with a segmentation model that loads the moment you open the app; load any other exported ONNX / ncnn / MNN model at any time.

## 🚀 Performance

Live camera inference (higher is better) on an **RTX 5070 Ti**, `v0.1-seg-N` at 640px:

| Backend | Device | FPS |
| :------ | :----- | ---: |
| ONNX Runtime | CUDA | **80** |
| MNN | OpenCL | 44 |
| ncnn | Vulkan | 37 |
| ONNX Runtime | CPU | 23 |

<!-- TODO(verify): the ncnn/MNN rows were measured BEFORE FP16 was enabled on the
     GPU paths. Re-measure both before publishing; they should be materially higher.
     Also fill in the CPU model for the CPU row, and add other GPUs if available. -->

Segmentation at real-time rates on a consumer laptop GPU, with no cloud round-trip.

## 🖥️ Demo Screenshot

<!-- TODO: paste a screenshot here (drag it into the GitHub release editor).
     Suggestion: the seg model on a street scene with masks + labels visible,
     the sidebar showing CUDA, and the folder thumbnail grid open. -->

## 📥 Installation

Two builds are attached. Both are self-contained: unzip anywhere and run — no installer, no admin rights, no dependencies to download.

| Download | Size | Use it if |
| :------- | ---: | :-------- |
| `YOLO-Master-Windows-1.0.0.zip` | ~150 MB | You want the small download. GPU inference via ncnn-Vulkan / MNN-OpenCL. |
| `YOLO-Master-Windows-CUDA-1.0.0.zip` | ~2 GB | You want the fastest path — ONNX on CUDA. Bundles the CUDA + cuDNN libraries. |

<!-- TODO: replace both sizes with the actual figures printed by package.cmd /
     package-cuda.cmd, and match the file names to the uploaded assets. -->

1. Download a zip below and unzip it.
2. Run **yolomaster_gui.exe**.

The bundled segmentation model loads automatically, so you can open an image straight away.

> **On first launch**, Windows SmartScreen may warn that the publisher is unrecognised — the executable is not code-signed (that needs a paid certificate). Click **More info → Run anyway**. Everything runs locally; the app makes no network requests.

> **In the CUDA build**, the *first* inference after selecting Device → GPU takes 30–90 seconds while cuDNN selects convolution algorithms and the driver compiles kernels for your GPU. The window may look frozen; it isn't. Every frame after that is fast.

## 💻 Requirements

- **Windows 10 (1703 or later) or Windows 11**, 64-bit
- **An NVIDIA GPU with a current driver** for GPU inference — GeForce RTX 30-series or newer recommended (Ampere / Ada / Blackwell). Vulkan and OpenCL come with the driver; the CUDA build additionally needs a CUDA 12-capable driver (R525+)
- **No CUDA toolkit or cuDNN installation required** — the CUDA build bundles them
- Runs on machines without a discrete GPU too, on CPU

## 🤝 Acknowledgements

Built as an extension tool of [YOLO-Master](https://github.com/Tencent/YOLO-Master).
We thank [Ultralytics](https://github.com/ultralytics/ultralytics), [ONNX Runtime](https://github.com/microsoft/onnxruntime), [ncnn](https://github.com/Tencent/ncnn), [MNN](https://github.com/alibaba/MNN), [OpenCV](https://github.com/opencv/opencv), and [Dear ImGui](https://github.com/ocornut/imgui) for their great work. Licensed under AGPL-3.0.

## 🔖 Future Work

Annotated export (images, folder batches, and MP4) is next, along with drag-and-drop and remembered settings. AMD and Intel GPU support already works through the Vulkan and OpenCL paths but has not been tested — reports welcome.
