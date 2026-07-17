# YOLO-Master Edge Inference C++ Runtime

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white"> <img alt="NCNN" src="https://img.shields.io/badge/NCNN-Tencent-blue.svg"> <img alt="MNN" src="https://img.shields.io/badge/MNN-Alibaba-orange.svg"> <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg">

This project provides a universal inference runtime for [YOLO-Master](https://github.com/Tencent/YOLO-Master) object-detection models, leveraging, [ONNX Runtime](https://onnxruntime.ai/), [NCNN](https://github.com/Tencent/ncnn), [MNN](https://github.com/alibaba/mnn), [TensorRT](https://github.com/nvidia/tensorrt), and [CoreML (NEW!)](https://github.com/apple/coremltools) backends. It runs on almost every platform: Linux, Windows (10/11), Jetson, and MacOS; supports CPU, [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit), and [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders). It's capable of auto-detecting the model format, class names, and input size -- designed for real-time, end-to-end edge deployment in some of the most challenging tasks (VisDrone, SKU-110K, AI-TOD-v2, etc.).

## 🚀🍎 Update (17-07-2026): YOLO-Master CoreML Runner for MacOS (GUI)

**[Download](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.0.0-macos/YOLO-Master-CoreML-Runner-1.0.0.zip) and try it now!**

Alongside the Linxu and Windows C++ runtime, we now provide a native, user-friendly macOS runner, **YOLO-Master CoreML Runner** — a [SwiftUI](https://developer.apple.com/xcode/swiftui/) frontend over an Apple [Core ML](https://developer.apple.com/documentation/coreml) backend for on-device [YOLO-Master](https://github.com/Tencent/YOLO-Master) inference, no command line required. It ships with a default `YOLO-Master-v0.1-seg-N` segmentation model, so it runs out of the box. 

<img width="480" alt="Screenshot1" src="https://github.com/user-attachments/assets/d1747b4d-0961-458e-99c5-2a9870b8df96" /> <img width="480" alt="Screenshot2" src="https://github.com/user-attachments/assets/5f71d80a-6238-49bd-a230-95ccd4020d29" /> <img width="480" alt="Screenshot3" src="https://github.com/user-attachments/assets/9cc60636-b795-4326-992c-06239a77db55" /> <img width="480" alt="Screenshot4" src="https://github.com/user-attachments/assets/b5ee48bb-52dc-4ff7-b0bd-f2461b34ad7c" />

- **Detection & Segmentation:** Runs both bounding-box detectors and instance-segmentation models, with anti-aliased mask overlays and a Masks / Boxes / Both toggle.
- **Images, Video & Live Camera:** Infers single images, whole folders (batch), and MP4 video, plus a low-latency **live webcam** mode with a real-time FPS / ms-per-frame readout.
- **⭐️ Real-Time Tuning:** Confidence, IoU, box style, labels, and letterbox/stretch preprocessing are all adjustable live:  the forward pass is cached, so tuning re-draws without re-inferring.
- **Signed & Notarized:** A **universal** (Apple Silicon + Intel) bundle, **Developer-ID signed and notarized by Apple**: it installs by a simple double-click on any Mac with MacOS 14+.

For more details, please check the [Release](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.0.0-macos) page.

The refined Windows 10/11 Runner with GUI is also in developmet.

---


## ✨ Benefits

- **Universal Binary for Linux and Windows:** A single executable integrates **ONNX Runtime**, **NCNN** and **MNN** backends; the backend, class names, and input size are auto-detected from the model — no recompilation or any dataset YAML needed at runtime.
- **Verified Accuracy:** Reproduces the PyTorch original to **< 0.5%** mAP50-95 across ONNX / NCNN / MNN, and **< 1.0%** under INT8 quantization, on 548 VisDrone validation images.
- **Deployment-Friendly:** Cross-platform [CMake](https://cmake.org/) build producing **self-contained and relocatable bundles** for Linux x86_64 and Windows 10/11 — installable by unzip, no dependencies on the target.
- **GPU Acceleration:** Supports FP32 CPU inference and [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) GPU acceleration through the ONNX Runtime CUDA Execution Provider on Linux, on [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) Orin via a native TensorRT backend (JetPack 7), and accelerated on MacOS via [MPS](https://developer.apple.com/documentation/metalperformanceshaders) beind Core ML.

## ☕ Note

The exported models embed their class names, input size, and stride as ONNX/NCNN/MNN metadata, so the runtime configures itself from the model file. Post-processing is tuned for the vertical domain — aspect-ratio-preserving letterbox, per-class **multi-label** NMS, and a low default confidence threshold appropriate for VisDrone's small, dense objects.

## 📦 Exporting Models

Pre-built models (trained on VisDrone) are attached to the [Releases](https://github.com/skywalker-lt/yolo-master-edge/releases) page. To export your own trained [YOLO-Master](https://github.com/Tencent/YOLO-Master) checkpoint, use the Ultralytics `export` mode.

### ONNX

```python
from ultralytics import YOLO

# Load a trained YOLO-Master-EsMoE-N checkpoint
model = YOLO("EsMoE-N_VisDrone.pt")

# opset=12 for broad compatibility (ORT + NCNN + MNN)
# simplify=True runs onnxsim; dynamic=False fixes the input shape for C++ deployment
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

### NCNN (via pnnx) and MNN

```bash
# NCNN — Ultralytics uses pnnx under the hood
yolo export model=EsMoE-N_VisDrone.pt format=ncnn imgsz=640

# MNN — convert the exported ONNX with MNN's converter
mnnconvert -f ONNX --modelFile esmoe_n_visdrone_sim.onnx --MNNModel esmoe_n_visdrone.mnn --bizCode edge
```

For more details on exporting, refer to the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/).

### Core ML

```zsh
# detector or segmenter (task auto-detected)
python coreml_export/export_coreml.py --weights model.pt --imgsz 640 --out model.mlpackage

# YOLO-Master default imgsz is 800 for AI-TOD models — pass --imgsz accordingly
python coreml_export/export_coreml.py --weights yolo-master-v0.1-N_aitodv2.pt --imgsz 800 --out v0.1-N.mlpackage

# sunsmarterjie/yolov12 checkpoints (split qk+v area-attention) — stock ultralytics + the flag
python coreml_export/export_coreml.py --weights yolov12x.pt --imgsz 640 --out yolov12x.mlpackage --yolov12-aattn

# a LoRA fine-tune: merge the trained adapters first
python coreml_export/export_coreml.py --weights base.pt --merge-lora-dir lora_adapter/ --imgsz 640 --out ft.mlpackage
```


## ⚙️ Dependencies

Ensure you have the following dependencies installed （not required if you only want to smoke-test the pre-built bundles):

### Linux & Windows

| Dependency                                                          | Version       | Notes                                                                                                          |
| :------------------------------------------------------------------ | :------------ | :------------------------------------------------------------------------------------------------------------- |
| [ONNX Runtime](https://onnxruntime.ai/docs/install/)                | >=1.18        | Download pre-built binaries or build from source. Use the GPU build for the CUDA Execution Provider.           |
| [NCNN](https://github.com/Tencent/ncnn/releases)                    | recent        | Tencent NCNN; on Windows use the `windows-vs2022` prebuilt.                                                     |
| [OpenCV](https://opencv.org/releases/)                              | >=4.5.0       | Used for image preprocessing (`core` + `imgproc`).                                                             |
| C++ Compiler                                                        | C++17 Support | Needed for `<filesystem>`. ([GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), MSVC 2022/2026)      |
| [CMake](https://cmake.org/download/)                                | >=3.16        | Cross-platform build system generator.                                                                         |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (Optional)| 12.x          | Required for GPU acceleration via ONNX Runtime's CUDA Execution Provider (match your ONNX Runtime GPU build).   |
| [MNN](https://github.com/alibaba/MNN) (Optional)                    | >=3.0         | Only for the third export format / benchmarking.                                                               |

> **Note:** The CUDA Execution Provider is ABI-coupled to a CUDA major version — use the ONNX Runtime GPU build that matches your CUDA Toolkit (e.g. the CUDA-12 build with CUDA 12.x), or you'll hit loader errors.


### MacOS
|                                                         | Version       | Notes                                                                                                          |
| :------------------------------------------------------------------ | :------------ | :------------------------------------------------------------------------------------------------------------- |
| MacOS | Sonoma or newer (14.0+) | SwiftUI API floor (onKeyPress, zero-param onChange)  |
| [Xcode Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/) | Xcode 15+    | Install with xcode-select --install. Provides swift, codesign, ditto. Full Xcode GUI not required for a build.   |
| [Swift toolchain](https://www.swift.org/swiftly/documentation/swiftly/install-toolchains/) | 5.9+  | swift-tools-version:5.9 in Package.swift; ships with the CLT/Xcode above. Build: swift build -c release --package-path mac. |       
| Apple SDK frameworks | macOS 14+ SDK (system) | SwiftUI, AppKit, AVFoundation, Core ML, Core Image, Core Video, ImageIO, QuartzCore, etc. |


## 🛠️ Build Instructions

### Linux & Windows

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/skywalker-lt/yolo-master-edge.git
    cd yolo-master-edge/cpp
    ```

2.  **Create Build Directory:**

    ```bash
    mkdir build && cd build
    ```

3.  **Configure with CMake:**
    Point CMake at your extracted ONNX Runtime and NCNN SDKs via `ONNXRUNTIME_ROOT` and `NCNN_ROOT`.

    ```bash
    # Example for Linux (adjust paths as needed)
    cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DNCNN_ROOT=/path/to/ncnn
    ```

    ```bat
    :: Example for Windows, from the "x64 Native Tools Command Prompt"
    cmake .. -DCMAKE_BUILD_TYPE=Release ^
      -DOpenCV_DIR=C:/dev/opencv/build/x64/vc16/lib ^
      -DONNXRUNTIME_ROOT=C:/dev/onnxruntime-win-x64 ^
      -DNCNN_ROOT=C:/dev/ncnn-windows-vs2022/x64
    ```

    **CMake Options:**
    - `-DONNXRUNTIME_ROOT=<path>`: **(Required)** Path to the extracted ONNX Runtime library.
    - `-DNCNN_ROOT=<path>`: **(Required)** Path to the extracted NCNN library.
    - `-DCMAKE_BUILD_TYPE=Release`: (Optional) Build with optimizations.
    - `-DPORTABLE=ON`: (Optional, Linux) Slim build for a small self-contained bundle (image inference only).
    - If CMake struggles to find OpenCV, set `-DOpenCV_DIR=/path/to/opencv/build`.

4.  **Build the Project:**
    Use the build tool generated by CMake (Make, Ninja, or Visual Studio).

    ```bash
    # Using CMake's generic build command (works with Make, Ninja, MSBuild)
    cmake --build . --config Release
    ```

5.  **Locate Executable:**
    The compiled executable (`yolomaster_edge`, or `yolomaster_edge.exe` on Windows) is located in the `build` directory. On Windows the required backend and OpenCV DLLs are auto-copied next to it.

### MacOS 

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/skywalker-lt/yolo-master-edge.git
    cd yolo-master-edge/cpp
    ```

2.  **Build the App and Run** 
    ```zsh
    xcode-select --install
    swift run -c release --package-path mac YOLOMasterApp
    ```

## 🚀 Usage

Run the executable, pointing it at a model and a source (image, directory, video, or `dataset.yaml`):

```bash
./yolomaster_edge --model ../../models/esmoe_n_visdrone_sim.onnx \
                  --source path/to/image_or_dir \
                  --conf 0.25 --out out
```

The backend is inferred from the model (`.onnx` → ONNX Runtime, an NCNN directory → NCNN), and class names and input size are read from the model metadata. Common options:

```text
--backend      auto | onnx | ncnn        (default: auto-detect)
--device       cpu | cuda                (ONNX backend; falls back to CPU)
--conf         confidence threshold      (default 0.25; lower for dense scenes)
--iou          NMS IoU threshold         (default 0.50)
--multi-label  one detection per class > conf per anchor (matches Ultralytics val mAP)
--save-txt     dir to write predictions  ('class conf x1 y1 x2 y2')
--out          dir for annotated outputs  --no-save / --quiet
```

See `cpp/run_tests.sh` for the 16-test robustness battery.

## 🤖 Jetson Orin (Native TensorRT)

A prebuilt aarch64 runner for **Jetson Orin** (Nano / NX / AGX) on **JetPack 7** is attached to the [Releases](https://github.com/skywalker-lt/yolo-master-edge/releases) page. It bundles OpenCV and uses JetPack's TensorRT + CUDA; the per-device FP16 engine is built once with the included script.

```bash
tar xzf yolomaster_edge-jetson-orin-jp7.tar.gz && cd yolomaster_edge-jetson-orin-jp7
./build_engine.sh    # builds the FP16 engine for this device (once, ~10-15 min)
./yolomaster_edge --model models/esmoe_n_fp16.engine --source <img|dir> --classes visdrone --out out
```

On an Orin Nano 4 GB the FP16 engine runs at **35.7 FPS** (27.8 ms) with **mAP50-95 0.2029 (−0.34% vs FP32)**. FP16 is the recommended target on this model — its area-attention does not quantize, so INT8 is both slower and less accurate here. To build from source, the [`jetson/`](jetson/) scripts drive the engine build and packaging; see [`jetson/README.md`](jetson/README.md) and [`jetson/DEPLOYMENT_LOG.md`](jetson/DEPLOYMENT_LOG.md).

## 📊 Results

Inference performed on full 548 VisDrone validation images against the PyTorch original (`mAP50-95 = 0.2036`), using identical settings (conf 0.001, NMS IoU 0.7, multi-label).

| Inference Backend                     | mAP50-95 | Δ vs PyTorch | Latency | FPS   |
| :------------------------ | :------- | :----------- | :------ | :---- |
| ONNX (ONNX Runtime, CPU)  | 0.2034   | −0.02%       | 40 ms   | 25.0  |
| NCNN (CPU)                | 0.2034   | −0.02%       | ~80 ms  | ~12.5 |
| MNN (CPU)                 | 0.2034   | −0.02%       | 74 ms   | 13.5  |
| INT8 mixed (CPU) ¹        | 0.1952   | −0.84%       | 137 ms  | 7.2   |
| ONNX CUDA (H200 GPU)      | 0.2033   | −0.03%       | 7.8 ms  | ~128  |
| TensorRT FP16 (Jetson Orin Nano 4GB) | 0.2029 | −0.34% | 27.8 ms | 35.7  |
| Core ML (M4 Max) | N/A (no validator bundled) | N/A | 17.4 ms | ~57.4 |

CPU latencies are x86 @ 4 threads on one host; mAP is identical across FP32 formats because they are of the same graph. The Jetson row is a native TensorRT FP16 engine, measured on-device (see below).

> ¹ INT8 is *slower* than FP32 on CPU — its throughput payoff needs INT8 tensor cores, not x86 CPUs. The CPU INT8 result is an **accuracy** proof (−0.84%, within budget); on the actual accelerator, note that even on the Orin's tensor cores FP16 wins here (the attention doesn't quantize — see the TensorRT row and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) §8).

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full methodology, INT8 quantization deep-dive, and numerical parity analysis.


## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the [project repository](https://github.com/skywalker-lt/yolo-master-edge).
