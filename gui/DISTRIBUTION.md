YOLO-Master - Windows Runner (GUI)
==================================

Self-contained build. Unzip anywhere and run yolomaster_gui.exe - no install,
no admin, no runtime downloads.

System requirements
-------------------
- Windows 10 (1703+) or Windows 11, 64-bit.
- An NVIDIA GPU, GeForce RTX 30-series or newer (Ampere / Ada / Blackwell), with
  a current NVIDIA driver. The driver already provides Vulkan and OpenCL - no
  CUDA toolkit or cuDNN install is needed.
- Works on machines without a discrete GPU too, on CPU (just slower).

GPU acceleration
----------------
Set Device -> GPU in the app, then load a model:
  - ncnn model  -> runs on the GPU via Vulkan   (EP label: ncnn-Vulkan)
  - .mnn model  -> runs on the GPU via OpenCL    (EP label: MNN-OpenCL)
  - .onnx model -> CPU only in this build (see note)

Both Vulkan and OpenCL are supplied by your NVIDIA driver, so GPU inference works
on any RTX 30/40/50 card with no extra downloads. Confirm it is really on the GPU
by watching GPU-Util rise in `nvidia-smi` while inferring.

Note on ONNX + CUDA: ONNX Runtime's CUDA provider plus cuDNN would add several GB
to the download, so it is intentionally NOT bundled. ONNX still runs on CPU here.
If you specifically need ONNX on the GPU, build from source and point the build at
the ONNX Runtime GPU package + a matching CUDA/cuDNN install (see gui/README.md).
For GPU inference in the shipped build, use the ncnn or MNN model instead.

What is in the bundle
---------------------
  yolomaster_gui.exe            the app
  onnxruntime.dll               ONNX Runtime (CPU)
  ncnn.dll (+ deps)             ncnn (Vulkan-enabled)
  MNN.dll (+ deps)              MNN (OpenCL/Vulkan-enabled)
  opencv_world*.dll             image / video / camera I/O
  opencv_videoio_ffmpeg*.dll    video decode (mp4/avi/mov/mkv)
  vcruntime140*.dll, msvcp140.dll, vcomp140.dll   MSVC runtime (app-local)
  assets\                       About-page logos + avatar
  models\                       bundled models (v0.1-seg-N segmentation is the
                                default and auto-loads on launch)

Getting started
---------------
1. Unzip.
2. Run yolomaster_gui.exe. The bundled YOLO-Master v0.1-seg-N (segmentation,
   COCO-80) loads automatically - the MODEL card should show it on startup.
3. Open image / video / folder, or Live Webcam.
4. Optional: Browse to a different model (.onnx / .mnn / ncnn folder), and set
   Device -> GPU (use the ncnn or MNN model for GPU in this bundle).

Notes
-----
- First launch may show a Windows SmartScreen prompt because the exe is not code-
  signed ("More info" -> "Run anyway"). Signing requires a code-signing certificate.
- License: AGPL-3.0. See the About page for full terms and acknowledgements.
