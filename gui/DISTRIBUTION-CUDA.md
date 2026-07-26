YOLO-Master - Windows Runner (GUI) - CUDA build
===============================================

Self-contained. Unzip anywhere and run yolomaster_gui.exe - no install, no admin,
and NO CUDA toolkit or cuDNN needed on this machine: the required libraries are
bundled.

This is the large build. If you do not need ONNX on the GPU, prefer the standard
bundle (~10x smaller); it still does GPU inference through ncnn-Vulkan and
MNN-OpenCL.

System requirements
-------------------
- Windows 10 (1703+) or Windows 11, 64-bit.
- An NVIDIA GPU with a current driver. For the CUDA path the driver must be new
  enough for CUDA 12 (R525+); any recent GeForce/Quadro driver qualifies.
  RTX 30-series or newer recommended (Ampere / Ada / Blackwell).
- No CUDA toolkit, no cuDNN install: those DLLs ship inside this folder.

GPU acceleration
----------------
Set Device -> GPU, then load a model:
  - .onnx model -> NVIDIA CUDA          (EP label: CUDA)      <- this build
  - ncnn model  -> Vulkan               (EP label: ncnn-Vulkan)
  - .mnn model  -> OpenCL               (EP label: MNN-OpenCL)

Check it really ran on the GPU: the MODEL card shows the execution provider, and
`nvidia-smi` should show GPU utilisation while inferring.

First inference after selecting CUDA is slow - often 30-90 seconds. cuDNN is
picking convolution algorithms and the driver may be JIT-compiling kernels for
your GPU. The app can look frozen; it is not. Subsequent frames are fast, and
the result is cached for the rest of the session.

If Device -> GPU falls back to CPU, the MODEL card prints the reason under the
model line ("CUDA EP failed: ..."). That message names the missing piece.

What is in the bundle
---------------------
  yolomaster_gui.exe            the app
  onnxruntime.dll               ONNX Runtime
  onnxruntime_providers_cuda.dll + _shared.dll    CUDA execution provider
  cudart64_12 / cublas64_12 / cublasLt64_12       CUDA runtime + BLAS
  cudnn*64_9.dll                cuDNN 9 (graph / ops / cnn / engines / heuristic)
  ncnn.dll (+ deps)             ncnn (Vulkan-enabled)
  MNN.dll (+ deps)              MNN (OpenCL/Vulkan-enabled)
  opencv_world*.dll             image / video / camera I/O
  opencv_videoio_ffmpeg*.dll    video decode (mp4/avi/mov/mkv)
  vcruntime140*.dll, msvcp140.dll, vcomp140.dll   MSVC runtime (app-local)
  assets\                       About-page logos + avatar
  models\                       bundled models (v0.1-seg-N segmentation is the
                                default and auto-loads on launch)

cuDNN's RNN/attention libraries (cudnn_adv*) are deliberately omitted: a
convolutional detector never loads them, and they add hundreds of MB.

Getting started
---------------
1. Unzip.
2. Run yolomaster_gui.exe. The bundled YOLO-Master v0.1-seg-N (segmentation,
   COCO-80) loads automatically.
3. Set Device -> GPU for CUDA, then open image / video / folder, or Live Webcam.

Notes
-----
- First launch may show a Windows SmartScreen prompt because the exe is not code-
  signed ("More info" -> "Run anyway"). Signing requires a code-signing certificate.
- License: AGPL-3.0. See the About page for full terms and acknowledgements.
