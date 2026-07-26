@echo off
rem ============================================================================
rem  SDK locations for the Windows runner build.
rem
rem  Copy this file to  sdk-paths.cmd  (same folder) and edit the paths to match
rem  your machine. sdk-paths.cmd is gitignored, so your local paths never get
rem  committed and are not clobbered when you pull.
rem
rem      copy sdk-paths.example.cmd sdk-paths.cmd
rem
rem  build.cmd / package.cmd / package-cuda.cmd all read this file. Anything you
rem  leave blank simply disables that backend (the build still succeeds without
rem  it). Forward slashes are fine and preferred.
rem ============================================================================

rem --- required -------------------------------------------------------------

rem OpenCV: the folder holding OpenCVConfig.cmake. For the official prebuilt
rem package that is <opencv>/build/x64/vc16/lib (NOT <opencv>/build).
set "OPENCV_DIR=C:/dev/opencv/build/x64/vc16/lib"

rem --- backends (set the ones you have) -------------------------------------

rem ONNX Runtime: folder containing include/ and lib/.
rem   CPU package  -> onnxruntime-win-x64-<ver>          (use for package.cmd)
rem   GPU package  -> onnxruntime-win-x64-gpu_cuda12-<ver>  (needed for CUDA)
set "ONNX_ROOT=C:/dev/onnxruntime-win-x64-gpu_cuda12-1.27.1"

rem ncnn: use the *-shared* release and point at its x64 subfolder
rem (the static build fails to link: it pulls in Vulkan symbols).
set "NCNN_ROOT=C:/dev/ncnn-<version>-windows-vs2022-shared/x64"

rem MNN: folder containing include/ and a MNN.lib. If your package nests the
rem libs (lib/x64/Release/Dynamic/MD), copy that folder's contents up to lib/.
set "MNN_ROOT=C:/dev/mnn_<version>_windows_x64_cpu_opencl_vulkan_avx512"

rem --- optional: ONNX on CUDA ------------------------------------------------
rem Only needed if ONNX_ROOT is the GPU package. These make the CUDA/cuDNN DLLs
rem get copied next to the exe, so ONNX-GPU keeps working after a clean rebuild.
rem cuDNN: the bin folder matching your CUDA major version, e.g. .../bin/12.9
set "CUDNN_DIR=C:/Program Files/NVIDIA/CUDNN/v9.12/bin/12.9"
set "CUDA_BIN_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"
