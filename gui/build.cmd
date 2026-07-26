@echo off
setlocal
rem ============================================================================
rem  Build the YOLO-Master Windows Runner (GUI). Batch equivalent of build.ps1,
rem  for when PowerShell execution policy blocks .ps1 files.
rem  Run from a "x64 Native Tools Command Prompt for VS" (cmake + MSVC on PATH).
rem
rem  Usage:  build.cmd          configure + build Release
rem          build.cmd run      build then launch
rem          build.cmd clean    wipe build\ first
rem
rem  Edit the SDK paths below to match your machine.
rem ============================================================================

set "ONNX_ROOT=C:/dev/onnxruntime-win-x64-gpu_cuda12-1.27.1"
set "NCNN_ROOT=C:/dev/ncnn-20260526-windows-vs2022-shared/x64"
set "MNN_ROOT=C:/dev/mnn_3.6.1_windows_x64_cpu_opencl_vulkan_avx512"
set "OPENCV_DIR=C:/dev/opencv/build/x64/vc16/lib"
rem optional: bundle cuDNN + CUDA runtime so ONNX CUDA works after a clean build
set "CUDNN_DIR=C:/Program Files/NVIDIA/CUDNN/v9.12/bin/12.9"
set "CUDA_BIN_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"

set "ROOT=%~dp0"
set "BUILD=%ROOT%build"

if /i "%~1"=="clean" if exist "%BUILD%" rmdir /s /q "%BUILD%"

rem the linker cannot overwrite a running exe -- close it first
taskkill /IM yolomaster_gui.exe /F >nul 2>&1

cmake -S "%ROOT%." -B "%BUILD%" -A x64 ^
  "-DONNXRUNTIME_ROOT=%ONNX_ROOT%" "-DNCNN_ROOT=%NCNN_ROOT%" ^
  "-DMNN_ROOT=%MNN_ROOT%" "-DOpenCV_DIR=%OPENCV_DIR%" ^
  "-DCUDNN_DIR=%CUDNN_DIR%" "-DCUDA_BIN_DIR=%CUDA_BIN_DIR%"
if errorlevel 1 exit /b 1

cmake --build "%BUILD%" --config Release -j
if errorlevel 1 exit /b 1

echo.
echo Built: %BUILD%\Release\yolomaster_gui.exe
if /i "%~1"=="run" start "" "%BUILD%\Release\yolomaster_gui.exe"
exit /b 0
