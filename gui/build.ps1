# Build the YOLO-Master Edge Windows GUI (Dear ImGui + D3D11, CPU backends).
# Usage (from a "x64 Native Tools Command Prompt for VS" or plain PowerShell with CMake on PATH):
#   ./build.ps1                       # configure + build Release
#   ./build.ps1 -Run                  # build then launch
#   ./build.ps1 -Clean                # wipe the build dir first
#
# Edit the four SDK paths below to match where you unpacked each dependency.
param(
  [switch]$Run,
  [switch]$Clean,
  [string]$OnnxRoot = "C:/dev/onnxruntime",   # onnxruntime-win-x64-1.18.1  (include/ + lib/)
  [string]$NcnnRoot = "C:/dev/ncnn",          # ncnn-YYYYMMDD-windows-vs2022-shared/x64 (include/ lib/ bin/)
  [string]$MnnRoot  = "C:/dev/mnn",           # MNN Windows build          (include/ + lib/ or build/Release)
  [string]$OpenCVDir= "C:/dev/opencv/build"   # OpenCV build dir (has OpenCVConfig.cmake + x64/vc16/bin)
)

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $root "build"

if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

cmake -S $root -B $build -G "Visual Studio 17 2022" -A x64 `
  "-DONNXRUNTIME_ROOT=$OnnxRoot" `
  "-DNCNN_ROOT=$NcnnRoot" `
  "-DMNN_ROOT=$MnnRoot" `
  "-DOpenCV_DIR=$OpenCVDir"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build $build --config Release -j
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$exe = Join-Path $build "Release/yolomaster_gui.exe"
Write-Host "`nBuilt: $exe" -ForegroundColor Green
if ($Run) { & $exe }
