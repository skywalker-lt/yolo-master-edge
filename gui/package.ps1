# Package the YOLO-Master Windows GUI into a self-contained, redistributable zip.
#
# GPU acceleration ships via ncnn-Vulkan + MNN-OpenCL (supplied by the NVIDIA driver on
# any RTX 30-series+), NOT ONNX-CUDA -- bundling the CUDA provider + cuDNN would add
# several GB. Result: a self-contained ~150 MB bundle that runs on any Windows 10/11
# machine with an NVIDIA GPU + driver, no install.
#
# IMPORTANT: point -OnnxRoot at the *CPU* ONNX Runtime (onnxruntime-win-x64-<ver>), NOT
# the gpu build -- otherwise the ~600 MB CUDA provider DLL gets bundled.
#
# Usage:  ./package.ps1                 (edit the default paths below, or pass them)
#         ./package.ps1 -Version 1.0.0
param(
  [string]$OnnxRoot  = "C:/dev/onnxruntime-win-x64-1.18.1",             # CPU ORT (not gpu)
  [string]$NcnnRoot  = "C:/dev/ncnn-20260526-windows-vs2022-shared/x64",
  [string]$MnnRoot   = "C:/dev/mnn_3.6.1_windows_x64_cpu_opencl_vulkan_avx512",
  [string]$OpenCVDir = "C:/dev/opencv/build/x64/vc16/lib",
  [string]$ModelsDir = "../models",                                     # source of model files
  [string[]]$Models  = @("v0.1-seg-n.onnx","v0.1-seg-n.mnn","v0.1-seg-n.metadata.yaml","v0.1-seg-n_ncnn",
                         "esmoe_n_visdrone_sim.onnx","esmoe_n_visdrone.mnn","esmoe_n_visdrone_ncnn"),
  [string]$Version   = "1.0.0",
  [string]$Generator = ""
)

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $root "build-dist"
$name  = "YOLO-Master-Windows-$Version"
$stage = Join-Path $root "dist/$name"

Write-Host "== 1/5  clean CPU-ORT release build ==" -ForegroundColor Cyan
if (Test-Path $build) { Remove-Item -Recurse -Force $build }
$gen = if ($Generator) { @("-G", $Generator) } else { @() }
cmake -S $root -B $build @gen -A x64 -DUSE_TRT=OFF `
  "-DONNXRUNTIME_ROOT=$OnnxRoot" "-DNCNN_ROOT=$NcnnRoot" "-DMNN_ROOT=$MnnRoot" "-DOpenCV_DIR=$OpenCVDir"
if ($LASTEXITCODE) { throw "cmake configure failed" }
cmake --build $build --config Release -j
if ($LASTEXITCODE) { throw "build failed" }

Write-Host "== 2/5  stage Release (exe + DLLs + assets) ==" -ForegroundColor Cyan
if (Test-Path (Join-Path $root "dist")) { Remove-Item -Recurse -Force (Join-Path $root "dist") }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item "$build/Release/*" $stage -Recurse -Force

Write-Host "== 3/5  strip stray CUDA / build artifacts (keep it small) ==" -ForegroundColor Cyan
$drop = @("onnxruntime_providers_cuda*.dll","onnxruntime_providers_tensorrt*.dll",
          "cudnn*.dll","cublas*.dll","cufft*.dll","cudart*.dll","curand*.dll","cusparse*.dll",
          "*.exp","*.lib","*.pdb")
foreach ($p in $drop) {
  Get-ChildItem $stage -Recurse -Filter $p -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

Write-Host "== 4/5  bundle default models ==" -ForegroundColor Cyan
$md = Join-Path $stage "models"
New-Item -ItemType Directory -Force -Path $md | Out-Null
foreach ($m in $Models) {
  $src = Join-Path $ModelsDir $m
  if (Test-Path $src) { Copy-Item $src $md -Recurse -Force }
  else { Write-Warning "model not found, skipping: $src" }
}
Copy-Item (Join-Path $root "DISTRIBUTION.md") (Join-Path $stage "README.txt") -Force

Write-Host "== 5/5  zip ==" -ForegroundColor Cyan
$zip = Join-Path $root "dist/$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip

$folderMB = "{0:N0}" -f ((Get-ChildItem $stage -Recurse | Measure-Object Length -Sum).Sum / 1MB)
$zipMB    = "{0:N0}" -f ((Get-Item $zip).Length / 1MB)
Write-Host "`nDone." -ForegroundColor Green
Write-Host "  folder: $stage  ($folderMB MB)"
Write-Host "  zip:    $zip  ($zipMB MB)"
