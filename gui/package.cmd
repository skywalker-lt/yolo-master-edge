@echo off
setlocal EnableDelayedExpansion
rem ============================================================================
rem  Package the YOLO-Master Windows Runner into a self-contained zip.
rem  Batch equivalent of package.ps1 - use this when PowerShell execution policy
rem  blocks .ps1 files. Needs: cmake + MSVC on PATH (run from a VS Dev Prompt).
rem
rem  Usage:  package.cmd [version]
rem          package.cmd 1.0.0
rem
rem  Edit the SDK paths below to match your machine. IMPORTANT: ONNX_ROOT must be
rem  the *CPU* ONNX Runtime -- the gpu package would add ~600MB of CUDA provider.
rem ============================================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1.0.0"

set "ONNX_ROOT=C:/dev/onnxruntime-win-x64-1.18.1"
set "NCNN_ROOT=C:/dev/ncnn-20260526-windows-vs2022-shared/x64"
set "MNN_ROOT=C:/dev/mnn_3.6.1_windows_x64_cpu_opencl_vulkan_avx512"
set "OPENCV_DIR=C:/dev/opencv/build/x64/vc16/lib"
set "MODELS_DIR=..\models"

set "ROOT=%~dp0"
set "BUILD=%ROOT%build-dist"
set "NAME=YOLO-Master-Windows-%VERSION%"
set "STAGE=%ROOT%dist\%NAME%"

echo == 1/5  clean CPU-ORT release build ==
if exist "%BUILD%" rmdir /s /q "%BUILD%"
cmake -S "%ROOT%." -B "%BUILD%" -A x64 -DUSE_TRT=OFF ^
  "-DONNXRUNTIME_ROOT=%ONNX_ROOT%" "-DNCNN_ROOT=%NCNN_ROOT%" ^
  "-DMNN_ROOT=%MNN_ROOT%" "-DOpenCV_DIR=%OPENCV_DIR%"
if errorlevel 1 goto :fail
cmake --build "%BUILD%" --config Release -j
if errorlevel 1 goto :fail

echo == 2/5  stage Release (exe + DLLs + assets) ==
if exist "%ROOT%dist" rmdir /s /q "%ROOT%dist"
mkdir "%STAGE%"
xcopy "%BUILD%\Release\*" "%STAGE%\" /E /I /Y >nul
if errorlevel 1 goto :fail

echo == 3/5  strip stray CUDA / build artifacts ==
for %%P in (onnxruntime_providers_cuda*.dll onnxruntime_providers_tensorrt*.dll ^
            cudnn*.dll cublas*.dll cufft*.dll cudart*.dll curand*.dll cusparse*.dll ^
            *.exp *.lib *.pdb) do (
  del /q /s "%STAGE%\%%P" >nul 2>&1
)

echo == 4/5  bundle default models ==
mkdir "%STAGE%\models" 2>nul
for %%M in (v0.1-seg-n.onnx v0.1-seg-n.mnn v0.1-seg-n.metadata.yaml ^
            esmoe_n_visdrone_sim.onnx esmoe_n_visdrone.mnn) do (
  if exist "%ROOT%%MODELS_DIR%\%%M" (
    copy /y "%ROOT%%MODELS_DIR%\%%M" "%STAGE%\models\" >nul
  ) else (
    echo   [warn] model not found, skipping: %%M
  )
)
for %%D in (v0.1-seg-n_ncnn esmoe_n_visdrone_ncnn) do (
  if exist "%ROOT%%MODELS_DIR%\%%D" xcopy "%ROOT%%MODELS_DIR%\%%D" "%STAGE%\models\%%D\" /E /I /Y >nul
)
rem the app auto-loads models\v0.1-seg-n.onnx -- never ship a bundle without it
if not exist "%STAGE%\models\v0.1-seg-n.onnx" (
  echo ERROR: default model models\v0.1-seg-n.onnx missing ^(looked in %MODELS_DIR%^)
  goto :fail
)
copy /y "%ROOT%DISTRIBUTION.md" "%STAGE%\README.txt" >nul

echo == 5/5  zip ==
rem tar.exe (bsdtar) ships with Windows 10 1803+ and writes zip via -a
pushd "%ROOT%dist"
if exist "%NAME%.zip" del /q "%NAME%.zip"
tar -a -c -f "%NAME%.zip" "%NAME%"
popd
if not exist "%ROOT%dist\%NAME%.zip" goto :fail

echo.
echo Done.
echo   folder: %STAGE%
echo   zip:    %ROOT%dist\%NAME%.zip
for %%F in ("%ROOT%dist\%NAME%.zip") do echo   size:   %%~zF bytes
exit /b 0

:fail
echo.
echo PACKAGING FAILED.
exit /b 1
