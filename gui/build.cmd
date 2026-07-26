@echo off
setlocal EnableDelayedExpansion
rem ============================================================================
rem  Build the YOLO-Master Windows Runner (GUI).
rem  Batch equivalent of build.ps1, for when PowerShell execution policy blocks
rem  .ps1 files. Run from a "x64 Native Tools Command Prompt for VS", or any
rem  shell with cmake + MSVC on PATH.
rem
rem  SDK paths come from YOUR config, not from this script:
rem     copy sdk-paths.example.cmd sdk-paths.cmd   (then edit it)
rem  Environment variables of the same names are used if that file is absent.
rem
rem  Usage:  build.cmd            configure + build Release
rem          build.cmd run        build then launch
rem          build.cmd clean      wipe build\ first, then build
rem ============================================================================

if exist "%~dp0sdk-paths.cmd" (
  call "%~dp0sdk-paths.cmd"
) else (
  echo [info] no sdk-paths.cmd - falling back to environment variables
  echo        ^(copy sdk-paths.example.cmd sdk-paths.cmd to set them permanently^)
)

if "%OPENCV_DIR%"=="" (
  echo ERROR: OPENCV_DIR is not set. OpenCV is required.
  echo        copy sdk-paths.example.cmd sdk-paths.cmd, then edit it.
  exit /b 1
)

rem --- validate each root, and drop any that is not really there --------------
set "ARGS="
call :check_dir  OPENCV_DIR   "%OPENCV_DIR%"    OpenCVConfig.cmake                 OpenCV_DIR
if errorlevel 1 exit /b 1
call :check_dir  ONNX_ROOT    "%ONNX_ROOT%"     include\onnxruntime_cxx_api.h      ONNXRUNTIME_ROOT
call :check_dir  NCNN_ROOT    "%NCNN_ROOT%"     include\ncnn\net.h                 NCNN_ROOT
call :check_dir  MNN_ROOT     "%MNN_ROOT%"      include\MNN\Interpreter.hpp        MNN_ROOT
call :check_dir  CUDNN_DIR    "%CUDNN_DIR%"     cudnn64_9.dll                      CUDNN_DIR
call :check_dir  CUDA_BIN_DIR "%CUDA_BIN_DIR%"  cudart64_12.dll                    CUDA_BIN_DIR

set "ROOT=%~dp0"
set "BUILD=%ROOT%build"

if /i "%~1"=="clean" if exist "%BUILD%" rmdir /s /q "%BUILD%"

rem the linker cannot overwrite a running exe -- close it first
taskkill /IM yolomaster_gui.exe /F >nul 2>&1

echo.
echo == configure ==
cmake -S "%ROOT%." -B "%BUILD%" -A x64 %ARGS%
if errorlevel 1 exit /b 1

echo.
echo == build ==
cmake --build "%BUILD%" --config Release -j
if errorlevel 1 exit /b 1

echo.
echo Built: %BUILD%\Release\yolomaster_gui.exe
if /i "%~1"=="run" start "" "%BUILD%\Release\yolomaster_gui.exe"
exit /b 0

rem ---------------------------------------------------------------------------
rem  :check_dir <VAR_NAME> <path> <marker file> <cmake var>
rem  Empty  -> skip quietly (that backend is simply not built).
rem  Set but wrong -> report and skip, so a typo can't silently disable a backend
rem  without you noticing. OpenCV is fatal because the build needs it.
rem ---------------------------------------------------------------------------
:check_dir
set "_name=%~1"
set "_path=%~2"
set "_marker=%~3"
set "_cmakevar=%~4"
if "%_path%"=="" (
  if /i "%_name%"=="OPENCV_DIR" exit /b 1
  echo   [skip] %_name% not set
  exit /b 0
)
if not exist "%_path%\%_marker%" (
  echo   [WARN] %_name%=%_path%
  echo          expected to find %_marker% there - ignoring this path.
  if /i "%_name%"=="OPENCV_DIR" (
    echo          OpenCV is required; fix the path and re-run.
    exit /b 1
  )
  exit /b 0
)
echo   [ok]   %_name% = %_path%
set "ARGS=%ARGS% "-D%_cmakevar%=%_path%""
exit /b 0
