cmake_minimum_required(VERSION 3.16)
project(yolomaster_gui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT WIN32)
  message(FATAL_ERROR "yolomaster_gui targets Windows (Win32 + D3D11). Build cpp/ for the Linux CLI.")
endif()

# ---- paths ----
set(CPP_ROOT "${CMAKE_SOURCE_DIR}/../cpp")                # shared runtime (backends, decode, letterbox)
set(IMGUI    "${CMAKE_SOURCE_DIR}/third_party/imgui")     # vendored Dear ImGui + dx11/win32 backends

# SDK roots — override on the command line (see build.ps1):
#   -DONNXRUNTIME_ROOT=C:/dev/onnxruntime -DNCNN_ROOT=C:/dev/ncnn -DMNN_ROOT=C:/dev/mnn -DOpenCV_DIR=C:/dev/opencv/build
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNXRuntime root (include/ + lib/)")
set(NCNN_ROOT        "" CACHE PATH "ncnn root (include/ + lib/ + bin/)")
set(MNN_ROOT         "" CACHE PATH "MNN root (include/ + lib/)")

option(USE_ORT  "Build ONNXRuntime backend" ON)
option(USE_NCNN "Build ncnn backend"        ON)
option(USE_MNN  "Build MNN backend"         ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)   # imgcodecs for cv::imread

# ---- sources ----
set(SRC
  src/main_win.cpp
  src/app.cpp
  ${CPP_ROOT}/src/common.cpp
  ${CPP_ROOT}/src/stb_impl.cpp
  ${IMGUI}/imgui.cpp
  ${IMGUI}/imgui_draw.cpp
  ${IMGUI}/imgui_tables.cpp
  ${IMGUI}/imgui_widgets.cpp
  ${IMGUI}/backends/imgui_impl_dx11.cpp
  ${IMGUI}/backends/imgui_impl_win32.cpp
)
set(INCS
  src ${CPP_ROOT}/include ${CPP_ROOT}/third_party
  ${IMGUI} ${IMGUI}/backends ${OpenCV_INCLUDE_DIRS}
)
set(LIBS ${OpenCV_LIBS} d3d11 dxgi comdlg32)
set(DEFS "")
set(RUNTIME_DLLS "")

if(USE_ORT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  list(APPEND SRC  ${CPP_ROOT}/src/ort_backend.cpp)
  list(APPEND INCS ${ONNXRUNTIME_ROOT}/include)
  list(APPEND LIBS ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
  file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll" "${ONNXRUNTIME_ROOT}/bin/*.dll")
  list(APPEND RUNTIME_DLLS ${ORT_DLLS})
  list(APPEND DEFS USE_ORT)
  message(STATUS "ONNXRuntime backend: ON (${ONNXRUNTIME_ROOT})")
else()
  message(WARNING "ONNXRuntime backend: OFF (set -DONNXRUNTIME_ROOT)")
endif()

if(USE_NCNN AND EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
  list(APPEND SRC  ${CPP_ROOT}/src/ncnn_backend.cpp)
  list(APPEND INCS ${NCNN_ROOT}/include ${NCNN_ROOT}/include/ncnn)
  find_library(NCNN_LIB ncnn PATHS ${NCNN_ROOT}/lib NO_DEFAULT_PATH)
  list(APPEND LIBS ${NCNN_LIB})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    list(APPEND LIBS OpenMP::OpenMP_CXX)
  endif()
  file(GLOB NCNN_DLLS "${NCNN_ROOT}/bin/*.dll" "${NCNN_ROOT}/lib/*.dll")
  list(APPEND RUNTIME_DLLS ${NCNN_DLLS})
  list(APPEND DEFS USE_NCNN)
  message(STATUS "ncnn backend: ON (${NCNN_ROOT})")
else()
  message(WARNING "ncnn backend: OFF (set -DNCNN_ROOT)")
endif()

if(USE_MNN AND EXISTS "${MNN_ROOT}/include/MNN/Interpreter.hpp")
  list(APPEND SRC  ${CPP_ROOT}/src/mnn_backend.cpp)
  list(APPEND INCS ${MNN_ROOT}/include)
  find_library(MNN_LIB MNN PATHS ${MNN_ROOT}/lib ${MNN_ROOT}/build ${MNN_ROOT}/build/Release NO_DEFAULT_PATH)
  if(NOT MNN_LIB)
    message(FATAL_ERROR "USE_MNN=ON but libMNN not found under ${MNN_ROOT}/{lib,build}")
  endif()
  list(APPEND LIBS ${MNN_LIB})
  file(GLOB MNN_DLLS "${MNN_ROOT}/lib/*.dll" "${MNN_ROOT}/bin/*.dll" "${MNN_ROOT}/build/*.dll" "${MNN_ROOT}/build/Release/*.dll")
  list(APPEND RUNTIME_DLLS ${MNN_DLLS})
  list(APPEND DEFS USE_MNN)
  message(STATUS "MNN backend: ON (${MNN_ROOT})")
else()
  message(WARNING "MNN backend: OFF (set -DMNN_ROOT)")
endif()

add_executable(yolomaster_gui WIN32 ${SRC})   # WIN32 -> WinMain, no console window
target_include_directories(yolomaster_gui PRIVATE ${INCS})
target_link_libraries(yolomaster_gui PRIVATE ${LIBS})
target_compile_definitions(yolomaster_gui PRIVATE ${DEFS} UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)

# bundle the MSVC runtime + backend DLLs next to the exe
set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
include(InstallRequiredSystemLibraries)
list(APPEND RUNTIME_DLLS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
file(GLOB OCV_DLLS
  "${OpenCV_DIR}/x64/*/bin/opencv_world*.dll"
  "${OpenCV_DIR}/../bin/opencv_world*.dll"
  "${OpenCV_DIR}/bin/opencv_world*.dll")
list(FILTER OCV_DLLS EXCLUDE REGEX "[0-9]d\\.dll$")   # drop the debug opencv_world*d.dll
foreach(dll ${RUNTIME_DLLS} ${OCV_DLLS})
  add_custom_command(TARGET yolomaster_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:yolomaster_gui>"
    COMMENT "copying ${dll}")
endforeach()
