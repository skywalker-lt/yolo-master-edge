cmake_minimum_required(VERSION 3.22.1)
project(yolomaster_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The C++ inference core is reused verbatim from the repo's cpp/ (the same sources
# the Linux/Windows/Jetson runners compile). This CMakeLists lives at
# android/runtime/src/main/cpp/ -> the repo root is five levels up.
set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp)

# ---- ncnn (official Android prebuilt, per-ABI; tag 20260526 for .param/.bin parity) ----
if(NOT NCNN_ANDROID_ROOT)
  message(FATAL_ERROR "NCNN_ANDROID_ROOT is not set. Copy android/sdk-paths.example.properties "
                      "to android/sdk-paths.properties and point it at the ncnn Android SDK.")
endif()
set(ncnn_DIR "${NCNN_ANDROID_ROOT}/${ANDROID_ABI}/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

# ---- OpenCV (opencv-mobile Android: core + imgproc only) ----
if(NOT OPENCV_ANDROID_ROOT)
  message(FATAL_ERROR "OPENCV_ANDROID_ROOT is not set. See android/sdk-paths.example.properties.")
endif()
set(OpenCV_DIR "${OPENCV_ANDROID_ROOT}/sdk/native/jni")
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(yolomaster_ncnn SHARED
  jni_bridge.cpp
  ${CORE}/src/common.cpp
  ${CORE}/src/ncnn_backend.cpp
  ${CORE}/src/stb_impl.cpp
)

target_include_directories(yolomaster_ncnn PRIVATE
  ${CORE}/include
  ${CORE}/third_party
  ${OpenCV_INCLUDE_DIRS}
)

# Compile only the ncnn backend (ORT/MNN/TRT are desktop-only).
target_compile_definitions(yolomaster_ncnn PRIVATE USE_NCNN)

find_library(log-lib log)

# Linking the `ncnn` package target propagates its include dirs (net.h/cpu.h/gpu.h)
# and Vulkan/glslang deps. jnigraphics is for AndroidBitmap_* pixel access.
target_link_libraries(yolomaster_ncnn
  ncnn
  ${OpenCV_LIBS}
  jnigraphics
  android
  ${log-lib}
)
