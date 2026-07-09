#!/usr/bin/env bash
# Build the C++ runner on macOS with ONNX Runtime (CPU + CoreML EP), to A/B --device cpu vs coreml.
# The CoreML EP runs supported subgraphs on ANE/GPU and the rest on CPU (this MoE+attention model
# fragments, so measure before trusting it). Needs: Xcode Command Line Tools + Homebrew.
#   bash mac/build_ort_macos.sh
set -e
cd "$(dirname "$0")"; ROOT="$(cd .. && pwd)"
ORT_VER="${ORT_VER:-1.20.1}"
ORT_DIR="$ROOT/third_party/onnxruntime-osx-arm64-$ORT_VER"

command -v brew >/dev/null || { echo "install Homebrew first: https://brew.sh"; exit 1; }
brew list opencv >/dev/null 2>&1 || brew install opencv

# ONNX Runtime osx-arm64 release ships libonnxruntime.dylib with the CoreML EP built in
if [ ! -d "$ORT_DIR" ]; then
  mkdir -p "$ROOT/third_party"
  url="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-osx-arm64-$ORT_VER.tgz"
  echo "downloading $url"
  curl -L "$url" | tar xz -C "$ROOT/third_party"
fi

cd "$ROOT/cpp"; rm -rf build_mac && mkdir build_mac && cd build_mac
cmake .. -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON -DUSE_NCNN=OFF -DUSE_TRT=OFF -DUSE_ORT=ON \
  -DONNXRUNTIME_ROOT="$ORT_DIR" -DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4" \
  2>&1 | grep -iE "backend:|error" || true
make -j"$(sysctl -n hw.ncpu)" 2>&1 | grep -iE "error|Built target" | tail -2

BIN="$ROOT/cpp/build_mac/yolomaster_edge"
ONNX="$ROOT/mac/esmoe_n_visdrone_sim.onnx"
echo
echo "built: $BIN"
echo "== A/B on a directory of images (>=50 for a stable mean) =="
echo "  $BIN --model $ONNX --source <val_dir> --device cpu    --no-save --quiet"
echo "  $BIN --model $ONNX --source <val_dir> --device coreml --no-save --quiet"
echo "compare [summary] infer= / model-FPS. Notes:"
echo "  - CoreML's FIRST run is slow (it compiles subgraphs); take the second run's number."
echo "  - startup prints ep=CoreML if the EP loaded (else ep=CPU fallback)."
echo "  - to see how many nodes CoreML actually took (the fragmentation tell), rebuild the ORT"
echo "    session at VERBOSE log level; a low count in many partitions = CoreML EP not worth it."
