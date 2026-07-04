# YOLO-Master-EsMoE-N — Edge Inference

Cross-platform C++ edge runner for YOLO-Master-EsMoE-N over **ONNXRuntime** and **NCNN**
backends (CPU + CUDA), with vertical-domain preprocessing (letterbox), tunable NMS, and a
universal CLI (auto-detects backend, class names, and imgsz from the model).

Verified vs PyTorch on VisDrone (548 val imgs): **mAP50 −0.10%, mAP50-95 −0.06%** (< 0.5%).

## Layout
```
cpp/            C++ runner (CMake): src/, include/, third_party/CLI11.hpp, run_tests.sh
scripts/        eval_map.py (mAP vs GT), quantize_int8.py, compare/quant helpers
models/         esmoe_n_visdrone_sim.onnx, esmoe_n_visdrone_ncnn/  (deploy models)
```

## Build — Linux
```bash
# deps: apt install cmake libopencv-dev ; SDKs (ORT + ncnn) under third_party/
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=../../third_party/onnxruntime \
  -DNCNN_ROOT=../../third_party/ncnn
make -j$(nproc)
```

## Build — Windows (VS 2022/2026 + "Desktop development with C++")
Install prebuilt SDKs and unzip to e.g. `C:\dev\` (OpenCV, onnxruntime-win-x64, ncnn-windows-vs2022).
From the **x64 Native Tools Command Prompt**:
```bat
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCV_DIR=C:/dev/opencv/build ^
  -DONNXRUNTIME_ROOT=C:/dev/onnxruntime ^
  -DNCNN_ROOT=C:/dev/ncnn
cmake --build . --config Release
```
The backend + OpenCV DLLs are auto-copied next to `yolomaster_edge.exe`.
If `opencv_world*.dll` isn't found at runtime, add `C:\dev\opencv\build\x64\vc16\bin` to PATH.

## Run
```bash
yolomaster_edge --model models/esmoe_n_visdrone_sim.onnx --source <img|dir|video|dataset.yaml> \
                --conf 0.25 --out out
# backend auto-detected (.onnx->ORT, ncnn dir->NCNN); classes/imgsz from model metadata
# --device cuda   (ONNX backend, ORT-GPU SDK)   --multi-label (mAP-parity mode, conf 0.001)
```

See `cpp/run_tests.sh` for the 16-test robustness battery.
