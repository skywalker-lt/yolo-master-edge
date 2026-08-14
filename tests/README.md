# tests

Every test and verification script in one place. Nothing here ships in the bundles.

| script | platform | what it checks |
|---|---|---|
| `run_tests.sh` | Linux x64 / Jetson | 18-test CLI robustness battery (sources, backends, metadata, slicing, CW-NMS, label export, output collisions). `BIN=... ONNX=... ./tests/run_tests.sh` overrides the defaults. |
| `validate_mixture.py` | Linux | v26.08 mixture harness (MoA / MoT / MoLoRA): raw eager-vs-ORT parity, CLI-boxes-in-decode match, feature smoke, MNN parity when a sibling `.mnn` exists. Run from the repo root. |
| `mnn_parity.py` | Linux | standalone MNN-vs-ONNX raw tensor parity + latency for a single model. |
| `coreml_parity_mac.py` | macOS | CoreML parity for the mixture `.mlpackage` exports; ships inside `coreml-mixture-kit.tar.gz` with precomputed inputs and references. |
| `verify_eval_map_cuda.sh` | Linux + CUDA | `eval_map.py` on the C++ CUDA runner's predictions vs `ultralytics val` references (EsMoE-N VisDrone). |
| `tri_backend_map_parity.py` | Linux | 50-image mAP parity across PyTorch / ONNX-sim / ncnn for EsMoE-N VisDrone; also restores ONNX metadata after simplification. |

The last two carry absolute paths into `/data/YOLO-Master` research checkpoints; they are
temporary single-purpose verifiers kept for reproducibility, not portable suites.
