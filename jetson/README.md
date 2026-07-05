# Jetson Orin Nano (Super) Deployment Kit — YOLO-Master-EsMoE-N

aarch64 / JetPack deployment for the edge runner + TensorRT. Runs on a **Jetson Orin Nano (Super)**
with **JetPack 6.x** (Ubuntu 22.04, CUDA 12, TensorRT 10, cuDNN 9 — all preinstalled).

## Prerequisites

- JetPack 6.x flashed (CUDA / TensorRT / cuDNN come with it — you do **not** install them).
- The VisDrone model files placed in `jetson/models/`:
  - `esmoe_n_visdrone_sim.onnx`  (for TensorRT + the ONNX backend)
  - `esmoe_n_visdrone_ncnn/`     (for the ncnn backend, optional)
  - Get them by `scp` from your server (`/data/yolo-master-edge/models/`) or from the repo's GitHub Release.
- Internet (for `apt` build deps + fetching the ONNXRuntime aarch64 SDK).

## Quick start (in order)

```bash
git clone https://github.com/skywalker-lt/yolo-master-edge.git
cd yolo-master-edge/jetson
# put the model in models/  (scp esmoe_n_visdrone_sim.onnx here)

bash 00_setup.sh          # verify JetPack, set MAX power, install build deps
bash 10_trt_bench.sh      # TensorRT FP16 + INT8 engines + throughput  <- the headline number
bash 20_build_runner.sh   # build the C++ runner (aarch64) + run it
```

## What each step gives you

| Step | Output | Why |
|---|---|---|
| `00_setup.sh` | versions, MAXN power mode, `cmake`/OpenCV installed | reproducible perf (clocks locked) |
| `10_trt_bench.sh` | `*.engine` + **GPU FPS** (FP16, INT8) | the real payoff — Orin's tensor cores; INT8 finally *faster* than FP16 |
| `20_build_runner.sh` | `yolomaster_edge` (aarch64) + per-frame latency | the portable runner, same binary as Linux/Windows |

## Notes

- **4 GB Orin Nano:** the TensorRT *builder* is memory-hungry. Build **headless**
  (`sudo systemctl isolate multi-user.target`) and use a small workspace (`WORKSPACE=256 bash 10_trt_bench.sh`),
  or tactic profiling OOMs. Runtime inference is fine — the model is tiny; only the build is tight.

- **INT8 accuracy:** `trtexec --int8` here measures *speed* with dynamic ranges (not calibrated). For the
  <1% mAP INT8 model, build the engine from a calibrated model — keep the detection head in FP16
  (`--precisionConstraints`/`--layerPrecisions`), mirroring the mixed-precision recipe from `TECHNICAL_REPORT.md §3`.
- **Power:** `00_setup.sh` sets `nvpmodel -m 0` (MAXN) + `jetson_clocks`. Re-run after every reboot for stable numbers.
- **GPU via the C++ runner:** `20_build_runner.sh` builds the CPU path (functional + portable). GPU acceleration
  through the runner is a follow-up (ncnn-Vulkan or the ONNXRuntime TensorRT EP); `trtexec` already gives the GPU ceiling.
