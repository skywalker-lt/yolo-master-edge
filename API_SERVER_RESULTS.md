# API Server vs Bare Runtime: COCO val2017 on an L40S

All numbers in this document are MEASURED on one machine in one session unless a row is
labelled REFERENCE. Nothing here is a claim about upstream YOLO-Master; the models are the
released v0.1-N and EsMoE-N COCO checkpoints exported by this repo (A3 exports) plus the
MoEPruner-pruned v0.1-N from project03.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA L40S 46 GB, driver 580.126.09, max SM clock 2520 MHz |
| CPU | AMD EPYC 9374F, 128 vCPUs visible, container quota 13.6 cores (`cpu.max` 1360000/100000) |
| RAM | 1.5 TB (container) |
| OS / toolchain | Ubuntu 22.04.5, g++ 11.4, CMake 4.4, Ninja |
| CUDA runtime | 12.9.79 (TensorRT build target); pod toolkit 12.4 |
| TensorRT | 10.16.1.11 (+cuda12.9), engines built by `yolomaster_server`/`yolomaster_edge` from ONNX |
| ONNX Runtime | 1.20.1 GPU build (CUDA EP), cuDNN 9.26 |
| ncnn | 20260526 shared build for Ubuntu 22.04, x86-64, OpenMP, no Vulkan |
| MNN | 3.6.1 built from source (AVX512 CPU build; CUDA backend build sm89) |
| HTTP engine | uWebSockets v20.80.0 + uSockets (epoll, no SSL, no zlib) |
| dataset | COCO val2017, 5000 images (4952 with labels), YOLO-format labels |
| scoring | `scripts/eval_map.py` (ultralytics 8.4.101 `DetMetrics` matcher, IoU 0.50:0.95), conf 0.001, IoU 0.7, max_det 300 |

Container CPU quota matters: the pod exposes 128 vCPUs but `cpu.max` caps it at 13.6 cores.
CPU backends therefore run 6 threads per worker, and the concurrency-8 API cells use 2 workers
(12 threads). Numbers for ncnn and MNN on CPU are for that budget, not for a 32-core host.

## Models

| id | source | fp32 | fp16 |
|---|---|---|---|
| v01n | `runs/a3/v01n/YOLO-Master-v0.1-N.onnx` (A3 P0 export, opset 17, static 640) | ONNX / TRT / ncnn (`models/v0.1-n_ncnn`, dense SDPA export) / MNN | fp16 ONNX (routing subgraph kept fp32), TRT fp16 flag, MNN fp16 weights |
| v01n-pruned | `runs/project03/trt/v01-n-coco-pruned-surgery/YOLO-Master-v0.1-N_pruned_t0.10.onnx` (MoEPruner t=0.10 + BN-router surgery) | ONNX / TRT / ncnn (dense SDPA export of `pod-prune/v01-n-coco/..._pruned_t0.10.pt`) / MNN | not part of the grid |
| esmoen | `runs/a3/esmoen/YOLO-Master-EsMoE-N.onnx` | ONNX / TRT / ncnn (`scripts/export_ncnn_dense.py` on the COCO checkpoint) / MNN | fp16 ONNX, TRT fp16 flag, MNN fp16 weights |

fp16 preparation (`scripts/server/prepare_models.py`): `onnxconverter_common` with fp32 I/O,
stale value_info stripped and shapes re-inferred, the `/routing/` subgraph and TopK/Gather ops
pinned to fp32 so expert selection does not depend on fp16 rounding. ORT-CPU probe drift on one
COCO image: v01n 3.1e-3 relative (101 vs 100 candidates at conf 0.25), esmoen 1.3e-2 relative
(102 vs 102). MNN fp16 = `MNNConvert --fp16` weights; on CPU MNN computes in fp32 regardless.
ncnn has no fp16 arithmetic on x86, so the ncnn fp16 cells are marked n/a rather than reported
from an fp32 path.

## Method

Every cell = one model file on one backend, three ways:

1. `cli`: `yolomaster_edge -s <val2017> --save-txt --no-save --quiet --warmup 20` (the bare
   runtime, sequential, stb JPEG decode, same pre/post code).
2. `api-c1`: `yolomaster_server` with one worker, `clients/python/bench_client.py --concurrency 1`
   (one request in flight, client-observed latency per image, 20 warm-up requests excluded).
3. `api-c8`: eight requests in flight; GPU models keep one worker, CPU models use two.

The API dumps come from the JSON response boxes at full float precision; parity against the CLI
txt files is checked numerically per detection (`scripts/server/parity_txt.py`, 0.01 px).
mAP is computed on both dump sets. "API overhead" = client p50 minus server (queue + total) p50,
i.e. HTTP framing, body copy and serialization.

## Results

RESULTS_TABLE_PLACEHOLDER

## Reading the numbers

READING_PLACEHOLDER

## Reproduce

```
bash deploy/pod/bootstrap_l40s.sh && bash deploy/pod/build_l40s.sh
bash deploy/pod/prepare_models_pod.sh
bash scripts/server/run_comparison.sh /data/results/api_bench      # about 6 hours on the pod
python scripts/server/summarize_comparison.py /data/results/api_bench
```

Raw per-cell artifacts (CLI logs, bench JSON, server stats, txt dumps, scores) live under
`results/api_bench/` on mdb (`api_bench.tgz`); they are not committed.
