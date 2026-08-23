⚠️因使用了错误的Upstream基线而非指定的`acce839c7e89`统一公共基线，本README中的smoke test结果及相关结论作废。目前正在基于统一公共基线重跑smoke test/准入测试

# A3: Real compatibility of dynamic routing, Softmax, top-k and heterogeneous experts under ONNX / TensorRT / INT8

Branch `dev/a3-smoke`, based on `dev/project03`. This README covers the **8.24 entry check only**:
the minimal `yolo.export` smoke, the environment matrix and backend versions, one failure log and one
success log, and the entry table below. The existing edge runtime is summarized in section 10.
Chinese version: `README_CN.md`.

---

## 1. Scope

**Already shipped (not claimed by this topic):** this repository provides the multi-backend runtime
(ONNX Runtime / NCNN / MNN / TensorRT / CoreML, v1.1.0), native TensorRT deployment on Jetson Orin, and
the project03 INT8 study on the **pruned** v0.1-N (TensorRT implicit PTQ vs explicit Q/DQ ladders, the
calibration coverage law, the sensitive-layer set, a negative QAT result). The `ultralytics` exporter
itself has no MoE awareness at all.

**Gaps this topic closes (closing the loop, rigorous re-measurement, tooling):**

| Gap | What this branch does |
|---|---|
| The exporter has no routing policy: the sparse/dense switch happens silently inside each MoE module's `is_in_onnx_export()` branch | `scripts/a3/smoke_export.py`: an explicit preserve / declared-dense / reject policy, decided and logged BEFORE export; `--dynamic` is rejected outright (the failure log) |
| No PyTorch-vs-backend accuracy baseline on the released (unpruned) weights | PyTorch / ONNX Runtime / TensorRT fp32 / fp16 mAP deltas under one protocol (`backend_val.py` + `quantize_trt.py`) |
| INT8 conclusions existed only on pruned models | Implicit INT8 and explicit Q/DQ re-run on the **released, unpruned** v0.1-N and EsMoE-N (COCO) |
| Code references cited with stale line numbers | Every policy reference is re-grepped at run time against the locked commit and written into `a3/results/smoke_*.json` |

Locked commits: YOLO-Master `3ea98305a8449d8d9f4a00845e26ff9d8bf3b66e` (2026-08-01); this repository per
`a3/env/matrix.md`.

---

## 2. 8.24 entry check

| 序号 (No.) | 姓名 (Name) | 第一志愿 (First choice) | 环境安装 (Environment) | 基线/最小任务 (Baseline / minimal task) | 复现命令 (Reproduction) | 配置文件 (Config) | 完整日志 (Full logs) | 结果证据 (Evidence) | 设计说明 (Design) | 风险与降级 (Risks / fallbacks) |
|---|---|---|---|---|---|---|---|---|---|---|
|  | Thomas | A3: real compatibility, precision deltas and routing-decision drift of dynamic routing / Softmax / top-k / heterogeneous experts under ONNX / TensorRT / INT8 | `scripts/a3/setup_l4.sh` -> `a3/env/matrix.md`, `a3/env/pip-freeze.txt`, log `a3/logs/setup_l4.log` | P0: `yolo.export` ONNX smoke (v0.1-N, EsMoE-N) with ORT parity; PyTorch vs ORT / TRT fp32 / fp16 deltas; plus the INT8 ladder (implicit + explicit Q/DQ) | Section 4 (`scripts/a3/run_all_l4.sh`) | `a3/config/coco-a3.yaml`, `a3/config/sensitive_sets.md` | success: `a3/logs/v01n_smoke.log`, `a3/logs/esmoen_smoke.log`; failure (policy rejection): `a3/logs/v01n_smoke_dynamic_reject.log`; ladders: `a3/logs/*_trt_ladder.log`, `*_trt_stempair.log`, `*_int8_qdq.log`, `*_backend_val.log` | `a3/results/*.json`, `a3/results/ladder_*.csv`, section 6 | Section 7 | Section 8 |

---

## 3. Environment matrix

The L4 box starts as a bare `torch 2.4.1+cu124` container; `scripts/a3/setup_l4.sh` installs every
backend at pinned versions (TensorRT `tensorrt-cu12==10.13.3.9`, the same version behind the project03
A100 conclusions; ONNX Runtime 1.20.2; modelopt 0.27.1; `ultralytics` as an editable install of the
locked YOLO-Master commit) and asserts after every step that torch was not replaced. `pip freeze` is
captured again after the whole chain and diffed, proving the export never triggered ultralytics'
auto-install (`YOLO_AUTOINSTALL=False`).

| item | value |
|---|---|
| captured_utc | 2026-08-23T02:24:49+00:00 |
| host | fea95d9954c8 |
| os | Ubuntu 22.04.5 LTS |
| kernel | 6.8.0-50-generic |
| cpu | AMD EPYC 7702 64-Core Processor |
| ram_gb | 503 |
| gpu | NVIDIA L4, 23034 MiB, 580.126.20 |
| cuda_driver_max | CUDA Version: 13.0 |
| cuda_toolkit_nvcc | Build cuda_12.4.r12.4/compiler.34097967_0 |
| python | 3.11.10 |
| torch | 2.4.1+cu124 |
| torch_cuda | 12.4 |
| cudnn | 90100 |
| torchvision | 0.19.1+cu124 |
| numpy | 2.4.6 |
| onnx | 1.17.0 |
| onnxslim | 0.1.96 |
| onnxruntime | 1.20.2 |
| onnxruntime_providers | TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider |
| tensorrt | 10.13.3.9 |
| modelopt | 0.27.1 |
| pycocotools | installed (no __version__) |
| opencv | 5.0.0 |
| ultralytics | 8.3.240 |
| ultralytics_file | /data/YOLO-Master/ultralytics/__init__.py |
| yolo_master_repo | `3ea98305a844` (2026-08-01, littlemod-moa, dirty: scripts/reproduce/bench_coco_latency.py) |
| edge_repo | `fdf7894549aa` (2026-08-23, dev/a3-smoke) |

---

## 4. Reproduction

```bash
# on the L4 (any CUDA box with /data mounted)
cd /data/yolo-master-edge && git switch dev/a3-smoke
bash scripts/a3/setup_l4.sh                                        # 1. environment (idempotent)
. /root/a3venv/bin/activate
nohup bash scripts/a3/run_all_l4.sh > a3/logs/run_all.log 2>&1 &   # 2. the whole chain
# STAGES=env,smoke,ladder,qdq,backend  (subset with e.g. STAGES=smoke)
bash scripts/a3/run_followup_l4.sh                                 # 3. stem-pair rungs, re-runs
```

Each stage on its own:

```bash
python scripts/a3/env_matrix.py --out a3/env
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-v0.1-N.pt \
    --out runs/a3/v01n --data a3/config/coco-a3.yaml \
    --json a3/results/smoke_v01n.json --log a3/logs/v01n_smoke.log
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-EsMoE-N.pt \
    --out runs/a3/esmoen --data a3/config/coco-a3.yaml \
    --json a3/results/smoke_esmoen.json --log a3/logs/esmoen_smoke.log
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-v0.1-N.pt \
    --out runs/a3/v01n-dynamic --dynamic ...            # exit 2: the failure log
python scripts/a3/quantize_trt.py --model <pt> --data a3/config/coco-a3.yaml --out runs/a3/<m>/trt \
    --calib-images /data/datasets/coco/images/train2017 --calib-n 1024 \
    --modes fp32,fp16,int8 --max-rounds 0 --tf32-baseline off
python scripts/a3/quantize_trt.py --model runs/a3/<m>/trt/<stem>.onnx ... --modes int8 --pin-modules 0,1
python scripts/a3/qat_moe.py --model <pt> --data a3/config/coco-a3.yaml --out runs/a3/<m>/qdq \
    --calib-n 1024 --skip-train                         # calibrate-only explicit Q/DQ
python scripts/a3/backend_val.py --model <pt> --onnx runs/a3/<m>/<stem>.onnx \
    --data a3/config/coco-a3.yaml --json a3/results/backend_<m>.json
```

---

## 5. Config

- `a3/config/coco-a3.yaml`: COCO paths, `train: images/train2017` (calibration only),
  `val: images/val2017` (evaluation).
- `a3/config/sensitive_sets.md`: calibration policy (>= 1024 train-split images, entropy for implicit,
  explicit per-channel Q/DQ) and the sensitive set kept in FP16 (stem pair + routers + DFL).
- One evaluation protocol everywhere: ultralytics `.val`, imgsz 640, batch 1, full val2017 (5000
  images); PyTorch, ONNX Runtime and TensorRT engines go through the same path, so every delta is a
  plain subtraction.

---

## 6. Evidence

All accuracies are ultralytics `.val` mAP50-95 (COCO val2017, 5000 images, imgsz 640, batch 1).
Latencies are TensorRT model-only medians of 200 executions on the L4; PyTorch / ORT rows carry no
latency column. Raw data in `a3/results/`, per-layer precision audits in the matching logs.

### 6.1 P0: `yolo.export` smoke and PyTorch-vs-backend accuracy deltas

| Model (released, unpruned) | Backend | mAP50-95 | Delta vs PyTorch (AP) | Latency ms | Size MB | Source |
|---|---|---|---|---|---|---|
| v0.1-N (OptimizedMOEImproved x3, E=4/8/16, k=2) | PyTorch (repaired) | **0.4292** | 0 | - | 15.1 (.pt) | `backend_v01n.json` |
| | ONNX Runtime CUDA EP (default TF32) | 0.4286 | -0.0006 | - | 30.5 (.onnx) | `backend_v01n.json` |
| | TensorRT fp32 (TF32 off) | 0.4287 | -0.0005 | 3.898 | 42.6 | `ladder_v01n.csv` |
| | TensorRT fp16 | 0.4285 | -0.0007 | 1.935 (-50.4%) | 21.3 | `ladder_v01n.csv` |
| EsMoE-N (ES_MOE x4, E=3, k=3) | PyTorch (dense == sparse) | **0.4270** | 0 | - | 5.7 (.pt) | `backend_esmoen.json` |
| | ONNX Runtime CUDA EP (default TF32) | 0.4267 | -0.0004 | - | 11.1 (.onnx) | `backend_esmoen.json` |
| | TensorRT fp32 (TF32 off) | 0.4267 | -0.0004 | 2.982 | 22.0 | `ladder_esmoen.csv` |
| | TensorRT fp16 | 0.4268 | -0.0003 | 1.720 (-42.3%) | 11.4 | `ladder_esmoen.csv` |

Smoke verification (`smoke_*.json`):

| Model | Policy decision | Routing ops in ONNX | Expert convs module / graph | ORT fp32 parity (scale-normalized error / top-100 anchor agreement) | ORT default TF32 |
|---|---|---|---|---|---|
| v0.1-N | 3 blocks `preserve` (exact gather) | TopK 3, GatherElements 6, Softmax 12, no If/Loop/NonZero | 56 / 56 | 3.9e-6 / 100% | max 1.38 px shift, 99.9% |
| EsMoE-N | 4 blocks `dense`, `semantic_change=false` (k==E, no top-k / threshold) | TopK 0, Softmax 13, no If/Loop/NonZero | 24 / 24 | 5.4e-6 / 100% | max 2.98 px shift, 100% |
| v0.1-N `--dynamic` | **rejected before export, exit 2** (`v01n_smoke_dynamic_reject.log`) | - | - | - | - |

### 6.2 INT8 ladder (implicit PTQ for contrast + the explicit Q/DQ delivery recipe), 1024 train2017 calibration images

| Model | INT8 method | mAP50-95 | Delta vs fp32 (AP) | Latency ms | Size MB | Engine layer precisions (Int8 / Half / Float) |
|---|---|---|---|---|---|---|
| v0.1-N | implicit entropy, head + routers pinned FP16 | 0.3754 | -5.33 | 1.944 | 17.1 | 189 / 224 / 14 |
| v0.1-N | implicit entropy, + stem pair (model.0/1) | 0.3955 | -3.32 | 1.965 | 17.0 | `v01n_trt_stempair.log` |
| v0.1-N | **explicit Q/DQ, calibrate-only (modelopt)** | **0.4164** | **-1.23** | 2.231 | 21.0 | 196 / 176 / 85 |
| EsMoE-N | implicit entropy, head + routers pinned FP16 | 0.3606 | -6.61 | 1.843 | 12.9 | 165 / 147 / 60 |
| EsMoE-N | implicit entropy, + stem pair | 0.3752 | -5.15 | 1.917 | 13.0 | `esmoen_trt_stempair.log` |
| EsMoE-N | **explicit Q/DQ, calibrate-only (modelopt)** | **0.4121** | **-1.49** | 2.141 | 11.5 | 174 / 168 / 112 |

How to read it:
- At N scale INT8 is **not faster than fp16** for either family (v0.1-N 1.94 vs 1.94 ms, EsMoE-N 1.84 vs
  1.72 ms), matching the project03 verdict on the pruned models: on a TensorRT GPU at this width the
  deployment precision is fp16.
- The implicit-PTQ loss ordering is the same as in the pruned study: surgical < stem pair < explicit Q/DQ;
  explicit Q/DQ is the only path that keeps the loss under 2 AP (pruned v0.1-N -0.80, unpruned -1.23:
  all 4/8/16 experts now enter the INT8 surface).
- **A routing-specific finding the pruned study could not expose** (it had k == E): calibrating the
  unpruned v0.1-N on the eager sparse path leaves experts that were never routed to with no activation
  statistics, while the exported graph computes every expert; modelopt then asserts at export,
  "Quantizer has not been calibrated". Fix: calibrate on the export-equivalent dense path (the same
  `is_in_onnx_export` mock the exporters use) and audit uncalibrated quantizers explicitly (this run:
  360 calibrated, 0 disabled). This is a concrete instance of the topic's question: **calibration
  coverage has to follow the exported graph, not the eager routing semantics.**
- ONNX Runtime's CUDA EP enables TF32 convolutions by default: box coordinates shift by up to
  1.4 / 3.0 px, yet mAP moves only -0.0006 / -0.0004. The smoke's parity gate therefore compares
  fp32 to fp32 (`use_tf32=0`) and records the TF32 result separately.

---

## 7. Design notes

### 7.1 Dynamic-routing export policy (explicit; no silent semantic change)

The exporter (`ultralytics/engine/exporter.py`) contains no MoE logic; each MoE family chooses its
export branch inside its own `forward` via `torch.onnx.is_in_onnx_export()`. `smoke_export.py` turns
that into an explicit decision before calling the very same `YOLO.export`, and stores the evidence
(file, line numbers re-grepped at run time, commit) in the result JSON:

| MoE family | Export-branch behaviour at the locked commit | Policy | Semantic change |
|---|---|---|---|
| `OptimizedMOEImproved` (v0.1 family) | computes all experts, then `torch.gather` on the top-k (`selected = torch.gather(all_outs, ...)`) | `preserve` | none (exact top-k) |
| `ES_MOE` (EsMoE family) | forced through `_dense_forward`: softmax-weighted sum over all experts, skipping `_sparse_forward`'s top-k / `dynamic_threshold` | `dense`; no semantic change only when `k == E` and no top-k / threshold is active (true for the released COCO weights: E=3, k=3, `use_top_k=False`, no `dynamic_threshold`) | otherwise it must be declared with `--routing dense`, and the eager-sparse vs eager-dense AP delta is reported |
| other families | no verified export branch | `reject` | - |
| any family + `--dynamic` | the gather branch bakes B/H/W as Python ints into `view/expand` | **rejected before export** (exit 2) | a dynamic-axes graph is silently wrong at other shapes |

Precondition: torch < 2.9 (`ultralytics/utils/export/engine.py` forces `dynamo=False` for torch >= 2.4;
the dynamo exporter bypasses the guards above and silently drops experts).

### 7.2 Post-export verification (three layers)
1. ONNX op histogram: TopK / Gather / GatherElements / Softmax counts; any `If / Loop / NonZero` fails
   the run (it means a guard did not take effect).
2. Expert-conv count: `Conv` nodes under `/experts` in the graph vs `nn.Conv2d` under `*.experts.*` in
   the module.
3. ORT parity on real val images: ONNX Runtime (fp32, `use_tf32=0`) vs **eager PyTorch keeping its native
   sparse semantics**, reporting max-abs, scale-normalized error and top-100 anchor agreement.

### 7.3 Two repairs the released weights need (or every conclusion is wrong)
- v0.1-N: the checkpoint predates the `add_residual` attribute; the compat shim defaults it to True,
  class confidence collapses to ~0.04 and mAP50-95 drops from 0.43 to 0.007. Every script forces it to
  False where the attribute is absent (reusing `scripts/project03/diagnose_moe._fix_add_residual`).
  The same pickles also carry instance attributes that are read-only properties on today's class
  (`aux_loss`), stripped at load (`_strip_property_shadows`).
- EsMoE-N: evaluation and export both use the dense path (`use_sparse_inference=False`), consistent
  with the decision in 7.1.

### 7.4 Why the INT8 method is "calibrate-only explicit Q/DQ"
The project03 results on the pruned v0.1-N (COCO, A100, TensorRT 10.13.3.9): implicit PTQ cannot
quantize the MoE blocks (the routing gather/expand breaks the INT8 chain and TensorRT silently falls
back to FP16); explicit per-channel Q/DQ reaches -0.80 AP; three QAT epochs regress to -3.2 AP. This
branch therefore only calibrates (`qat_moe.py --skip-train`) and keeps implicit INT8 in the ladder as
the "failing operators" contrast rung (the engine-inspector audit lists every layer's precision).

---

## 8. Risks and fallbacks

| Risk | Handling |
|---|---|
| Dynamic-axes export | rejected by policy; export is static 1x3x640x640 only. Export per resolution when several are needed. |
| TensorRT implicit INT8 silently leaves MoE blocks in FP16 | per-layer precision audit via the engine inspector, written to the logs; MoE blocks that must be INT8 go through explicit Q/DQ only. |
| Other ES_MOE weights with top-k / threshold enabled | rejected by default; `--routing dense` declares the fallback and must ship the eager-sparse vs eager-dense AP delta. |
| Calibration coverage under sparse routing | calibrate on the export-equivalent dense path; uncalibrated quantizers are audited and reported (never a silent export assert). |
| torch >= 2.9 / dynamo exporter | asserted against before running. |
| ONNX Runtime CUDA EP "available" but not loadable | cuDNN/cuBLAS from torch's wheels on `LD_LIBRARY_PATH`; `backend_val.py` proves the EP loaded before the GPU-bound val. |
| modelopt / TensorRT / ORT versions | pinned by `setup_l4.sh`, `pip freeze` diffed before/after; L4 and A100 (project03) numbers are not directly comparable, only deltas within one machine's ladder are. |
| MoT family (P1) has no trained weights | not run here; MoT drops experts under TorchScript/CoreML tracing (only the ONNX guard holds), so P1 needs a weight source first. |
| `/data` is a network volume (5000-image val + 1024-image calibration I/O) | rsync to local disk and point the yaml there; the calibration list is deterministic even-spread sampling, so it is reproducible. |

---

## 9. Next (P1 / P2)

- P1: add MoT on the same chain (needs trained weights); record failing operators, size and latency.
- P2: routing-agreement tool: export the TopK index tensors as extra graph outputs and compare expert
  selection per image between FP32 ORT and the INT8 engine (the v0.1 family has real top-k; EsMoE at
  k == E has no selection to compare, so compare routing-weight drift instead); five-family comparison;
  mixed-precision sensitive-layer fallback already has the `quantize_trt.py --bisect/--ablate` tooling.

---

## 10. Existing runtime (unchanged on this branch)

Cross-platform inference runtime (C++17; ONNX Runtime / NCNN / MNN / TensorRT / CoreML; Linux, Windows,
Jetson, macOS; CPU / CUDA / Metal), Windows GUI, native TensorRT deployment scripts for Jetson Orin
(`jetson/`), CoreML export and the macOS app (`mac/`), benchmark results on VisDrone / SKU-110K /
AI-TOD-v2. See `TECHNICAL_REPORT.md`, `jetson/README.md`, `mac/README.md` and the `main` branch README.
