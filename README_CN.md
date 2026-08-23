# A3：动态路由、Softmax、top-k 与异构专家在 ONNX / TensorRT / INT8 下的真实兼容性

分支 `dev/a3-smoke`，基于 `dev/project03` 创建。本文档只覆盖 **8.24 准入检查** 的范围：最小 `yolo.export`
冒烟测试、环境矩阵与后端版本、一份失败日志与一份成功日志，以及下方的准入表格。仓库原有的边缘推理运行时在第 10 节简述。
英文版见 `README.md`。

---

## 1. 课题定位

**仓库已有的能力（非本课题新建）：** 多后端推理运行时（ONNX Runtime / NCNN / MNN / TensorRT / CoreML，v1.1.0）、
Jetson Orin 原生 TensorRT 部署，以及 project03 阶段在 **剪枝后** v0.1-N 上完成的 INT8 研究（TensorRT 隐式 PTQ 与显式
Q/DQ 阶梯、校准覆盖率规律、敏感层集合、QAT 负结果）。需要强调的是，`ultralytics` 的导出器本身对 MoE 完全没有感知。

**本课题补齐的缺口（补齐闭环、严谨复测、工具化）：**

| 缺口 | 本分支的做法 |
|---|---|
| 导出器没有路由策略：稀疏 / 稠密切换只依赖各 MoE 模块内部的 `is_in_onnx_export()` 分支，静默发生 | `scripts/a3/smoke_export.py`：显式的 preserve / 声明式 dense / reject 三态策略，在导出之前完成决策并写入日志；`--dynamic` 直接拒绝（即失败日志） |
| 发布版（未剪枝）权重缺少 PyTorch 对后端的精度差基线 | 同一评测协议下给出 PyTorch / ONNX Runtime / TensorRT fp32 / fp16 的 mAP 差（`backend_val.py` 与 `quantize_trt.py`） |
| INT8 结论只在剪枝模型上成立 | 在 **发布的未剪枝** v0.1-N 与 EsMoE-N（COCO）上重跑隐式 INT8 与显式 Q/DQ |
| 代码引用沿用旧行号 | 所有策略依据在运行时按锁定 commit 重新 grep，连同命中行一起写入 `a3/results/smoke_*.json` |

锁定 commit：YOLO-Master `3ea98305a8449d8d9f4a00845e26ff9d8bf3b66e`（2026-08-01）；本仓库的 commit 记录在 `a3/env/matrix.md`。

---

## 2. 8.24 准入检查

| 序号 | 姓名 | 第一志愿 | 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 |
|---|---|---|---|---|---|---|---|---|---|---|
|  | Thomas | A3：动态路由、Softmax、top-k 和异构专家在 ONNX/TensorRT/INT8 下的真实兼容性、精度差和路由决策漂移 | `scripts/a3/setup_l4.sh` 生成 `a3/env/matrix.md`、`a3/env/pip-freeze.txt`，日志 `a3/logs/setup_l4.log` | P0：`yolo.export` ONNX 冒烟（v0.1-N、EsMoE-N）及 ORT 对齐；PyTorch 对 ORT / TRT fp32 / fp16 的精度差；附 INT8 阶梯（隐式 + 显式 Q/DQ） | 第 4 节（`scripts/a3/run_all_l4.sh`） | `a3/config/coco-a3.yaml`、`a3/config/sensitive_sets.md` | 成功：`a3/logs/v01n_smoke.log`、`a3/logs/esmoen_smoke.log`；失败（策略拒绝）：`a3/logs/v01n_smoke_dynamic_reject.log`；阶梯：`a3/logs/*_trt_ladder.log`、`*_trt_stempair.log`、`*_int8_qdq.log`、`*_backend_val.log` | `a3/results/*.json`、`a3/results/ladder_*.csv`，第 6 节 | 第 7 节 | 第 8 节 |

---

## 3. 环境矩阵

L4 机器最初只是一个裸 `torch 2.4.1+cu124` 容器。`scripts/a3/setup_l4.sh` 按固定版本安装全部后端（TensorRT 使用
`tensorrt-cu12==10.13.3.9`，与 project03 在 A100 上得出结论时的版本一致；ONNX Runtime 1.20.2；modelopt 0.27.1；
`ultralytics` 以可编辑方式安装锁定 commit 的 YOLO-Master），并在每一步之后断言 torch 没有被替换。整条链路跑完后再次采集
`pip freeze` 并做 diff，以证明导出过程没有触发 ultralytics 的自动安装（`YOLO_AUTOINSTALL=False`）。

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

## 4. 复现命令

```bash
# 在 L4 上执行（任何挂载了 /data 的 CUDA 机器均可）
cd /data/yolo-master-edge && git switch dev/a3-smoke
bash scripts/a3/setup_l4.sh                                        # 1. 环境（可重复执行）
. /root/a3venv/bin/activate
nohup bash scripts/a3/run_all_l4.sh > a3/logs/run_all.log 2>&1 &   # 2. 完整链路
# STAGES=env,smoke,ladder,qdq,backend（可用 STAGES=smoke 等只跑子集）
bash scripts/a3/run_followup_l4.sh                                 # 3. stem-pair 档位与复跑
```

各阶段的独立命令：

```bash
python scripts/a3/env_matrix.py --out a3/env
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-v0.1-N.pt \
    --out runs/a3/v01n --data a3/config/coco-a3.yaml \
    --json a3/results/smoke_v01n.json --log a3/logs/v01n_smoke.log
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-EsMoE-N.pt \
    --out runs/a3/esmoen --data a3/config/coco-a3.yaml \
    --json a3/results/smoke_esmoen.json --log a3/logs/esmoen_smoke.log
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-v0.1-N.pt \
    --out runs/a3/v01n-dynamic --dynamic ...            # 退出码 2：即失败日志
python scripts/a3/quantize_trt.py --model <pt> --data a3/config/coco-a3.yaml --out runs/a3/<m>/trt \
    --calib-images /data/datasets/coco/images/train2017 --calib-n 1024 \
    --modes fp32,fp16,int8 --max-rounds 0 --tf32-baseline off
python scripts/a3/quantize_trt.py --model runs/a3/<m>/trt/<stem>.onnx ... --modes int8 --pin-modules 0,1
python scripts/a3/qat_moe.py --model <pt> --data a3/config/coco-a3.yaml --out runs/a3/<m>/qdq \
    --calib-n 1024 --skip-train                         # 只校准的显式 Q/DQ
python scripts/a3/backend_val.py --model <pt> --onnx runs/a3/<m>/<stem>.onnx \
    --data a3/config/coco-a3.yaml --json a3/results/backend_<m>.json
```

---

## 5. 配置文件

- `a3/config/coco-a3.yaml`：COCO 路径，`train: images/train2017`（仅用于校准），`val: images/val2017`（用于评测）。
- `a3/config/sensitive_sets.md`：校准策略（不少于 1024 张训练集图片、隐式量化用 entropy、显式 per-channel Q/DQ）以及保持
  FP16 的敏感层集合（stem 前两层、路由器、DFL）。
- 全程统一评测协议：ultralytics `.val`，imgsz 640，batch 1，val2017 全量 5000 张；PyTorch、ONNX Runtime 与 TensorRT engine
  走同一条路径，因此所有精度差都可以直接相减。

---

## 6. 结果证据

所有精度均为 ultralytics `.val` 的 mAP50-95（COCO val2017，5000 张，imgsz 640，batch 1）。时延为 TensorRT engine 在 L4 上
纯模型 200 次执行的中位数；PyTorch 与 ORT 行不给时延。原始数据在 `a3/results/`，逐层精度审计在对应日志中。

### 6.1 P0：`yolo.export` 冒烟与 PyTorch 对后端的精度差

| 模型（发布版，未剪枝） | 后端 | mAP50-95 | 对 PyTorch 的差（AP） | 时延 ms | 体积 MB | 来源 |
|---|---|---|---|---|---|---|
| v0.1-N（OptimizedMOEImproved x3，E=4/8/16，k=2） | PyTorch（已修复） | **0.4292** | 0 | - | 15.1（.pt） | `backend_v01n.json` |
| | ONNX Runtime CUDA EP（默认 TF32） | 0.4286 | -0.0006 | - | 30.5（.onnx） | `backend_v01n.json` |
| | TensorRT fp32（关闭 TF32） | 0.4287 | -0.0005 | 3.898 | 42.6 | `ladder_v01n.csv` |
| | TensorRT fp16 | 0.4285 | -0.0007 | 1.935（-50.4%） | 21.3 | `ladder_v01n.csv` |
| EsMoE-N（ES_MOE x4，E=3，k=3） | PyTorch（dense 与 sparse 等价） | **0.4270** | 0 | - | 5.7（.pt） | `backend_esmoen.json` |
| | ONNX Runtime CUDA EP（默认 TF32） | 0.4267 | -0.0004 | - | 11.1（.onnx） | `backend_esmoen.json` |
| | TensorRT fp32（关闭 TF32） | 0.4267 | -0.0004 | 2.982 | 22.0 | `ladder_esmoen.csv` |
| | TensorRT fp16 | 0.4268 | -0.0003 | 1.720（-42.3%） | 11.4 | `ladder_esmoen.csv` |

冒烟验证（`smoke_*.json`）：

| 模型 | 策略判定 | ONNX 中的路由算子 | 专家卷积 模块/图 | ORT fp32 对齐（归一化误差 / top-100 anchor 一致率） | ORT 默认 TF32 |
|---|---|---|---|---|---|
| v0.1-N | 3 个块判定为 `preserve`（精确 gather） | TopK 3、GatherElements 6、Softmax 12，无 If/Loop/NonZero | 56 / 56 | 3.9e-6 / 100% | 框坐标最大偏移 1.38 px，99.9% |
| EsMoE-N | 4 个块判定为 `dense`，`semantic_change=false`（k==E，无 top-k / 阈值） | TopK 0、Softmax 13，无 If/Loop/NonZero | 24 / 24 | 5.4e-6 / 100% | 最大偏移 2.98 px，100% |
| v0.1-N `--dynamic` | **导出前即拒绝，退出码 2**（`v01n_smoke_dynamic_reject.log`） | - | - | - | - |

### 6.2 INT8 阶梯（隐式 PTQ 作对照 + 显式 Q/DQ 交付配方），均使用 1024 张 train2017 校准图

| 模型 | INT8 方式 | mAP50-95 | 对 fp32 的差（AP） | 时延 ms | 体积 MB | engine 逐层精度（Int8 / Half / Float） |
|---|---|---|---|---|---|---|
| v0.1-N | 隐式 entropy，head 与路由器钉为 FP16 | 0.3754 | -5.33 | 1.944 | 17.1 | 189 / 224 / 14 |
| v0.1-N | 隐式 entropy，再加 stem 前两层（model.0/1） | 0.3955 | -3.32 | 1.965 | 17.0 | 见 `v01n_trt_stempair.log` |
| v0.1-N | **显式 Q/DQ，只校准（modelopt）** | **0.4164** | **-1.23** | 2.231 | 21.0 | 196 / 176 / 85 |
| EsMoE-N | 隐式 entropy，head 与路由器钉为 FP16 | 0.3606 | -6.61 | 1.843 | 12.9 | 165 / 147 / 60 |
| EsMoE-N | 隐式 entropy，再加 stem 前两层 | 0.3752 | -5.15 | 1.917 | 13.0 | 见 `esmoen_trt_stempair.log` |
| EsMoE-N | **显式 Q/DQ，只校准（modelopt）** | **0.4121** | **-1.49** | 2.141 | 11.5 | 174 / 168 / 112 |

结果解读：
- 在 N 尺度上，两个模型族的 INT8 都 **不比 fp16 快**（v0.1-N 为 1.94 对 1.94 ms，EsMoE-N 为 1.84 对 1.72 ms），与 project03
  在剪枝模型上的结论一致：TensorRT GPU 上这个宽度的模型，部署精度应选 fp16。
- 隐式 PTQ 的精度损失排序与剪枝研究完全相同：surgical < stem-pair < 显式 Q/DQ；显式 Q/DQ 是唯一把损失压进 2 AP 以内的路径
  （剪枝版 v0.1-N 为 -0.80，未剪枝版为 -1.23，因为 4/8/16 个专家全部进入了 INT8 覆盖面）。
- **只有未剪枝模型才暴露出的路由专属问题**（剪枝模型 k==E，不可能出现）：未剪枝 v0.1-N 若在 eager 的稀疏路径上做校准，
  校准集里从未被路由到的专家得不到任何激活统计，而导出图会计算全部专家，于是 modelopt 在导出时直接断言
  "Quantizer has not been calibrated"。修复办法是校准时走与导出等价的 dense 路径（与导出器相同的 `is_in_onnx_export`
  mock），并对未校准的量化器做显式审计（本次 360 个全部校准，0 个被禁用）。这正是课题所问"动态路由与 INT8 校准的真实
  兼容性"的一个具体样本：**校准覆盖必须以导出图为准，而不是以 eager 的路由语义为准。**
- ONNX Runtime 的 CUDA EP 默认对卷积启用 TF32：框坐标最大偏移 1.4 / 3.0 px，但 mAP 只差 -0.0006 / -0.0004。因此冒烟测试的
  对齐门限用 `use_tf32=0` 做 fp32 对 fp32 的比较，TF32 的结果单独记录。

---

## 7. 设计说明

### 7.1 动态路由导出策略（显式，不允许静默改变语义）

导出器（`ultralytics/engine/exporter.py`）不含任何 MoE 逻辑；每个 MoE 族在自己的 `forward` 里通过
`torch.onnx.is_in_onnx_export()` 选择导出分支。`smoke_export.py` 在调用同一个 `YOLO.export` 之前，先把这个选择变成显式决策，
并把依据（文件、运行时重新 grep 得到的行号、commit）写进结果 JSON：

| MoE 族 | 锁定 commit 下的导出分支行为 | 策略 | 语义变化 |
|---|---|---|---|
| `OptimizedMOEImproved`（v0.1 系列） | 先计算全部专家，再用 `torch.gather` 取 top-k（`selected = torch.gather(all_outs, ...)`） | `preserve` | 无（精确 top-k） |
| `ES_MOE`（EsMoE 系列） | 强制走 `_dense_forward`：对全部专家做 softmax 加权求和，跳过 `_sparse_forward` 的 top-k / `dynamic_threshold` | `dense`；仅当 `k == E` 且未启用 top-k / 阈值时视为无语义变化（发布的 COCO 权重满足：E=3，k=3，`use_top_k=False`，无 `dynamic_threshold`） | 否则必须用 `--routing dense` 显式声明，并报告 eager-sparse 与 eager-dense 的 AP 差 |
| 其他族 | 没有经过验证的导出分支 | `reject` | - |
| 任何族 + `--dynamic` | gather 分支把 B/H/W 以 Python 整数写死在 `view/expand` 里 | **导出前拒绝**（退出码 2） | 动态轴图在其他形状下会静默出错 |

前提条件：torch < 2.9（`ultralytics/utils/export/engine.py` 对 torch >= 2.4 强制 `dynamo=False`；dynamo 导出器会绕过上述
guard，静默丢掉专家）。

### 7.2 导出后验证（三层）
1. ONNX 算子直方图：统计 TopK / Gather / GatherElements / Softmax 的数量；一旦出现 `If / Loop / NonZero` 即判为失败
   （说明 guard 没有生效）。
2. 专家卷积计数：图中 `/experts` 前缀下的 `Conv` 节点数，对比模块中 `*.experts.*` 下的 `nn.Conv2d` 数。
3. ORT 对齐：在真实 val 图片上，ONNX Runtime（fp32，`use_tf32=0`）对比 **保留原生稀疏语义的 eager PyTorch**，报告最大绝对误差、
   归一化误差以及 top-100 anchor 一致率。

### 7.3 发布权重必须做的两处修复（不修则所有结论皆错）
- v0.1-N：该 checkpoint 早于 `add_residual` 属性诞生，兼容 shim 把它默认为 True，导致分类置信度坍缩到约 0.04，mAP50-95 从 0.43
  掉到 0.007。所有脚本在加载后对缺失该属性的块强制置 False（复用 `scripts/project03/diagnose_moe._fix_add_residual`）。
  这些旧 pickle 还携带了如今已变成只读属性的实例字段（`aux_loss`），加载时一并剥离（`_strip_property_shadows`）。
- EsMoE-N：评测与导出统一走 dense 路径（`use_sparse_inference=False`），与 7.1 的判定保持一致。

### 7.4 为什么 INT8 方案选择"只校准的显式 Q/DQ"
project03 在剪枝 v0.1-N（COCO，A100，TensorRT 10.13.3.9）上的结论：隐式 PTQ 无法量化 MoE 块（路由的 gather/expand 会打断 INT8
链，TensorRT 静默回退到 FP16）；显式 per-channel Q/DQ 可达 -0.80 AP；而 3 个 epoch 的 QAT 反而退化到 -3.2 AP。因此本分支只做
校准（`qat_moe.py --skip-train`），并把隐式 INT8 留在阶梯中作为"失败算子"的对照档（engine inspector 的审计会列出每一层的精度）。

---

## 8. 风险与降级

| 风险 | 处理 |
|---|---|
| 动态轴导出 | 策略层直接拒绝；只导出静态 1x3x640x640。需要多分辨率时按分辨率分别导出。 |
| TensorRT 隐式 INT8 把 MoE 块静默留在 FP16 | 用 engine inspector 做逐层精度审计并写入日志；必须量化为 INT8 的 MoE 块只走显式 Q/DQ。 |
| 其他启用了 top-k / 阈值的 ES_MOE 权重 | 默认拒绝；可用 `--routing dense` 声明回退，但必须附上 eager-sparse 与 eager-dense 的 AP 差。 |
| 稀疏路由下的校准覆盖 | 校准走与导出等价的 dense 路径；未校准的量化器会被审计并报告（绝不让导出静默断言）。 |
| torch >= 2.9 / dynamo 导出器 | 运行前断言拒绝。 |
| ONNX Runtime 的 CUDA EP "可用"但加载失败 | 把 torch 自带的 cuDNN / cuBLAS 放入 `LD_LIBRARY_PATH`；`backend_val.py` 在 GPU 评测前先证明 EP 真正加载成功。 |
| modelopt / TensorRT / ORT 版本 | 由 `setup_l4.sh` 固定，`pip freeze` 前后 diff；L4 与 A100（project03）的数字不可直接比较，只比较同一台机器阶梯内的差值。 |
| MoT 族（P1 要求）没有训练权重 | 本阶段不做；MoT 在 TorchScript / CoreML trace 下会丢专家（只有 ONNX 的 guard 有效），P1 需先解决权重来源。 |
| `/data` 是网络卷（5000 张评测 + 1024 张校准的 I/O） | 可 rsync 到本地盘后修改 yaml 指向；校准列表由确定性的等距采样生成，可复现。 |

---

## 9. 后续（P1 / P2）

- P1：在同一条链路上加入 MoT（需要训练权重）；记录失败算子、体积与时延。
- P2：路由一致率工具：把 TopK 索引张量作为额外的图输出导出，逐图对比 FP32 ORT 与 INT8 engine 的专家选择一致率
  （v0.1 系列有真实的 top-k；EsMoE 在 k == E 下没有"选择"可比，改比路由权重的漂移）；五族对比；混合精度的敏感层回退已经有
  `quantize_trt.py --bisect/--ablate` 工具链可用。

---

## 10. 仓库原有能力（本分支未改动）

跨平台推理运行时（C++17；ONNX Runtime / NCNN / MNN / TensorRT / CoreML；Linux、Windows、Jetson、macOS；CPU / CUDA / Metal）、
Windows 图形界面、Jetson Orin 原生 TensorRT 部署脚本（`jetson/`）、CoreML 导出与 macOS 应用（`mac/`），以及 VisDrone / SKU-110K /
AI-TOD-v2 上的基准结果。详见 `TECHNICAL_REPORT.md`、`jetson/README.md`、`mac/README.md` 与 `main` 分支的 README。
