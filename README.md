# A3：动态路由、Softmax、top-k 与异构专家在 ONNX / TensorRT / INT8 下的真实兼容性

分支 `dev/a3-smoke`，基于边缘仓库冻结基线 `446f53ac` 创建。本文档只覆盖 **8.24 准入检查** 的范围：最小 `yolo.export`
冒烟测试、环境矩阵与后端版本、一份失败日志与一份成功日志，以及下方的准入表格。所有结果均在统一公共基线
**Tencent/YOLO-Master `acce839c7e895d6b179de7f7093fa879e237cc7b`**（2026-08-21，ultralytics 8.4.101）上测得；
此前在 `3ea98305`（8.3.240）上的一轮结果作废，仅保留在 `a3/results-3ea98305/` 与 `a3/logs-3ea98305/` 供对照
（`scripts/a3/compare_baselines.py` 可打印两版对比与结论是否翻转）。

---

## 1. 课题定位

**上游仓库（Tencent/YOLO-Master，即题目所指的"仓库"）已有的能力，本课题不据为己有：** 开箱即用的完整流程（安装、验证、
训练、推理以及 ONNX、TensorRT 等部署）、`yolo.export` Skill runner、MoE / MoA / MoT 模块及其路由机制、专家利用率分析与剪枝、
LoRA，以及第三方部署（TensorRT-YOLO、ncnn-YOLO-Master-android）。在 `acce839c` 上，导出器还自带了 `export_preflight`：
按能力矩阵（`ultralytics/cfg/export-capability-matrix.yaml`）对每个路由模块给出 dynamic / dense_fallback / merged /
routing_preserved / refuse 的策略，并在 strict 模式下拒绝导出。另有两项输入来自我们自己在配套仓库 `yolo-master-edge` 里的
前期工作（多后端运行时、Jetson 部署，以及 project03 在 **剪枝后** v0.1-N 上的 INT8 研究），本课题复用它们，同样不作为新成果呈现。

**本课题补齐的缺口（补齐闭环、严谨复测、工具化）：**

| 缺口 | 本分支的做法 |
|---|---|
| 上游 preflight 的判定粒度是"模块族"：能力矩阵把所有 MoE 族的 ONNX 导出一律标为 `dense_fallback`（"data-dependent routing is replaced by dense fallback"），而 v0.1 系列的导出分支其实是精确 top-k（gather） | `scripts/a3/smoke_export.py` 按 **模块实例** 判定 preserve / dense / reject，并用三层导出后验证（算子直方图、专家卷积计数、ORT 对齐）证明判定；`--dynamic` 在导出前拒绝（失败日志） |
| 发布版（未剪枝）权重在新 lineage 下的兼容性：旧 pickle 缺少今天的类在 eval forward 里无条件读取的属性 | 显式的遗留属性修复层（`scripts/a3/diagnose_moe.py`），逐项记录补了什么 |
| 未剪枝模型的 INT8 校准覆盖 | 按导出图（dense 等价路径）做校准，并审计未校准的量化器 |
| INT8 结论只在剪枝模型上成立 | 在 **发布的未剪枝** v0.1-N 与 EsMoE-N（COCO）上重跑隐式 INT8 与显式 Q/DQ |
| 代码引用沿用旧行号 | 所有策略依据在运行时按锁定 commit 重新 grep，连同命中行一起写入 `a3/results/smoke_*.json` |

锁定 commit：YOLO-Master `acce839c7e895d6b179de7f7093fa879e237cc7b`（fork `skywalker-lt/YOLO-Master` 的 `main` 已强制对齐到
该 commit）；边缘仓库基线 `446f53ac4b91abfa193a56ab11f91a5d158611c8`。

---

## 2. 8.24 准入检查

| 序号 | 姓名 | 第一志愿 | 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 |
|---|---|---|---|---|---|---|---|---|---|---|
|  | Thomas | A3：动态路由、Softmax、top-k 和异构专家在 ONNX/TensorRT/INT8 下的真实兼容性、精度差和路由决策漂移 | `scripts/a3/setup_l4.sh` 生成 `a3/env/matrix.md`、`a3/env/pip-freeze.txt`，日志 `a3/logs/setup_l4.log` | P0：`yolo.export` ONNX 冒烟（v0.1-N、EsMoE-N）及 ORT 对齐；PyTorch 对 ORT / TRT fp32 / fp16 的精度差；附 INT8 阶梯（隐式 + 显式 Q/DQ） | 第 4 节（`scripts/a3/run_all_l4.sh`） | `a3/config/coco-a3.yaml`、`a3/config/sensitive_sets.md` | 成功：`a3/logs/v01n_smoke.log`、`a3/logs/esmoen_smoke.log`；失败（策略拒绝）：`a3/logs/v01n_smoke_dynamic_reject.log`；失败（TensorRT 构建不可行）：`a3/results/failed_v01n_int8_headrouters.txt`；阶梯：`a3/logs/*_trt_ladder.log`、`*_int8_pins_*.log`、`*_trt_stempair.log`、`*_int8_qdq.log`、`*_backend_val.log` | `a3/results/*.json`、`a3/results/ladder_*.csv`，第 6 节 | 第 7 节 | 第 8 节 |

---

## 3. 环境矩阵

L4 机器最初只是一个裸 `torch 2.4.1+cu124` 容器。`scripts/a3/setup_l4.sh` 按固定版本安装全部后端（TensorRT 使用
`tensorrt-cu12==10.13.3.9`，与 project03 在 A100 上得出结论时的版本一致；ONNX Runtime 1.20.2；modelopt 0.27.1；
`ultralytics` 以可编辑方式安装锁定 commit 的 YOLO-Master，依赖按其声明解析，但通过 constraints 文件把 torch 钉死在预装的
2.4.1+cu124）。每装完一组依赖就重新检查 torch 版本，防止 pip 在解析 modelopt、onnxruntime-gpu 等依赖时顺手把 torch 换掉。
整条链路跑完后再次采集 `pip freeze` 并做 diff，以证明导出过程没有触发 ultralytics 的自动安装（`YOLO_AUTOINSTALL=False`）。

| item | value |
|---|---|
| captured_utc | 2026-08-23T07:14:33+00:00 |
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
| ultralytics | 8.4.101 |
| ultralytics_file | /data/YOLO-Master/ultralytics/__init__.py |
| yolo_master_repo | `acce839c7e89` (2026-08-21, main) |
| edge_repo | dev/a3-smoke（见 `a3/env/matrix.json`） |

---

## 4. 复现命令

```bash
# 在 L4 上执行（任何挂载了 /data 的 CUDA 机器均可）
cd /data/yolo-master-edge && git switch dev/a3-smoke
bash scripts/a3/setup_l4.sh                                        # 1. 环境（可重复执行）
. /root/a3venv/bin/activate
nohup bash scripts/a3/run_all_l4.sh > a3/logs/run_all.log 2>&1 &   # 2. 完整链路
# STAGES=env,smoke,ladder,qdq,backend（可用 STAGES=smoke 等只跑子集）
bash scripts/a3/run_v01n_pins_l4.sh                                # 3. v0.1-N 隐式 INT8 钉层集合诊断
python scripts/a3/compare_baselines.py --old a3/results-3ea98305 --new a3/results   # 4. 两版基线对比
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
    --modes fp32,fp16,int8 --max-rounds 0 --tf32-baseline off      # 构建不可行的档位会被记录而不是中止
python scripts/a3/quantize_trt.py --model runs/a3/<m>/trt/<stem>.onnx ... --modes int8 --pins head|routers|none
python scripts/a3/quantize_trt.py --model runs/a3/<m>/trt/<stem>.onnx ... --modes int8 --pins head --pin-modules 0,1
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
- 全程统一评测协议：ultralytics `.val`（8.4.101），imgsz 640，batch 1，val2017 全量 5000 张；PyTorch、ONNX Runtime 与
  TensorRT engine 走同一条路径，因此所有精度差都可以直接相减。

---

## 6. 结果证据

所有精度均为 ultralytics 8.4.101 `.val` 的 mAP50-95（COCO val2017，5000 张，imgsz 640，batch 1）。时延为 TensorRT engine 在
L4 上纯模型 200 次执行的中位数；PyTorch 与 ORT 行不给时延。原始数据在 `a3/results/`，逐层精度审计在对应日志中。

**关于绝对值的说明：** 同一份权重、同一台机器、CPU fp32，在 8.3.240 的验证器下 v0.1-N 得 0.4292，在 8.4.101 下得 0.4250
（EsMoE-N 0.4270 对 0.4228）；两个 lineage 的模型原始输出逐元素一致（框坐标最大差 0.001 px，类别分数 5e-5），差异全部来自
8.4.101 的验证/后处理路径（`nms.py`、`ops.py`、`augment.py`、`val.py` 均有大改）。因此 **绝对 mAP 不跨 lineage 比较，
本节所有差值都在同一 lineage 内计算。**

### 6.1 P0：`yolo.export` 冒烟与 PyTorch 对后端的精度差

| 模型（发布版，未剪枝） | 后端 | mAP50-95 | 对 PyTorch 的差（AP） | 时延 ms | 体积 MB | 来源 |
|---|---|---|---|---|---|---|
| v0.1-N（OptimizedMOEImproved x3，E=4/8/16，k=2） | PyTorch（已修复） | **0.4250** | 0 | - | 15.1（.pt） | `backend_v01n.json` |
| | ONNX Runtime CUDA EP（默认 TF32） | 0.4245 | -0.0005 | - | 30.5（.onnx） | `backend_v01n.json` |
| | TensorRT fp32（关闭 TF32） | 0.4246 | -0.0004 | 3.990 | 41.2 | `ladder_v01n.csv` |
| | TensorRT fp16 | 0.4244 | -0.0006 | 1.927（-51.7%） | 21.6 | `ladder_v01n.csv` |
| EsMoE-N（ES_MOE x4，E=3，k=3） | PyTorch（dense 与 sparse 等价） | **0.4228** | 0 | - | 5.7（.pt） | `backend_esmoen.json` |
| | ONNX Runtime CUDA EP（默认 TF32） | 0.4225 | -0.0003 | - | 11.1（.onnx） | `backend_esmoen.json` |
| | TensorRT fp32（关闭 TF32） | 0.4225 | -0.0003 | 2.822 | 20.7 | `ladder_esmoen.csv` |
| | TensorRT fp16 | 0.4225 | -0.0003 | 1.720（-39.1%） | 12.4 | `ladder_esmoen.csv` |

冒烟验证（`smoke_*.json`）：

| 模型 | 本分支判定 | 上游 preflight 判定 | ONNX 中的路由算子 | 专家卷积 模块/图 | ORT fp32 对齐（归一化误差 / top-100 anchor 一致率） | ORT 默认 TF32 |
|---|---|---|---|---|---|---|
| v0.1-N | 3 个块 `preserve`（精确 gather，无语义变化） | `dense_fallback` x3 | TopK 3、GatherElements 6、Softmax 12，无 If/Loop/NonZero | 56 / 56 | 6.3e-6 / 100% | 框坐标最大偏移 1.38 px，99.9% |
| EsMoE-N | 4 个块 `dense`，`semantic_change=false`（k==E，无 top-k / 阈值） | `dense_fallback` x4 | TopK 0、Softmax 13，无 If/Loop/NonZero | 24 / 24 | 5.4e-6 / 100% | 最大偏移 2.98 px，100% |
| v0.1-N `--dynamic` | **导出前即拒绝，退出码 2**（`v01n_smoke_dynamic_reject.log`） | - | - | - | - | - |

### 6.2 INT8 阶梯（隐式 PTQ 作对照 + 显式 Q/DQ 交付配方），均使用 1024 张 train2017 校准图

| 模型 | INT8 方式 | mAP50-95 | 对 fp32 的差（AP） | 时延 ms | 体积 MB | engine 逐层精度（Int8 / Half / Float） |
|---|---|---|---|---|---|---|
| v0.1-N | 隐式 entropy，head 与路由器钉为 FP16 | **构建不可行**（`failed_v01n_int8_headrouters.txt`） | - | - | - | TensorRT OBEY：`/model.7` Conv+SiLU 融合无可用实现 |
| v0.1-N | 隐式 entropy，只钉 head | 0.3706 | -5.40 | 1.969 | 17.4 | 199 / 179 / 56 |
| v0.1-N | 隐式 entropy，只钉路由器 | 0.3340 | -9.06 | 1.930 | 16.7 | `v01n_int8_pins_routers.log` |
| v0.1-N | 隐式 entropy，不钉任何层 | 0.3341 | -9.05 | 1.902 | 17.2 | `v01n_int8_pins_none.log` |
| v0.1-N | 隐式 entropy，head + stem 前两层 | **构建不可行**（`failed_v01n_stempair.txt`） | - | - | - | 同上 |
| v0.1-N | **显式 Q/DQ，只校准（modelopt）** | **0.4129** | **-1.17** | 2.238 | 21.1 | 196 / 176 / 85 |
| EsMoE-N | 隐式 entropy，head 与路由器钉为 FP16 | 0.3554 | -6.71 | 1.937 | 13.8 | 165 / 147 / 60 |
| EsMoE-N | 隐式 entropy，head + stem 前两层 | 0.3705 | -5.20 | 1.912 | 13.2 | `esmoen_trt_stempair.log` |
| EsMoE-N | **显式 Q/DQ，只校准（modelopt）** | **0.4082** | **-1.43** | 2.195 | 12.6 | 174 / 168 / 112 |

结果解读：
- 在 N 尺度上，两个模型族的 INT8 都 **不比 fp16 快**（v0.1-N 1.97 对 1.93 ms，EsMoE-N 1.94 对 1.72 ms），与 project03 在剪枝模型
  上的结论一致：TensorRT GPU 上这个宽度的模型，部署精度应选 fp16。
- 显式 Q/DQ 是唯一把损失压进 2 AP 以内的路径（剪枝版 v0.1-N 为 -0.80，未剪枝版为 -1.17，因为 4/8/16 个专家全部进入了 INT8
  覆盖面）；这一结论在 `3ea98305` 与 `acce839c` 两版基线上一致（-1.23 / -1.49 对 -1.17 / -1.43）。
- **隐式 INT8 的"失败算子"证据在新基线上变得更具体：** 8.4.101 导出器生成的 v0.1-N 图上，TensorRT 在 OBEY 精度约束下，
  只要同时钉 head 与任何另一组层（路由器或 stem），`/model.7` 的 Conv+SiLU 融合就找不到可实现的 kernel，构建直接失败；
  只钉 head 可以构建，精度与旧基线上 head+路由器的档位同级（-5.4 对 -5.3）；只钉路由器与不钉任何层结果相同（-9.1），说明
  v0.1-N 的路由器钉层在隐式量化里没有贡献，精度损失几乎全部来自 head。EsMoE-N 的图不受影响（head+路由器 -6.7，+stem -5.2）。
- **路由专属的校准问题**（剪枝模型 k==E 时不可能出现）：未剪枝 v0.1-N 若在 eager 的稀疏路径上做校准，校准集里从未被路由到
  的专家得不到任何激活统计，而导出图会计算全部专家，modelopt 在导出时直接断言 "Quantizer has not been calibrated"。
  修复办法是校准时走与导出等价的 dense 路径，并对未校准的量化器做显式审计（本次 v0.1-N 360 个、EsMoE-N 282 个全部校准，
  0 个被禁用）。**校准覆盖必须以导出图为准，而不是以 eager 的路由语义为准。**
- ONNX Runtime 的 CUDA EP 默认对卷积启用 TF32：框坐标最大偏移 1.4 / 3.0 px，但 mAP 只差 -0.0005 / -0.0003。因此冒烟测试的
  对齐门限用 `use_tf32=0` 做 fp32 对 fp32 的比较，TF32 的结果单独记录。

---

## 7. 设计说明

### 7.1 动态路由导出策略（显式，不允许静默改变语义）

在 `acce839c` 上，`yolo.export` 会先运行 `ultralytics/utils/export_preflight.py`：按能力矩阵与模块自报的
`export_capabilities()` 给每个路由模块一个策略。矩阵把所有 MoE 族的 ONNX 导出一律标为 `dense_fallback`，默认的运行时声明也一律
`exact_sparse_export=False`。这对 EsMoE 是准确的，对 v0.1 系列则过于保守：`OptimizedMOEImproved` 的导出分支先计算全部专家、
再用 `torch.gather` 取 top-k，语义与 eager 完全一致。`smoke_export.py` 在调用同一个 `YOLO.export` 之前，按 **模块实例** 做判定，
并把依据（文件、运行时重新 grep 得到的行号、commit）写进结果 JSON：

| MoE 族 | 锁定 commit 下的导出分支行为 | 本分支策略 | 语义变化 |
|---|---|---|---|
| `OptimizedMOEImproved`（v0.1 系列） | 先计算全部专家，再用 `torch.gather` 取 top-k（`selected = torch.gather(all_outs, ...)`） | `preserve` | 无（精确 top-k）；上游 preflight 标为 `dense_fallback` 属于保守标注 |
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

### 7.3 发布权重在 8.4.101 下必须做的修复（不修则要么崩溃、要么所有结论皆错）
- v0.1-N：checkpoint 早于 `add_residual` 属性诞生；8.3.x 的兼容 shim 把它默认为 True（分类置信度坍缩到约 0.04，mAP50-95
  从 0.43 掉到 0.007），8.4.x 干脆删掉了这个 shim，同时路由器基类新增了 forward 里无条件读取的 `capacity_factor`，旧 pickle
  直接 `AttributeError`。修复层（`scripts/a3/diagnose_moe.py`）对缺失的属性逐项补默认值并打印补了什么：`add_residual=False`、
  `routing.capacity_factor=None`、progressive-sparsity 相关字段等。
- 两个 checkpoint 的旧 pickle 还携带了如今已变成只读属性的实例字段（`aux_loss`），加载时一并剥离（`_strip_property_shadows`）。
- EsMoE-N：评测与导出统一走 dense 路径（`use_sparse_inference=False`），与 7.1 的判定保持一致。

### 7.4 为什么 INT8 方案选择"只校准的显式 Q/DQ"
project03 在剪枝 v0.1-N（COCO，A100，TensorRT 10.13.3.9）上的结论：隐式 PTQ 无法量化 MoE 块（路由的 gather/expand 会打断 INT8
链，TensorRT 静默回退到 FP16）；显式 per-channel Q/DQ 可达 -0.80 AP；而 3 个 epoch 的 QAT 反而退化到 -3.2 AP。因此本分支只做
校准（`qat_moe.py --skip-train`），并把隐式 INT8 留在阶梯中作为"失败算子"的对照档（engine inspector 的审计会列出每一层的精度；
新基线上 v0.1-N 的隐式档位本身就以"构建不可行"的形式成为失败证据）。

---

## 8. 风险与降级

| 风险 | 处理 |
|---|---|
| 动态轴导出 | 策略层直接拒绝；只导出静态 1x3x640x640。需要多分辨率时按分辨率分别导出。 |
| 上游 preflight 把精确 top-k 导出标为 `dense_fallback` | 本分支按模块实例判定并用三层验证证明；若上游矩阵在后续版本改为 `refuse`，需用 `--routing` 显式声明或改矩阵。 |
| TensorRT 隐式 INT8 在 OBEY 约束下构建不可行（v0.1-N，8.4.101 图） | 失败被记录为档位证据（`failed_*.txt`），阶梯继续；可行配方为只钉 head；需要真正 INT8 的 MoE 块只走显式 Q/DQ。 |
| TensorRT 隐式 INT8 把 MoE 块静默留在 FP16 | 用 engine inspector 做逐层精度审计并写入日志。 |
| 其他启用了 top-k / 阈值的 ES_MOE 权重 | 默认拒绝；可用 `--routing dense` 声明回退，但必须附上 eager-sparse 与 eager-dense 的 AP 差。 |
| 稀疏路由下的校准覆盖 | 校准走与导出等价的 dense 路径；未校准的量化器会被审计并报告（绝不让导出静默断言）。 |
| lineage 切换导致绝对 mAP 变化 | 已证明模型输出一致、差异在验证路径；只在同一 lineage 内比较差值，`compare_baselines.py` 用于对照。 |
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

## 10. 配套运行时（我们的 `yolo-master-edge` main 分支，本分支未改动）

我们为上游模型编写的跨平台推理运行时（C++17；ONNX Runtime / NCNN / MNN / TensorRT / CoreML；Linux、Windows、Jetson、macOS；
CPU / CUDA / Metal）、Windows 图形界面、Jetson Orin 原生 TensorRT 部署脚本（`jetson/`）、CoreML 导出与 macOS 应用（`mac/`），
以及 VisDrone / SKU-110K / AI-TOD-v2 上的基准结果。详见 `TECHNICAL_REPORT.md`、`jetson/README.md`、`mac/README.md` 与
`main` 分支的 README。
