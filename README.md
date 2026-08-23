# A3: 动态路由 / Softmax / top-k / 异构专家在 ONNX / TensorRT / INT8 下的真实兼容性
# A3: Real compatibility of dynamic routing under ONNX / TensorRT / INT8

分支 `dev/a3-smoke`，基于 `dev/project03`。本 README 只覆盖 **8.24 准入检查** 的范围：
最小 `yolo.export` smoke、环境矩阵与后端版本、一份失败日志 + 一份成功日志、以及下方准入表格。
完整的边缘推理运行时文档见文末 "现有能力"。

This branch covers the **8.24 entry check only**: the minimal `yolo.export` smoke, the
environment matrix and backend versions, one failure and one success log, and the entry
table below. The existing runtime documentation is summarized at the end.

---

## 1. 课题定位 / Scope

**已有（不是本课题新建）/ Already shipped, not claimed here:** 本仓库已提供 ONNX Runtime /
NCNN / MNN / TensorRT / CoreML 多后端运行时（v1.1.0）、Jetson Orin 原生 TensorRT 部署、
以及 project03 阶段在 **剪枝后** v0.1-N 上完成的 INT8 研究（TensorRT 隐式 PTQ 与显式 Q/DQ
阶梯、校准覆盖率定律、敏感层集合、QAT 负结果）。`ultralytics` 的导出器本身对 MoE 毫无感知。

**本课题补齐的缺口 / Gaps this topic closes (补齐闭环、严谨复测、工具化):**

| 缺口 | 本分支做法 |
|---|---|
| 导出器对动态路由无策略：稀疏/稠密切换只靠模块内部的 `is_in_onnx_export()` 分支，静默发生 | `scripts/a3/smoke_export.py`：显式 preserve / declared dense / reject 策略，导出前决策并写日志，`--dynamic` 直接拒绝（失败日志） |
| 发布权重（未剪枝）的 PyTorch 对后端精度差没有基线 | 同一协议下 PyTorch / ORT / TensorRT fp32 / fp16 的 mAP 差（`backend_val.py` + `quantize_trt.py`） |
| INT8 结论只在剪枝模型上 | 在 **发布的未剪枝** v0.1-N 与 EsMoE-N (COCO) 上重跑隐式 INT8 与显式 Q/DQ |
| 代码引用用旧行号 | 所有策略依据在运行时按锁定 commit 重新 grep，写入 `a3/results/smoke_*.json` |

锁定 commit / Locked commits: YOLO-Master `3ea98305a8449d8d9f4a00845e26ff9d8bf3b66e`
(2026-08-01); 本仓库见 `a3/env/matrix.md`。

---

## 2. 8.24 准入检查 / Entry check

| 序号 | 姓名 | 第一志愿 | 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 |
|---|---|---|---|---|---|---|---|---|---|---|
|  | Thomas | A3: 动态路由、Softmax、top-k 和异构专家在 ONNX/TensorRT/INT8 下的真实兼容性、精度差和路由决策漂移 | `scripts/a3/setup_l4.sh` -> `a3/env/matrix.md`, `a3/env/pip-freeze.txt`, 日志 `a3/logs/setup_l4.log` | P0: `yolo.export` ONNX smoke (v0.1-N, EsMoE-N) + ORT 对齐；PyTorch vs ORT / TRT fp32 / fp16 精度差；附 INT8 (隐式 + 显式 Q/DQ) 阶梯 | 第 4 节 (`scripts/a3/run_all_l4.sh`) | `a3/config/coco-a3.yaml`, `a3/config/sensitive_sets.md` | 成功: `a3/logs/v01n_smoke.log`, `a3/logs/esmoen_smoke.log`；失败(策略拒绝): `a3/logs/v01n_smoke_dynamic_reject.log`；阶梯: `a3/logs/*_trt_ladder.log`, `*_int8_qdq.log`, `*_backend_val.log` | `a3/results/*.json`, `a3/results/ladder_*.csv`，第 6 节表格 | 第 7 节 | 第 8 节 |

---

## 3. 环境矩阵 / Environment matrix

L4 机器从裸 `torch 2.4.1+cu124` 容器起步，`scripts/a3/setup_l4.sh` 固定安装全部后端
（TensorRT `tensorrt-cu12==10.13.3.9`，与 project03 在 A100 上的结论同版本；ORT 1.20.2；
modelopt 0.27.1；`ultralytics` 为锁定 commit 的可编辑安装）。脚本在每一步后断言 torch 未被替换。
`pip freeze` 在跑完整个链路后再次采集并 diff，证明导出过程没有触发 ultralytics 的自动安装
（`YOLO_AUTOINSTALL=False`）。

ENV_MATRIX_PLACEHOLDER

---

## 4. 复现命令 / Reproduction

```bash
# on the L4 (or any CUDA box with /data mounted)
cd /data/yolo-master-edge && git switch dev/a3-smoke
bash scripts/a3/setup_l4.sh                       # 1. environment (idempotent)
. /root/a3venv/bin/activate
nohup bash scripts/a3/run_all_l4.sh > a3/logs/run_all.log 2>&1 &   # 2. the whole chain
# STAGES=env,smoke,ladder,qdq,backend  (subset with STAGES=smoke etc.)
```

链路内每一步的独立命令 / Each stage on its own:

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
python scripts/a3/qat_moe.py --model <pt> --data a3/config/coco-a3.yaml --out runs/a3/<m>/qdq \
    --calib-n 1024 --skip-train                         # calibrate-only explicit Q/DQ
python scripts/a3/backend_val.py --model <pt> --onnx runs/a3/<m>/<stem>.onnx \
    --data a3/config/coco-a3.yaml --json a3/results/backend_<m>.json
```

---

## 5. 配置文件 / Config

- `a3/config/coco-a3.yaml`: COCO 路径，`train: images/train2017`（仅用于校准），`val: images/val2017`（评测）。
- `a3/config/sensitive_sets.md`: 校准策略（>=1024 张训练集图、entropy、显式 per-channel Q/DQ）与敏感层集合
  （stem 两层 + 路由器 + DFL 保持 FP16）。
- 评测协议统一：ultralytics `.val`，imgsz 640，batch 1，val2017 全量 5000 张；PyTorch / ONNX Runtime /
  TensorRT engine 走同一条路径，所有精度差可直接相减。

---

## 6. 结果证据 / Evidence

RESULTS_PLACEHOLDER

---

## 7. 设计说明 / Design notes

### 7.1 动态路由导出策略（显式，不允许静默改变语义）

导出器（`ultralytics/engine/exporter.py`）不含任何 MoE 逻辑；每个 MoE 族在自身 `forward` 里用
`torch.onnx.is_in_onnx_export()` 选择导出分支。`smoke_export.py` 在调用同一个 `YOLO.export`
之前把这件事变成显式决策，并把依据（文件、运行时重新 grep 出的行号、commit）写进结果 JSON：

| MoE 族 | 导出分支行为（锁定 commit 下） | 策略 | 语义变化 |
|---|---|---|---|
| `OptimizedMOEImproved`（v0.1 系列） | 计算全部专家后 `torch.gather` 取 top-k（`selected = torch.gather(all_outs, ...)`） | `preserve` | 无（精确 top-k） |
| `ES_MOE`（EsMoE 系列） | 强制走 `_dense_forward`：softmax 加权求和全部专家，跳过 `_sparse_forward` 的 top-k / `dynamic_threshold` | `dense`，仅当 `k == E` 且未启用 top-k / 阈值时视为无语义变化（发布的 COCO 权重满足：E=3, k=3, `use_top_k=False`, 无 `dynamic_threshold`） | 否则必须用 `--routing dense` 显式声明，并报告 eager-sparse vs eager-dense 的 AP 差 |
| 其它族 | 无经过验证的导出分支 | `reject` | - |
| 任何族 + `--dynamic` | gather 分支把 B/H/W 以 Python int 写死在 `view/expand` 里 | **导出前拒绝**（exit 2） | 动态轴图在其它形状下会静默出错 |

前置条件：torch < 2.9（`ultralytics/utils/export/engine.py` 对 torch>=2.4 强制 `dynamo=False`；
dynamo 导出器不经过上述 guard，会静默丢专家）。

### 7.2 导出后验证（三层）
1. ONNX 算子直方图：记录 TopK / Gather / GatherElements / Softmax 数量；出现 `If / Loop / NonZero`
   即判失败（说明 guard 没生效）。
2. 专家卷积计数：图中 `/experts` 前缀的 Conv 数对比模块中 `*.experts.*` 下的 `nn.Conv2d` 数。
3. ORT 对齐：真实 val 图上 ONNX Runtime 输出 vs **eager PyTorch（保留其原生稀疏语义）**，
   报告 max-abs / max-rel / top-100 anchor 一致率。

### 7.3 发布权重的两个修复（不修复则结论全错）
- v0.1-N：checkpoint 早于 `add_residual` 属性，兼容 shim 默认 `True`，分类置信度坍缩到 ~0.04，
  mAP50-95 从 0.43 变成 0.007。所有脚本加载后对缺失该属性的块强制 `False`（复用
  `scripts/project03/diagnose_moe._fix_add_residual`）。
- EsMoE-N：评测与导出统一走 dense 路径（`use_sparse_inference=False`），与 7.1 的判定一致。

### 7.4 INT8 方法为什么是"只校准的显式 Q/DQ"
project03 在剪枝 v0.1-N (COCO, A100, TRT 10.13.3.9) 上的结论：隐式 PTQ 无法把 MoE 块量化
（路由 gather/expand 打断 INT8 链，TensorRT 静默回退 FP16）；显式 per-channel Q/DQ 达到 -0.80 AP；
3 轮 QAT 反而退化到 -3.2 AP。因此本分支只做校准（`qat_moe.py --skip-train`），并把隐式 INT8
作为"失败算子"对照档保留在阶梯里（engine inspector 审计会给出每层精度）。

---

## 8. 风险与降级 / Risks and fallbacks

| 风险 | 处理 |
|---|---|
| 动态轴导出 | 策略拒绝；只导出静态 1x3x640x640。需要多分辨率时按分辨率分别导出。 |
| TensorRT 隐式 INT8 静默把 MoE 块留在 FP16 | 用 engine inspector 审计逐层精度并写入日志；需要真正 INT8 的 MoE 块时只走显式 Q/DQ。 |
| 其它 ES_MOE 权重启用 top-k / 阈值 | 默认拒绝；`--routing dense` 可声明回退，但必须附 eager-sparse vs eager-dense 的 AP 差。 |
| torch >= 2.9 / dynamo 导出器 | 脚本前置断言拒绝运行。 |
| modelopt / TensorRT / ORT 版本 | `setup_l4.sh` 固定版本，`pip-freeze` 前后 diff；L4 与 A100 (project03) 数字不可直接比较，只比较同机阶梯内的差。 |
| MoT 族（P1 要求）没有训练权重 | 本阶段不做；MoT 在 TorchScript/CoreML trace 下会丢专家（仅 ONNX guard 生效），P1 需先解决权重来源。 |
| `/data` 为网络卷，5000 张评测 + 1024 张校准 I/O 慢 | 可 rsync 到本地盘后改 yaml；校准列表由确定性等距采样生成，可复现。 |

---

## 9. 后续 P1 / P2 / Next

- P1: 在同一链路上加入 MoT（需训练权重）；记录失败算子与体积/时延。
- P2: 路由一致率工具：把 TopK 索引张量作为额外图输出导出，逐图对比 FP32 ORT 与 INT8 engine 的
  专家选择一致率（v0.1 系列有真实 top-k；EsMoE 的 k==E 没有"选择"可比，改比路由权重漂移）；
  五族对比；混合精度敏感层回退已有 `quantize_trt.py --bisect/--ablate` 工具链。

---

## 10. 现有能力 / Existing runtime (unchanged on this branch)

跨平台推理运行时（C++17，ONNX Runtime / NCNN / MNN / TensorRT / CoreML；Linux、Windows、Jetson、
macOS；CPU / CUDA / Metal），Windows GUI，Jetson Orin 原生 TensorRT 部署脚本（`jetson/`），
CoreML 导出与 macOS 应用（`mac/`），VisDrone / SKU-110K / AI-TOD-v2 的基准结果。
详见 `TECHNICAL_REPORT.md`、`jetson/README.md`、`mac/README.md` 以及 `main` 分支 README。
