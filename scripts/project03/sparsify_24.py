#!/usr/bin/env python
"""2:4 structured sparsity for pruned YOLO-Master checkpoints (project03).

One-shot magnitude masking to the 2:4 pattern TensorRT's sparse tensor cores
require: within every group of 4 consecutive weights along the GEMM reduction
axis (K = in_channels * kh * kw for convs), the 2 smallest magnitudes are
zeroed. Uses modelopt's sparsity module when available (pattern-exact),
otherwise a self-contained magnitude masker.

Risk profile (measured elsewhere, to be confirmed per-model with --eval):
one-shot typically costs 2-5 AP on detection; a short MASK-FROZEN plain-fp
finetune (ordinary training - NOT QAT, which is verified harmful on this MoE
family) recovers to ~0-0.5. Expected TRT payoff: ~10-20% model-level on
Ampere+ sparse tensor cores, composable with INT8.

Skips: depthwise convs, layers with K % 4 != 0, and the accuracy-critical
regions (stem pair, routing, DFL) - sparsifying what we refuse to quantize
would be incoherent.

Usage (GPU pod, PYTHONPATH=/data/YOLO-Master, /root/p03 venv):
  python scripts/project03/sparsify_24.py \
    --model runs/project03/pod-prune/scaling-S/YOLO-Master-v0.1-S_pruned_t0.05.pt \
    --out runs/project03/sparse/v01-S --eval --data /data/tmp/coco-val.yaml
Then: quantize/build with TRT_SPARSE=1 (quantize_trt.py) or
      trtexec --sparsity=enable on the exported ONNX (jetson script SPARSE=1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

PROTECT = ("model.0.", "model.1.", "routing", "router", "dfl")


def mask_24(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return 2:4-masked copy of a conv weight [O, I, kh, kw] and its sparsity."""
    O = w.shape[0]
    flat = w.detach().reshape(O, -1)               # [O, K]
    K = flat.shape[1]
    g = flat.reshape(O, K // 4, 4)
    idx = g.abs().argsort(dim=2)                   # ascending: first 2 = drop
    mask = torch.ones_like(g)
    mask.scatter_(2, idx[:, :, :2], 0.0)
    out = (g * mask).reshape(O, K).reshape_as(w)
    return out, float((out == 0).float().mean())


def sparsify(net: nn.Module) -> dict:
    stats = {"masked": 0, "skipped_protected": 0, "skipped_shape": 0}
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if any(t in name for t in PROTECT):
            stats["skipped_protected"] += 1
            continue
        K = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        if m.groups != 1 or K % 4 != 0:
            stats["skipped_shape"] += 1
            continue
        with torch.no_grad():
            masked, sp = mask_24(m.weight.data)
            m.weight.data.copy_(masked)
        stats["masked"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--data", default="/data/tmp/coco-val.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    yolo = YOLO(args.model)
    net = yolo.model.float().eval()

    # prefer modelopt's exact sparsifier when present
    used = "manual"
    try:
        import modelopt.torch.sparsity as mts

        mts.sparsify(net, mode="sparse_magnitude")
        used = "modelopt sparse_magnitude"
        stats = {}
    except Exception:
        stats = sparsify(net)
    print(f"[24] sparsifier: {used}  {stats}")

    # verify the pattern on a sample layer
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and not any(t in name for t in PROTECT) \
                and m.groups == 1 and (m.in_channels * m.kernel_size[0] * m.kernel_size[1]) % 4 == 0:
            g = m.weight.data.reshape(m.out_channels, -1)
            g = g.reshape(g.shape[0], g.shape[1] // 4, 4)
            ok = bool(((g == 0).sum(dim=2) >= 2).all())
            print(f"[24] pattern check on {name}: {'OK' if ok else 'VIOLATED'}")
            break

    ckpt_path = out / (Path(args.model).stem + "_sp24.pt")
    torch.save({"model": net, "sparsity": "2:4 one-shot magnitude"}, ckpt_path)
    print(f"[24] saved {ckpt_path}")

    if args.eval:
        r = yolo.val(data=args.data, imgsz=args.imgsz, batch=16,
                     workers=args.workers, device=0, verbose=False, plots=False)
        print(f"[24] one-shot mAP50 {r.box.map50:.4f}  mAP50-95 {r.box.map:.4f}")

    # ONNX for TRT (--sparsity=enable detects the pattern in the weights)
    head = net.model[-1]
    head.export = True
    head.format = "onnx"
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    onnx_path = out / (Path(args.model).stem + "_sp24.onnx")
    torch.onnx.export(net.cpu(), dummy, str(onnx_path), opset_version=17,
                      input_names=["images"], output_names=["output0"],
                      do_constant_folding=True)
    print(f"[24] exported {onnx_path}")


if __name__ == "__main__":
    main()
