#!/usr/bin/env python3
"""Export the v26.08 mixture families (MoA / MoT / MoLoRA) to ONNX for edge validation.

Runs inside a venv that has torch/onnx/onnxruntime but NO ultralytics installed;
--repo points at a YOLO-Master v26.08+ checkout and is prepended to sys.path, so the
research environment's editable install is never touched.

Produces, under --out:
  moa-n.onnx / mot-n.onnx / moa-mot-n.onnx      v0_10 lineage, [1, 4+nc, anchors]
  yolo26-moa-n-e2e.onnx                          yolo26 lineage, end2end [1, max_det, 6]
  yolo26-moa-n-anchors.onnx                      same cfg exported with end2end=False
  molora-merged.onnx                             MoLoRA(uniform-merged) on the MoA base
  molora-routed.onnx                             MoLoRA molora_export_mode=routing_preserved
plus, per model: <stem>.metadata.yaml (names/imgsz/end2end sidecar for MNN/ncnn) and
<stem>.ref.npz (eager predict reference on --ref-images: xyxy/conf/cls per image, the
ground truth for scripts/validate_mixture.py).

Weights are random-init (upstream ships none for these families) - detections are
arbitrary but deterministic (seeded), which is exactly what pipeline/parity validation
needs. Pass --base <pt> to try wrapping real weights for the MoLoRA pair instead.
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="YOLO-Master v26.08 checkout (prepended to sys.path)")
    ap.add_argument("--out", default="models/mixture")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ref-images", default="", help="dir with images for the predict reference (first --ref-n)")
    ap.add_argument("--ref-n", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.01, help="reference predict conf (low: random weights)")
    ap.add_argument("--base", default="", help="optional .pt to use as the MoLoRA base model")
    ap.add_argument("--only", default="", help="comma-separated subset of target names")
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.repo)
    import numpy as np
    import torch
    from ultralytics import YOLO, __version__

    print(f"[env] ultralytics {__version__} from {args.repo}")
    assert not __version__.startswith("8.3."), "old-lineage ultralytics resolved - wrong sys.path"
    torch.manual_seed(0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_imgs = []
    if args.ref_images:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        ref_imgs = sorted(p for p in Path(args.ref_images).iterdir() if p.suffix.lower() in exts)[: args.ref_n]
        print(f"[ref] {len(ref_imgs)} reference images")

    V010 = "ultralytics/cfg/models/master/v0_10/det"
    Y26 = "ultralytics/cfg/models/26"

    def build(cfg):
        torch.manual_seed(0)                      # deterministic random init per target
        return YOLO(str(Path(args.repo) / cfg))

    def build_molora(merged):
        from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
        y = build(f"{V010}/yolo-master-moa-n.yaml") if not args.base else YOLO(args.base)
        torch.manual_seed(1)
        mol = MoLoRAModel(y.model, MoLoRAConfig(r=8, alpha=16, num_experts=4, top_k=2))
        if merged:
            mol.merge(mode="uniform")             # calibration-free; works untrained
        y.model = mol.model
        return y

    TARGETS = {
        "moa-n":                (lambda: build(f"{V010}/yolo-master-moa-n.yaml"), {}),
        "mot-n":                (lambda: build(f"{V010}/yolo-master-mot-n.yaml"), {}),
        "moa-mot-n":            (lambda: build(f"{V010}/yolo-master-moa-mot-n.yaml"), {}),
        "yolo26-moa-n-e2e":     (lambda: build(f"{Y26}/yolo26-master-moa-n.yaml"), {}),
        "yolo26-moa-n-anchors": (lambda: build(f"{Y26}/yolo26-master-moa-n.yaml"), {"end2end": False}),
        "molora-merged":        (lambda: build_molora(True), {}),
        "molora-routed":        (lambda: build_molora(False), {"molora_export_mode": "routing_preserved"}),
    }
    only = {s for s in args.only.split(",") if s} or set(TARGETS)

    results = {}
    for name, (factory, kw) in TARGETS.items():
        if name not in only:
            continue
        print(f"\n=== {name} ===")
        try:
            y = factory()
            onnx_path = y.export(format="onnx", imgsz=args.imgsz, device="cpu",
                                 simplify=False, dynamic=False, **kw)
            dst = out / f"{name}.onnx"
            Path(onnx_path).replace(dst)

            # sidecar for MNN/ncnn conversions (parsed by the edge runtime's read_ncnn_yaml)
            names = y.model.names if isinstance(y.model.names, dict) else dict(enumerate(y.model.names))
            e2e = bool(getattr(y.model.model[-1], "end2end", False)) and kw.get("end2end", True) is not False
            with open(out / f"{name}.metadata.yaml", "w") as f:
                f.write(f"imgsz:\n- {args.imgsz}\n- {args.imgsz}\n")
                f.write(f"end2end: {str(e2e).lower()}\n")
                f.write("names:\n")
                for k in sorted(names):
                    f.write(f"  {k}: {names[k]}\n")

            # eager predict reference (the exported-vs-eager ground truth for the harness)
            if ref_imgs:
                ref = {}
                for img in ref_imgs:
                    r = y.predict(str(img), imgsz=args.imgsz, conf=args.conf, iou=0.7,
                                  max_det=300, device="cpu", verbose=False)[0]
                    b = r.boxes
                    ref[img.stem + "_xyxy"] = b.xyxy.cpu().numpy()
                    ref[img.stem + "_conf"] = b.conf.cpu().numpy()
                    ref[img.stem + "_cls"] = b.cls.cpu().numpy()
                np.savez(out / f"{name}.ref.npz", **ref)

            results[name] = f"OK ({dst.stat().st_size // 1024} KB)"
        except Exception as e:  # noqa: BLE001 - a per-target failure must not kill the sweep
            results[name] = f"FAILED: {type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()

    print("\n===== export summary =====")
    for name, res in results.items():
        print(f"  {name:24s} {res}")
    return 0 if all(r.startswith("OK") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
