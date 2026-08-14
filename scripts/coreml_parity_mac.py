#!/usr/bin/env python
"""Core ML parity check for the v26.08 mixture .mlpackage exports. Runs on macOS.

Self-contained: needs only `pip install coremltools numpy` (python 3.9+). All inputs are
precomputed and shipped in the kit, so there is no image preprocessing on this side and
no drift from resize interpolation differences:

  <name>.mlpackage   the converted model (fp32 mlprogram)
  <name>.ref.npz     torch-eager references from the export env: per image stem,
                     <stem>_raw (raw det tensor), <stem>_xyxy/_conf/_cls (predict boxes)
  inputs.npz         per image stem: <stem>_blob [1,3,640,640] float32 (the exact
                     letterboxed input) and <stem>_lb [r, px, py, w, h]

Checks per model, mirroring scripts/validate_mixture.py levels A and B:
  A  raw det tensor: CoreML vs torch-eager max abs diff < --tol (default 5e-3 at fp32).
     For end2end models with a tie-degenerate untrained trunk (>50 percent of rows share
     one score), the in-graph top-k row set is implementation defined, so A is skipped
     and B still binds - identical rule to the ORT/MNN harness.
  B  every torch-predict reference box must appear among the CoreML decoded candidates
     (same class, IoU >= 0.9); mean match rate >= 0.9.

Usage:  python coreml_parity_mac.py [--dir .] [--only mot-n,moa-n] [--compute cpu|all]
Exit code 0 = all pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def decode_np(out, lb, conf):
    """numpy replica of the edge decode for both layouts -> (xyxy, conf, cls)."""
    r, px, py, w, h = lb
    if out.ndim == 3:
        out = out[0]
    if out.shape[-1] == 6 and out.shape[0] >= 32:          # end2end [N, 6]
        keep = out[:, 4] >= conf
        d = out[keep]
        xyxy = (d[:, :4] - [px, py, px, py]) / r
        return clip_xyxy(xyxy, w, h), d[:, 4], d[:, 5].astype(int)
    scores = out[4:, :]                                     # [4+nc, anchors]
    cls = scores.argmax(0)
    sc = scores.max(0)
    keep = sc >= conf
    cx, cy, bw, bh = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
    x0 = (cx - bw / 2 - px) / r
    y0 = (cy - bh / 2 - py) / r
    xyxy = np.stack([x0, y0, x0 + bw / r, y0 + bh / r], 1)
    return clip_xyxy(xyxy, w, h), sc[keep], cls[keep]


def clip_xyxy(xyxy, w, h):
    if len(xyxy) == 0:
        return xyxy
    out = xyxy.copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    ar_a = np.prod(a[:, 2:] - a[:, :2], 1)
    ar_b = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


def match_rate(ref_xyxy, ref_cls, got_xyxy, got_cls, iou_thr=0.9):
    """fraction of ref boxes with a same-class IoU>=thr counterpart (greedy)."""
    if len(ref_xyxy) == 0:
        return 1.0
    m = iou_matrix(ref_xyxy, got_xyxy)
    m[ref_cls[:, None] != got_cls[None, :]] = 0
    matched = 0
    used = set()
    for i in np.argsort(-m.max(1) if m.size else []):
        for j in np.argsort(-m[i]) if m.size else []:
            if m[i, j] < iou_thr:
                break
            if j not in used:
                used.add(j)
                matched += 1
                break
    return matched / len(ref_xyxy)


def main():
    ap = argparse.ArgumentParser(description="CoreML mixture parity check (macOS)")
    ap.add_argument("--dir", default=".", help="kit directory (mlpackages + npz files)")
    ap.add_argument("--only", default="", help="comma-separated subset of model stems")
    ap.add_argument("--compute", default="cpu", choices=["cpu", "all"],
                    help="cpu = CPU_ONLY fp32 (parity-grade, default); all = GPU/ANE allowed")
    ap.add_argument("--conf", type=float, default=0.25, help="must match the export reference conf")
    ap.add_argument("--tol", type=float, default=5e-3, help="raw max abs diff threshold")
    args = ap.parse_args()

    import coremltools as ct

    kit = Path(args.dir)
    inputs = np.load(kit / "inputs.npz")
    stems = sorted({k.rsplit("_", 1)[0] for k in inputs.files})
    models = sorted(kit.glob("*.mlpackage"))
    only = {s for s in args.only.split(",") if s}
    cu = ct.ComputeUnit.CPU_ONLY if args.compute == "cpu" else ct.ComputeUnit.ALL

    failures = []
    print(f"{'model':26s} {'layout':10s} {'A raw maxdiff':>14s} {'B box match':>12s}  verdict")
    for pkg in models:
        stem = pkg.stem
        if only and stem not in only:
            continue
        ref_path = kit / f"{stem}.ref.npz"
        if not ref_path.exists():
            failures.append(f"{stem}: missing {ref_path.name}")
            print(f"{stem:26s} missing reference npz")
            continue
        ref = np.load(ref_path)
        try:
            ml = ct.models.MLModel(str(pkg), compute_units=cu)
        except Exception as e:
            failures.append(f"{stem}: load failed: {e}")
            print(f"{stem:26s} LOAD FAILED: {e}")
            continue
        out_name = ml.user_defined_metadata.get("output") or ml.get_spec().description.output[0].name

        rates_a, rates_b = [], []
        layout = "?"
        tie_note = ""
        for s in stems:
            blob = inputs[s + "_blob"]
            lb = inputs[s + "_lb"]
            out = ml.predict({"images": blob})[out_name]
            out = np.asarray(out, dtype=np.float32)
            e2e = out.ndim == 3 and out.shape[2] == 6 and out.shape[1] >= 32
            layout = "end2end" if e2e else "anchors"
            if s + "_raw" in ref:
                tie_skip = False
                if e2e:
                    sc = out[0][:, 4]
                    _, counts = np.unique(np.round(sc, 5), return_counts=True)
                    tie_skip = counts.max() > 0.5 * len(sc)
                    if tie_skip:
                        tie_note = " (A skipped: tie-degenerate top-k)"
                if not tie_skip:
                    rates_a.append(float(np.abs(out - ref[s + "_raw"]).max()))
            c_xyxy, c_conf, c_cls = decode_np(out, lb, args.conf)
            r_xyxy, r_cls = ref[s + "_xyxy"], ref[s + "_cls"].astype(int)
            if len(r_xyxy) and len(c_xyxy) == 0:
                rates_b.append(0.0)
            else:
                rates_b.append(match_rate(r_xyxy, r_cls, c_xyxy, c_cls, iou_thr=0.9))

        a = max(rates_a) if rates_a else float("nan")
        b = float(np.mean(rates_b)) if rates_b else float("nan")
        ok = (not rates_a or a < args.tol) and (rates_b and b >= 0.9)
        if not ok:
            failures.append(f"{stem}: A={a:.2e} B={b:.3f}")
        print(f"{stem:26s} {layout:10s} {a:14.2e} {b:12.3f}  {'PASS' if ok else 'FAIL'}{tie_note}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("all models passed")


if __name__ == "__main__":
    main()
