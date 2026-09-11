#!/usr/bin/env python3
"""Numeric parity of two prediction-dump directories ('class conf x1 y1 x2 y2' per line).

    python scripts/server/parity_txt.py cli_txt api_txt [--tol 0.01]
Prints files compared, files with a count/class mismatch, max box and conf deltas; exit 1 on mismatch.
"""
import sys, argparse
from pathlib import Path


def load(p):
    rows = []
    for line in Path(p).read_text().splitlines():
        t = line.split()
        if len(t) == 6:
            rows.append((int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]), float(t[5])))
    rows.sort(key=lambda r: (-r[1], r[0], r[2], r[3]))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--conf-tol", type=float, default=1e-4)
    x = ap.parse_args()
    fa = {p.name: p for p in Path(x.a).glob("*.txt")}; fb = {p.name: p for p in Path(x.b).glob("*.txt")}
    common = sorted(set(fa) & set(fb))
    missing = sorted((set(fa) | set(fb)) - set(common))
    bad_count = bad_cls = bad_box = 0; max_box = max_conf = 0.0; dets = 0
    for n in common:
        ra, rb = load(fa[n]), load(fb[n])
        if len(ra) != len(rb):
            bad_count += 1; continue
        for a, b in zip(ra, rb):
            dets += 1
            if a[0] != b[0]:
                bad_cls += 1; continue
            max_conf = max(max_conf, abs(a[1] - b[1]))
            d = max(abs(a[i] - b[i]) for i in range(2, 6))
            max_box = max(max_box, d)
            if d > x.tol:
                bad_box += 1
    ok = not missing and bad_count == 0 and bad_cls == 0 and bad_box == 0 and max_conf <= x.conf_tol
    print(f"files={len(common)} missing={len(missing)} count_mismatch={bad_count} class_mismatch={bad_cls} "
          f"dets={dets} box_mismatch={bad_box} max_box_delta={max_box:.5f} max_conf_delta={max_conf:.6f} parity={'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
