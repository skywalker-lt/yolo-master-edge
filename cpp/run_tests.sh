#!/usr/bin/env bash
# Robustness battery for yolomaster_edge. Re-runnable on any platform (x86_64 / Jetson).
# Usage: BIN=./build/yolomaster_edge ONNX=... NCNN=... DIR=... YAML=... ./run_tests.sh
set -u
ROOT=/data/yolo-master-edge
BIN=${BIN:-$ROOT/cpp/build/yolomaster_edge}
ONNX=${ONNX:-$ROOT/models/esmoe_n_visdrone_sim.onnx}
NCNN=${NCNN:-$ROOT/models/esmoe_n_visdrone_ncnn}
DIR=${DIR:-$ROOT/visdrone50/images/val}
YAML=${YAML:-$ROOT/visdrone50/visdrone50.yaml}
OUT=$(mktemp -d)
IMG=$(ls "$DIR"/*.jpg | sort | head -1)
P=0; F=0
ok(){ echo "  PASS  $1"; P=$((P+1)); }
no(){ echo "  FAIL  $1"; F=$((F+1)); }
run(){ "$BIN" "$@" 2>&1; }

# build a 6-frame test video if opencv-python is present
python - "$DIR" "$OUT/test.mp4" <<'PY' 2>/dev/null || true
import cv2,glob,sys
imgs=sorted(glob.glob(sys.argv[1]+"/*.jpg"))[:6]
vw=cv2.VideoWriter(sys.argv[2],cv2.VideoWriter_fourcc(*'mp4v'),5,(640,480))
for p in imgs: vw.write(cv2.resize(cv2.imread(p),(640,480)))
vw.release()
PY

echo "== sources & auto-detection =="
run -m "$ONNX" -s "$IMG" --no-save | grep -q "backend=onnx.*model-metadata" && ok "T1 onnx auto backend+classes" || no T1
run -m "$NCNN" -s "$IMG" --no-save | grep -q "backend=ncnn.*model-metadata" && ok "T2 ncnn auto backend+classes" || no T2
run -m "$ONNX" -s "$DIR" --limit 4 --quiet --no-save | grep -q "frames=4" && ok "T3 directory source" || no T3
run -m "$NCNN" -s "$YAML" --limit 3 --quiet --no-save | grep -q "frames=3" && ok "T4 dataset.yaml source" || no T4
[ -f "$OUT/test.mp4" ] && { run -m "$ONNX" -s "$OUT/test.mp4" --quiet --no-save | grep -q "frames=6" && ok "T5 video source" || no T5; } || echo "  SKIP  T5 (no video)"

echo "== parity (post-refactor) =="
c1=$(run -m "$ONNX" -s "$IMG" --no-save | grep -oE "total_dets=[0-9]+")
c2=$(run -m "$NCNN" -s "$IMG" --no-save | grep -oE "total_dets=[0-9]+")
[ -n "$c1" ] && [ "$c1" = "$c2" ] && ok "T6 onnx==ncnn ($c1)" || no "T6 parity ($c1 vs $c2)"

echo "== overrides =="
run -m "$ONNX" -s "$IMG" --classes sku --conf 0.5 --no-save | grep -qE "nc=1 \(flag:sku\)  conf=0.5" && ok "T7 --classes/--conf override" || no T7
run -m "$ONNX" -s "$IMG" --imgsz 512 --no-save | grep -q "requires fixed imgsz" && ok "T8 imgsz auto-align warn" || no T8

echo "== error handling / robustness =="
run -m /nope/x.onnx -s "$IMG" --no-save >/dev/null 2>&1; [ $? -ne 0 ] && ok "T9 missing model -> nonzero" || no T9
run -m "$ONNX" -s /nope/x.jpg --no-save >/dev/null 2>&1; [ $? -ne 0 ] && ok "T10 missing source -> nonzero" || no T10
run -m model.bin -s "$IMG" --no-save 2>&1 | grep -qi "cannot infer backend" && ok "T11 unknown ext -> ask backend" || no T11
"$BIN" -m "$ONNX" >/dev/null 2>&1; [ $? -ne 0 ] && ok "T12 missing --source -> CLI error" || no T12
run --help 2>&1 | grep -q "universal YOLO-Master" && ok "T13 --help" || no T13
mkdir -p "$OUT/corrupt"; cp "$IMG" "$OUT/corrupt/good.jpg"; echo x > "$OUT/corrupt/bad.jpg"
run -m "$ONNX" -s "$OUT/corrupt" --no-save 2>&1 | grep -q "skip. unreadable.*bad.jpg" && ok "T14 corrupt image skipped" || no T14
"$BIN" -m "$ONNX" -s "$IMG" --imgsz 512 --no-save >/dev/null 2>&1; [ $? -eq 0 ] && ok "T15 no crash on imgsz mismatch" || no T15
run -m "$ONNX" -s "$IMG" --out "$OUT/w" >/dev/null 2>&1; ls "$OUT"/w/*.jpg >/dev/null 2>&1 && ok "T16 writes annotated output" || no T16

echo "== output-shape assertions (count what actually lands on disk) =="
# T17: a video source must yield ONE annotated mp4 (no per-frame jpg spam / overwrites)
# and one --save-txt file PER FRAME (frame-indexed names).
if [ -f "$OUT/test.mp4" ]; then
  run -m "$ONNX" -s "$OUT/test.mp4" --quiet --out "$OUT/v" --save-txt "$OUT/vtxt" >/dev/null 2>&1
  NFRAMES=$(run -m "$ONNX" -s "$OUT/test.mp4" --quiet --no-save 2>&1 | grep -o "frames=[0-9]*" | head -1 | cut -d= -f2)
  MP4S=$(ls "$OUT"/v/*_annotated.mp4 2>/dev/null | wc -l)
  JPGS=$(ls "$OUT"/v/*.jpg 2>/dev/null | wc -l)
  TXTS=$(ls "$OUT"/vtxt/*.txt 2>/dev/null | wc -l)
  [ "$MP4S" = 1 ] && [ "$JPGS" = 0 ] && [ "$TXTS" = "$NFRAMES" ] \
    && ok "T17 video -> one mp4 + per-frame txt ($NFRAMES frames)" \
    || no "T17 video output shape (mp4=$MP4S jpg=$JPGS txt=$TXTS frames=$NFRAMES)"
else
  echo "  SKIP  T17 (no test video; opencv-python missing)"
fi
# T18: duplicate stems (1.jpg + 1.png in one dir) must yield one DISTINCT output per input
# in every stem-keyed writer: annotated jpgs, --save-txt, YOLO label export.
mkdir -p "$OUT/dup"
cp "$IMG" "$OUT/dup/1.jpg"
python3 - "$IMG" "$OUT/dup/1.png" <<'PY' 2>/dev/null || cp "$IMG" "$OUT/dup/1.png"
import sys
try:
    import cv2
    cv2.imwrite(sys.argv[2], cv2.imread(sys.argv[1]))
except Exception:
    raise SystemExit(1)
PY
run -m "$ONNX" -s "$OUT/dup" --quiet --out "$OUT/dout" --save-txt "$OUT/dtxt" --export-labels "$OUT/dlbl" >/dev/null 2>&1
DJ=$(ls "$OUT"/dout/*.jpg 2>/dev/null | wc -l)
DT=$(ls "$OUT"/dtxt/*.txt 2>/dev/null | wc -l)
DL=$(ls "$OUT"/dlbl/*.txt 2>/dev/null | grep -vc classes)
[ "$DJ" = 2 ] && [ "$DT" = 2 ] && [ "$DL" = 2 ] \
  && ok "T18 duplicate stems -> distinct outputs" \
  || no "T18 duplicate-stem collision (jpg=$DJ txt=$DT lbl=$DL, want 2 each)"

rm -rf "$OUT"
echo "======================================"
echo "RESULT: $P passed, $F failed"
[ $F -eq 0 ]
