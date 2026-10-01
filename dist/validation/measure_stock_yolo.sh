#!/usr/bin/env bash
# The v0.1-N protocol applied to the stock nano models: TensorRT via the v1.2.0 CLI, CUDA preprocessing,
# fp16 and fp32, with and without --cuda-graph; warm 200-frame timing on coco500, bench cold median,
# coco500 accuracy (val protocol). Engines are built on this GPU on first use and cached beside the .onnx.
set -uo pipefail
BIN=/root/build/yolomaster_edge; X=/root/xfer; C=/root/coco500/coco500/images; L=/root/measure_stock.log
: > $L
log() { echo "$*" | tee -a $L; }
log "host: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | $(/root/build/yolomaster_edge --version 2>&1 | head -1) | TensorRT $(dpkg -s libnvinfer10 | awk '/^Version/{print $2}')"
for m in yolo11n yolo26n yolov12n yolov13n; do
  M=$X/$m.onnx
  for prec in fp16 fp32; do
    pa="-b trt --precision $prec"
    # engine build happens inside this first run; record the build time line
    t0=$(date +%s)
    first="$(timeout 1800 $BIN -m $M -s $C $pa --no-save --quiet --limit 5 2>&1 | grep -E "\[trt\] built|\[model\]" | sed -E 's/\[model\] [^ ]+ +//' | tr '\n' ' ' | cut -c1-220)"
    log "== $m $prec: $first (engine step $(( $(date +%s) - t0 )) s)"
    for extra in "" "--cuda-graph"; do
      t="$(timeout 900 $BIN -m $M -s $C $pa $extra --no-save --quiet --warmup 10 --limit 200 2>&1 | grep -E '^\[summary\]' | sed -E 's/.*avg\/frame: //; s/ +wall=.*//')"
      b="$(timeout 900 $BIN -m $M -s $C $pa $extra --bench cold --bench-iters 100 --bench-warmup 10 --bench-json /root/bench_${m}_${prec}${extra:+_graph}.json --no-save --quiet 2>&1 | grep -oE 'median=[0-9.]+ms p90=[0-9.]+' | head -1)"
      log "  $m $prec ${extra:-plain}: $t | bench $b"
    done
    a="$(timeout 2400 $BIN -m $M -s $C $pa --accuracy auto --no-save --quiet 2>&1 | grep -E '^\[accuracy\]' | sed -E 's/\[accuracy\] images=500 +//')"
    log "  $m $prec accuracy coco500: $a"
  done
done
log "MEASURE DONE $(date +%H:%M:%S)"
