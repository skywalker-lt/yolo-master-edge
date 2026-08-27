#!/usr/bin/env bash
# Stage ncnn model dirs from the repo's (gitignored) models/ into the runtime's test
# assets so the on-device harness can load them. Run before ./gradlew connectedAndroidTest.
#
# Models are NOT committed (they ship as GitHub Release assets); this copies your local
# working-tree copies into assets/ where the instrumented test reads them.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DST="$REPO/android/runtime/src/main/assets/models"
mkdir -p "$DST"

# name-in-assets  <-  path-under-repo/models
STAGE=(
  "v0.1-seg-n_ncnn:v0.1-seg-n_ncnn"
  "moa-n_ncnn:mixture/moa-n_ncnn"
)

for pair in "${STAGE[@]}"; do
  name="${pair%%:*}"; rel="${pair##*:}"
  src="$REPO/models/$rel"
  if [ -d "$src" ]; then
    rm -rf "$DST/$name"
    cp -r "$src" "$DST/$name"
    echo "staged  $name"
  else
    echo "WARN: $src not found - fetch the ncnn models first" >&2
  fi
done

PROBE="$REPO/android/runtime/src/main/assets/probe.jpg"
[ -f "$PROBE" ] || echo "NOTE: drop a test image at $PROBE (any scene with objects)"
echo "done. now: cd android && ./gradlew :runtime:connectedAndroidTest"
