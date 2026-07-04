#!/usr/bin/env bash
# Publish a GitHub Release with both prebuilt bundles attached.
# Needs a token with contents:write on the repo (fine-grained PAT or classic 'repo').
#   GITHUB_TOKEN=ghp_xxx bash scripts/make_release.sh [tag]
set -euo pipefail
REPO="skywalker-lt/yolo-master-edge"
TAG="${1:-v0.1-edge}"
DIST="/data/yolo-master-edge/dist"
: "${GITHUB_TOKEN:?set GITHUB_TOKEN (contents:write PAT) in the environment}"
API="https://api.github.com/repos/$REPO"
H=(-H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json")

read -r -d '' NOTES <<'EOF' || true
Prebuilt, self-contained edge-inference bundles for YOLO-Master-EsMoE-N
(ONNX + ncnn backends, VisDrone, verified <0.5% mAP vs PyTorch).

- **linux-x64** (`.tar.gz`) — any Ubuntu 22.04+ x86_64, no install. `$ORIGIN`-rpath'd, 10 bundled libs.
- **win-x64** (`.zip`) — any Windows 10/11 x64, no install. Bundles the MSVC runtime.

Both ship the ONNX + ncnn models under `models/`. Run: `yolomaster_edge --model models/esmoe_n_visdrone_sim.onnx --source <img|dir> --out out`.
x86_64 only (Jetson/aarch64 is a native rebuild). See README.md for build-from-source.
EOF

echo "creating release $TAG on $REPO ..."
resp="$(curl -fsS "${H[@]}" "$API/releases" \
  -d "$(printf '{"tag_name":"%s","name":"%s","body":%s}' "$TAG" "$TAG" "$(printf '%s' "$NOTES" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")")"
up="$(printf '%s' "$resp" | python3 -c 'import json,sys;print(json.load(sys.stdin)["upload_url"].split("{")[0])')"
echo "  release created; uploading assets ..."
for f in "$DIST/yolomaster_edge-linux-x64.tar.gz" "$DIST/yolomaster_edge-win-x64.zip"; do
  name="$(basename "$f")"
  echo "  -> $name"
  curl -fsS "${H[@]}" -H "Content-Type: application/octet-stream" \
       --data-binary @"$f" "$up?name=$name" >/dev/null
done
echo "done: https://github.com/$REPO/releases/tag/$TAG"
