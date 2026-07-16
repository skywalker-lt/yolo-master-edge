#!/usr/bin/env bash
# Build YOLOMasterApp and assemble a double-clickable, redistributable YOLOMaster.app.
# Run on macOS (needs the Swift toolchain + Core ML). Usage: mac/make_app.sh [version]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../mac
APP_NAME="YOLOMaster"
BUNDLE_ID="com.yolomaster.coreml"
VERSION="${1:-1.0.0}"

echo "[1/3] swift build -c release (YOLOMasterApp)…"
swift build -c release --package-path "$HERE" --product YOLOMasterApp
BIN="$(swift build -c release --package-path "$HERE" --show-bin-path)/YOLOMasterApp"
[ -x "$BIN" ] || { echo "build product not found: $BIN" >&2; exit 1; }

APP="$HERE/dist/$APP_NAME.app"
echo "[2/3] assembling $APP…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/$APP_NAME"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key>       <string>YOLO-Master</string>
  <key>CFBundleExecutable</key>        <string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key>        <string>${BUNDLE_ID}</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key>           <string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key>    <string>14.0</string>
  <key>NSCameraUsageDescription</key>  <string>Live real-time object detection runs the selected Core ML model on the camera feed. Frames are processed on-device and never leave your Mac.</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>LSApplicationCategoryType</key> <string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

echo "[3/3] ad-hoc codesign (lets it run locally; re-sign with a Developer ID to distribute)…"
codesign --force --deep --sign - "$APP" 2>/dev/null || echo "  (codesign skipped — app still runs locally)"

echo "done -> $APP"
echo "  run:  open \"$APP\""
echo "  ship: zip -r ${APP_NAME}.zip \"$APP\"  (recipients right-click > Open on first launch;"
echo "        sign + notarize with a Developer ID for Gatekeeper-clean distribution)"
