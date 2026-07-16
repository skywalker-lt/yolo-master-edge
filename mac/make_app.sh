#!/usr/bin/env bash
# Build + package YOLOMaster.app as a redistributable, universal (Apple Silicon + Intel) macOS app.
# Run on macOS with the Swift toolchain + Command Line Tools. Usage: mac/make_app.sh [version]
#
#   Basic (ad-hoc signed; recipients right-click > Open on first launch):
#       mac/make_app.sh 1.0.0
#   Apple-Silicon only (smaller/faster build):
#       ARCHS=arm64 mac/make_app.sh
#   Developer ID signed (Gatekeeper-clean, no right-click needed once notarized):
#       CODESIGN_ID="Developer ID Application: Your Name (TEAMID)" mac/make_app.sh 1.0.0
#   Signed + notarized (fully shippable to anyone):
#       CODESIGN_ID="Developer ID Application: ..." NOTARY_PROFILE=ac-notary mac/make_app.sh 1.0.0
#       (create the profile once: xcrun notarytool store-credentials ac-notary
#          --apple-id you@example.com --team-id TEAMID --password APP_SPECIFIC_PW)
#
# Note: intentionally NOT using `set -u` -- macOS's stock bash 3.2 aborts on benign expansions.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../mac
APP_NAME="YOLOMaster"
BUNDLE_ID="com.yolomaster.coreml"
VERSION="${1:-1.0.0}"
ARCHS="${ARCHS:-arm64 x86_64}"                 # universal by default; override e.g. ARCHS=arm64
DIST="$HERE/dist"
APP="$DIST/$APP_NAME.app"
ZIP="$DIST/${APP_NAME}-${VERSION}.zip"

archargs=""
for a in $ARCHS; do archargs="$archargs --arch $a"; done

echo "[1/4] swift build -c release (arch:$ARCHS)..."
swift build -c release --package-path "$HERE" $archargs --product YOLOMasterApp
BIN="$(swift build -c release --package-path "$HERE" $archargs --show-bin-path)/YOLOMasterApp"
[ -x "$BIN" ] || { echo "build product not found: $BIN" >&2; exit 1; }
echo "  binary: $(lipo -archs "$BIN" 2>/dev/null || echo "$ARCHS")"

echo "[2/4] assembling $APP ..."
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

echo "[3/4] codesign..."
if [ -n "${CODESIGN_ID:-}" ]; then
  codesign --force --deep --options runtime --timestamp --sign "$CODESIGN_ID" "$APP"
  echo "  signed: $CODESIGN_ID (hardened runtime)"
  codesign --verify --deep --strict "$APP" && echo "  verify: OK"
else
  codesign --force --deep --sign - "$APP" 2>/dev/null \
    && echo "  ad-hoc signed (recipients: right-click > Open on first launch)" \
    || echo "  (codesign unavailable -- app still runs locally)"
fi

echo "[4/4] packaging -> $ZIP"
mkdir -p "$DIST"; rm -f "$ZIP"
/usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"

if [ -n "${NOTARY_PROFILE:-}" ]; then
  echo "[notarize] submitting to Apple (this can take a few minutes)..."
  xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
  rm -f "$ZIP"; /usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"   # re-zip the stapled app
  echo "  notarized + stapled."
fi

echo
echo "done."
echo "  app: $APP"
echo "  zip: $ZIP   ($(du -h "$ZIP" 2>/dev/null | cut -f1))"
echo "  run locally:  open \"$APP\""
echo "  ship:         send the .zip. If ad-hoc/unsigned, tell the recipient to right-click the app"
echo "                and choose Open on first launch (Gatekeeper). Developer ID + notarize = no prompt."
