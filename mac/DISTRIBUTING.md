# Shipping YOLO-Master CoreML Runner.app

`mac/make_app.sh` builds and packages a **universal** (Apple Silicon + Intel) macOS app
into `mac/dist/YOLO-Master CoreML Runner.app` and a redistributable `mac/dist/YOLO-Master-CoreML-Runner-<version>.zip`.

## App icon
The Dock/Finder icon is `Resources/AppIcon.icns`, packed from the finished 1024×1024 master
`Resources/AppIcon.png` (exported from the Icon Composer source in `AppIcon.icon/`). `make_app.sh`
copies the `.icns` into the bundle and sets `CFBundleIconFile`. To change the artwork, drop a new
1024² PNG at `Resources/AppIcon.png` (e.g. export from Icon Composer) then regenerate:

```bash
cd mac && python3 scripts/make_icon.py     # needs Python + Pillow; resize-only PNG->icns, no distortion
```

## Requirements
- **Build machine:** macOS with the Swift toolchain + Command Line Tools (`xcode-select --install`).
- **Recipient machine:** any Mac on **macOS 14 (Sonoma) or later**, Apple Silicon *or* Intel. No
  runtime dependencies — the Core ML backend is built in.

## Build & package

```bash
# Universal, ad-hoc signed (personal / internal sharing)
mac/make_app.sh 1.0.0

# Apple-Silicon only (smaller, faster build)
ARCHS=arm64 mac/make_app.sh 1.0.0
```

Send the resulting `mac/dist/YOLO-Master-CoreML-Runner-1.0.0.zip`.

## First launch (ad-hoc / unsigned builds)
Ad-hoc signed apps are not notarized, so Gatekeeper blocks a double-click on another Mac.
The recipient launches it once via **right-click (or Control-click) the app → Open → Open**.
After that first approval it opens normally. The camera prompt appears the first time they
start **Live Camera** (on-device only; frames never leave the Mac).

## Friction-free distribution (Developer ID + notarization)
To ship without the right-click step, sign with a Developer ID and notarize. Needs a paid Apple
Developer account. **No App ID / provisioning profile is required** — Developer-ID direct
distribution signs against your certificate, and the bundle id can be anything.

One-time setup:
1. Create a **Developer ID Application** certificate: Xcode > Settings > Accounts > Manage
   Certificates > `+` > Developer ID Application (installs cert + key into your login keychain).
2. Store notary credentials as a keychain profile:
   ```bash
   xcrun notarytool store-credentials ac-notary \
     --apple-id you@example.com --team-id TEAMID --password APP_SPECIFIC_PASSWORD
   ```
   (app-specific password from appleid.apple.com > Sign-In & Security > App-Specific Passwords)

Then every release is one command — it auto-detects your signing identity, builds universal,
signs (hardened runtime), notarizes, staples, zips, and verifies:
```bash
mac/scripts/release.sh 1.0.0
```
Overrides if needed: `NOTARY_PROFILE=...`, `CODESIGN_ID="Developer ID Application: ... (TEAMID)"`,
`BUNDLE_ID=com.you.app`, `ARCHS=arm64`.

The stapled `dist/YOLO-Master-CoreML-Runner-1.0.0.zip` opens with a normal double-click on any Mac (Sonoma+,
Apple Silicon or Intel). Publish it via GitHub Releases:
```bash
gh release create v1.0.0 mac/dist/YOLO-Master-CoreML-Runner-1.0.0.zip --title "YOLO-Master CoreML Runner 1.0.0" --notes "..."
```
(`dist/` is gitignored — bundles are release assets, not tracked in git.)
