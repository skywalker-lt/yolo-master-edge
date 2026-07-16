# Shipping YOLOMaster.app

`mac/make_app.sh` builds and packages a **universal** (Apple Silicon + Intel) macOS app
into `mac/dist/YOLOMaster.app` and a redistributable `mac/dist/YOLOMaster-<version>.zip`.

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

Send the resulting `mac/dist/YOLOMaster-1.0.0.zip`.

## First launch (ad-hoc / unsigned builds)
Ad-hoc signed apps are not notarized, so Gatekeeper blocks a double-click on another Mac.
The recipient launches it once via **right-click (or Control-click) the app → Open → Open**.
After that first approval it opens normally. The camera prompt appears the first time they
start **Live Camera** (on-device only; frames never leave the Mac).

## Friction-free distribution (Developer ID + notarization)
To ship without the right-click step, sign with a Developer ID and notarize (needs a paid
Apple Developer account):

```bash
# one-time: store notarization credentials in the keychain
xcrun notarytool store-credentials ac-notary \
  --apple-id you@example.com --team-id TEAMID --password APP_SPECIFIC_PASSWORD

# build -> sign (hardened runtime) -> notarize -> staple -> zip
CODESIGN_ID="Developer ID Application: Your Name (TEAMID)" \
NOTARY_PROFILE=ac-notary \
mac/make_app.sh 1.0.0
```

The stapled zip opens with a normal double-click on any Mac. Distribute via GitHub Releases
(the `dist/` folder is gitignored — bundles are release assets, not tracked in git).
