# YOLO-Master iOS

Live camera detection + on-device benchmark. The inference backend is
`YOLOMasterKit` (../mac/Sources/YOLOMasterKit) **verbatim** - the exact
letterbox -> Core ML -> decode -> NMS path the macOS CLI and GUI run; this
directory adds only the iOS shell (AVFoundation camera, SwiftUI views, bench
harness).

## Build (on the Mac)

```bash
brew install xcodegen              # once
# segmentation model (in git): enough on its own to build and run the app
cp -R mac/Resources/v0.1-seg-N.mlpackage ios/Models/
# detection models are NOT in git: export your own with the iOS 17 target
python coreml_export/export_coreml.py --weights YOLO-Master-v0.1-N.pt --imgsz 640 \
    --target ios17 --out ios/Models/v0.1-N.mlpackage
cd ios && xcodegen                 # -> YOLOMasterIOS.xcodeproj
open YOLOMasterIOS.xcodeproj
```

In Xcode: set your signing team on the YOLOMasterIOS target, select the
iPhone, Run. (No team? Simulator works for the UI, but Core ML performance and
the ANE only exist on-device.)

## Release build (TestFlight / App Store)

A Debug build runs from a fresh clone. A Release build (what Archive produces)
also needs the inputs that are not in git, and a `Release inputs guard` build
phase fails the build with a clear error when one is missing:

- `ios/AppIcon.icon` - the Icon Composer app icon;
- at least one Core ML model in `ios/Models/` (the release bundles the
  detection packages plus `v0.1-seg-N`).

Before every upload:

1. bump `CFBundleVersion` in `project.yml` (App Store Connect refuses a build
   number it has already seen), then re-run `xcodegen`;
2. keep `PrivacyInfo.xcprivacy` in step with the code: it declares UserDefaults
   (`@AppStorage`) as the only required-reason API. Add an entry if new code
   reads file timestamps, disk space, system uptime or the keyboard list;
3. the photo library is add-only (`NSPhotoLibraryAddUsageDescription`). The
   Photo tab reads through the system picker and must not request read access;
4. the privacy policy linked from Settings is `PRIVACY.md` on the default
   branch - the same URL goes into App Store Connect.

## What the app does

- **Live tab** - camera preview with detection overlay; model and compute-unit
  (ANE / GPU / CPU) pickers; per-frame inference latency + FPS readout.
- **Photo tab** - batch detection over up to 100 album images; gallery and
  pager views; conf/IoU sliders retune from cached passes without re-running.
- **Segmentation** - bundle a seg .mlpackage (e.g. v0.1-seg-N above) and both
  tabs render instance masks tinted by class, with a Masks / Boxes / Both
  switch in the tuning panel and a mask-compose stage bar in the HUD. Same
  decode + SGEMM mask math as macOS, shared through the Kit.
- **Bench tab** - the criterion harness:
  - *cold run*: every bundled model x every compute unit, 50-iter median +
    p90 after warmup;
  - *sustained pass*: N-minute loop reporting the last-quarter median - the
    thermal steady state. Flagship phones throttle; a real-time claim needs
    both numbers (methodology carried over from the Orin work);
  - results export via the share sheet (CSV).

## Model expectations (from the measured Apple-silicon ladder)

Same .mlpackage files as macOS (converted with the iOS 17 deployment target).
On M4 Max: fp16 GPU 14.3ms / ANE 18.3ms; W8 = identical speed, smaller bundle
(15 -> 12MB); W8A8 slower everywhere (ANE fusion breaks) - do not ship A8.
On A-series the ANE/GPU ranking is expected to flip (near-desktop ANE, small
GPU): verify with the cold run, then W8-on-ANE is the likely shipping pick.

## Xcode-native alternative for per-op numbers

Select any bundled .mlpackage in Xcode -> Performance tab -> run a performance
report on the connected iPhone: per-compute-unit latency and the op-to-ANE
mapping, no code involved. Use it to confirm which ops fall off the ANE if the
bench numbers look off.
