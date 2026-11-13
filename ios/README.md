# YOLO-Master iOS

Live camera detection + on-device benchmark. The inference backend is
`YOLOMasterKit` (../mac/Sources/YOLOMasterKit) **verbatim** - the exact
letterbox -> Core ML -> decode -> NMS path the macOS CLI and GUI run; this
directory adds only the iOS shell (AVFoundation camera, SwiftUI views, bench
harness).

## Build (on the Mac)

```bash
brew install xcodegen              # once
# No model weights are in git: they are GitHub Release assets.
# segmentation model: enough on its own to build and run the app
curl -L -o /tmp/seg.zip \
    https://github.com/skywalker-lt/yolo-master-edge/releases/download/v0.1-seg-n/v0.1-seg-N.mlpackage.zip
unzip -q /tmp/seg.zip -d ios/Models/
# detection models: export your own with the iOS 17 target
python coreml_export/export_coreml.py --weights YOLO-Master-v0.1-N.pt --imgsz 640 \
    --target ios17 --out ios/Models/v0.1-N.mlpackage
cd ios && xcodegen                 # -> YOLOMasterIOS.xcodeproj
open YOLOMasterIOS.xcodeproj
```

In Xcode: set your signing team on the YOLOMasterIOS target, select the
iPhone, Run. (No team? Simulator works for the UI, but Core ML performance and
the ANE only exist on-device.)

## Release build (TestFlight / App Store)

A Debug build runs from a fresh clone (with no model in `ios/Models/` it opens
on "No models bundled"). A Release build (what Archive produces)
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

- **Live tab** - camera preview with detection overlay; back/front camera
  toggle beside the shutter (the front preview is mirrored like a selfie view,
  the frames the model sees and the saved photos are not); model and compute-unit
  (Auto / GPU / CPU) pickers; per-frame inference latency + FPS readout. Auto is
  Core ML's own placement (`MLComputeUnits.all`): it decides which parts of the
  model run on the Neural Engine, GPU and CPU. It was labelled "ANE" up to
  1.1.0 build 1; saved benchmark history with that label still loads.
- **Photo tab** - batch detection over up to 100 album images; gallery and
  pager views; conf/IoU sliders retune from cached passes without re-running.
- **Segmentation** - bundle a seg .mlpackage (e.g. v0.1-seg-N above) and both
  tabs render instance masks tinted by class, with a Masks / Boxes / Both
  switch in the tuning panel and a mask-compose stage bar in the HUD. Same
  decode + SGEMM mask math as macOS, shared through the Kit.
- **Pose** - bundle a pose .mlpackage (e.g. `yolo26n-pose`, exported with
  `--anchors`) and both tabs draw each person's skeleton over the box: the 17
  COCO keypoints, limbs coloured by body part, joints the model is less than
  50% sure it can see left out. Saved photos carry the skeleton too. Decoding
  and the skeleton rules live in the Kit (`Detection.keypoints`, `PoseSkeleton`).
  A 133-keypoint COCO-WholeBody model draws the body skeleton plus feet, finger
  bones and the 68 face points (those at half size).
  With a pose model running, the tuning panel adds pose controls in both tabs:
  a Skeleton / Boxes / Both switch (Skeleton drops the person boxes and their
  labels, and hides the box-style row, which has nothing to style then), a
  `kpt` slider for the keypoint visibility to draw from (0.50 is the model's
  own cut-off) and a `size` slider for joints and limbs. A whole-body model
  also gets a Body / Feet / Face / Hands row - four separate buttons, blue
  while their group is drawn - and a Hand skeleton
  switch that removes the finger bones and keeps the hand points; that switch
  shares a line with Show stats HUD. Saved photos and exported images follow
  the same settings (the Kit's `PoseDisplay`).
  In Live the top controls, the tuning panel and the stats HUD are all as
  wide as the tab bar (`barSideInset`).
- **Top-down pose (DWPose)** - `dwpose-t-wholebody` (exported with
  `coreml_export/export_dwpose.py`) does not find people itself. The Kit runs a
  detection model on the frame first and the pose model once per person (up to
  8), so it needs a plain detector with a `person` class in the app: `yolo26n`
  is used when present, else `yolo11n`, `yolo12n`, or any other one. The conf
  and IoU sliders act on the person boxes (conf down to 0.10). The HUD's model
  time is both stages together. Without a detector the model is refused with
  that reason, in Live and at import. The larger models of the family
  (DWPose-m / -l, RTMW-l) run the same way and place hands much better at a
  higher cost per person; `coreml_export/README.md` has the comparison.
  These models always return all 133 points, also for parts that are not in
  the frame, so the feet, the face and each hand are shown or hidden as a
  whole by the part's typical score (a frame filled by one hand shows that
  hand, not an invented body). Each package carries its own score cut-off.
- **NMS-free models** - a model in the end-to-end layout (`end2end=true` in its
  metadata: YOLO26 / YOLOv10 exported without `--anchors`) emits one row per
  object. The app runs no NMS on it and hides the IoU slider in both tabs; only
  conf applies. A model exported with `--anchors` does go through NMS and keeps
  the slider.
- **Stock models** - a bundled model whose file name starts with `yolo`
  (yolo11n, yolo12n, yolo26n-pose) is listed under its own name, not as
  YOLO-Master-...
- **Model type at a glance** - every model picker (Live, Photo, Bench) shows
  the model's type as an icon beside its name: a viewfinder for detection, a
  filled shape in a dashed box for segmentation, a figure with its arms out
  for pose, a hand for whole-body pose. SF Symbols are not drawn to a common
  size (a figure is a narrow glyph, a square fills its box), so the app draws
  the four itself at sizes tuned by eye (`ModelIcon`): the figure fills the
  icon square, the viewfinder is drawn smaller. The
  type is read from the `metadata.json` Core ML's compiler stores in the
  compiled model (the exporter's `task` and `kpt_shape`), so no model is
  loaded for it; a package that has not been compiled yet is judged by its
  file name until its first load.
- **Model gallery** - Settings > Models lists every model in the app, bundled
  and imported. A model opens to its attributes: type, keypoints, pipeline
  (top-down, NMS-free), classes, declared inputs and outputs, precision,
  format and minimum iOS, operation count, size on disk, origin, and what
  the exporter stamped (source checkpoint, keypoint score).
- **Bench tab** - the criterion harness:
  - *cold run*: the selected models x every compute unit, 50-iter median +
    p90 after warmup. The model menu is a list of switches, all on by default
    (a model imported later is on too); it stays open while models are ticked;
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
