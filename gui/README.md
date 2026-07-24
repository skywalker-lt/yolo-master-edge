# YOLO-Master Edge — Windows GUI

A native Windows GUI (Dear ImGui + Direct3D 11) for the YOLO-Master edge runtime.
Mirrors the macOS CoreML runner's frontend, but runs on **ONNX Runtime, NCNN, and
MNN** CPU backends. It reuses the exact same C++ runtime as the CLI (`../cpp`), so
preprocessing (letterbox, RGB /255), decode, per-class NMS, and the class palette
are identical.

**Now:** image + folder-batch + video + **live webcam**, segmentation mask
overlays, one unified **Open…** (autodetects image vs video), live conf/IoU
tuning without re-inferring ("forward once, tune cheap"), stretch/letterbox
preprocess, box styles (hud / solid / neon), label modes, per-class counts,
timing stats. **Video** pre-infers every frame once (progress bar), then plays the raw clip
from cache at true source fps — inference is never on the playback path.
**Webcam** infers on a background thread with drop-late-frames so the live feed
stays smooth. Both keep conf/IoU/style tuning instant.

**Device: CPU / GPU.** GPU maps to each backend's native accelerator —
ONNX→**CUDA**, ncnn→**Vulkan**, MNN→**OpenCL**. ncnn/MNN GPU works with the
prebuilts you already have (Vulkan/OpenCL are compiled in; needs GPU drivers).
ONNX CUDA additionally needs the **GPU** ONNX Runtime build + CUDA + cuDNN (the
`onnxruntime-win-x64-gpu-*` package, not the CPU one).

---

## What you need (Windows 10/11, x64)

Install these once. Put each wherever you like and point `build.ps1` at them.

| Dependency | Get it | Notes |
|---|---|---|
| **Visual Studio 2022** | community edition, "Desktop development with C++" workload | gives you MSVC + the Windows SDK (D3D11 + comdlg32 already included) |
| **CMake ≥ 3.16** | cmake.org, or the VS installer component | must be on `PATH` |
| **ONNX Runtime** | `onnxruntime-win-x64-1.18.1.zip` from the [ORT releases](https://github.com/microsoft/onnxruntime/releases) | CPU build is enough for stage 1. Unzip → has `include/` + `lib/onnxruntime.{lib,dll}` |
| **NCNN** | `ncnn-YYYYMMDD-windows-vs2022-shared.zip` from [ncnn releases](https://github.com/Tencent/ncnn/releases) | use the **shared** vs2022 build; point at its `x64/` (has `include/ lib/ bin/`) |
| **MNN** | build from source, or a prebuilt Windows package | `cmake -B build -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_TOOLS=OFF`; gives `include/` + `build/Release/MNN.{lib,dll}` |
| **OpenCV** | `opencv-4.x-windows.exe` self-extractor from [opencv.org](https://opencv.org/releases/) | extract → point at `.../opencv/build` (has `OpenCVConfig.cmake` + `x64/vc16/bin/opencv_world4xx.dll`) |

The MSVC runtime DLLs, ORT/NCNN/MNN DLLs, and `opencv_world*.dll` are all
auto-copied next to the built `.exe`, so it runs standalone.

---

## Build

Edit the four SDK paths at the top of `build.ps1` (or pass them as parameters), then:

```powershell
cd gui
./build.ps1            # configure + build Release
./build.ps1 -Run       # build then launch
./build.ps1 -Clean     # wipe build/ and reconfigure
```

Output: `gui/build/Release/yolomaster_gui.exe`.

A backend whose `*_ROOT` isn't set is skipped with a warning — you can build with
just one backend to start (e.g. ONNX only), then add the others.

---

## Using it

1. **Model** — type or *Browse* to a `.onnx` file, an NCNN `.param` (its `.bin`
   must sit beside it), or a `.mnn`. Leave backend on **auto** (inferred from the
   extension) or force one. Click **Load**. *(NCNN as a directory: type the folder
   path — the file dialog can't pick folders.)*
2. **Open image...** — pick a `.jpg/.png`. Inference runs automatically. Or
   **Open folder...** to load every image in a directory: a file-list panel appears
   between the sidebar and preview — click a name, or use the **← / →** (or ↑ / ↓)
   arrow keys to step through. Inference runs on each as you navigate (candidates
   re-cached per image; conf/IoU stay live). Or **Open video...** (`.mp4/.avi/.mov/
   .mkv`) for a transport bar under the preview: **Play/Pause** (or **Space**), a
   **seek** slider, and a frame/fps readout. Playback is paced to the video's fps,
   or runs at inference speed if the CPU can't keep up; **← / →** step frames while
   paused. Each frame runs a fresh forward pass; pause to retune conf/IoU live on
   that frame.
3. **Conf / IoU** sliders retune live with no re-inference (candidates are cached
   at a 0.05 floor on the forward pass). **Box style / Labels** are pure redraw.
4. Per-class counts and model/total timing show in the sidebar.

Models live in `../models` (e.g. `esmoe_n_visdrone_sim.onnx`,
`esmoe_n_visdrone.mnn`, the `esmoe_n_visdrone_ncnn/` dir).

---

## First on-device build — checklist

The GUI has never been built on Windows before, so budget your first ~20–30 min for
the **build**, not the features. Work top to bottom.

### 1. DLLs that must sit next to `yolomaster_gui.exe`

CMake auto-copies all of these into `build/Release/` on a successful build — just
**verify they're actually there** before running:

- [ ] `onnxruntime.dll`
- [ ] `ncnn.dll` (+ its OpenMP dll, e.g. `libomp140.x86_64.dll` / `vcomp140.dll`)
- [ ] `MNN.dll`
- [ ] `opencv_world4xx.dll`
- [ ] `opencv_videoio_ffmpeg4xx_64.dll` ← **needed for video; `Open video...` silently fails to open a file without it**
- [ ] MSVC runtime dlls (`vcruntime140.dll`, `msvcp140.dll`, …)

If any are missing, the exe either won't launch or a feature dies at runtime with no
error. A missing `opencv_world*.dll` = no launch; a missing ffmpeg dll = video won't open.

### 2. Two likely first-build snags

1. **A backend is silently absent.** If a backend's `*_ROOT` path is wrong, CMake
   **warns and skips it** rather than failing. If a backend is missing from the app,
   re-read the CMake *configure* output for a line like `ncnn backend: OFF`.
2. **NCNN as a folder.** The model *file* dialog can't pick a directory — type the
   ncnn folder path straight into the **Model** field (the loader resolves
   `model.ncnn.param` + `.bin`). The folder picker is only for image **sources**.

### 3. Smoke test (do it in this order)

1. **ONNX + single image** — Load an `.onnx`, *Open image...*, confirm boxes draw.
   This one step exercises the entire runtime + D3D11 + overlay path; if it works,
   the rest is just source plumbing.
2. **Backend swap** — Load the `.mnn` and the ncnn dir on the same image; det counts
   should match ONNX (parity is 27 on the sample val image).
3. **Folder** — *Open folder...* on `../visdrone50/images/val`, arrow-key through it.
4. **Video** — *Open video...* on `../results/test.mp4`, Play/Pause, drag the seek bar.
5. **Live tuning** — drag **Conf/IoU** (instant, no re-infer), flip **Box style** and
   **Labels** (pure redraw), toggle **Preprocess** letterbox↔stretch (re-infers).

Paste any MSVC compile errors back and they can be turned around quickly.
