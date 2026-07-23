# YOLO-Master Edge — Windows GUI

A native Windows GUI (Dear ImGui + Direct3D 11) for the YOLO-Master edge runtime.
Mirrors the macOS CoreML runner's frontend, but runs on **ONNX Runtime, NCNN, and
MNN** CPU backends. It reuses the exact same C++ runtime as the CLI (`../cpp`), so
preprocessing (letterbox, RGB /255), decode, per-class NMS, and the class palette
are identical.

**Stage 1 (this build):** single-image + folder-batch inference, live conf/IoU
tuning without re-inferring ("forward once, tune cheap"), stretch/letterbox
preprocess, box styles (hud / solid / neon), label modes, per-class counts, timing
stats. CPU only. Detection models.

**Later:** video, webcam, segmentation overlays, CUDA.

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
   re-cached per image; conf/IoU stay live).
3. **Conf / IoU** sliders retune live with no re-inference (candidates are cached
   at a 0.05 floor on the forward pass). **Box style / Labels** are pure redraw.
4. Per-class counts and model/total timing show in the sidebar.

Models live in `../models` (e.g. `esmoe_n_visdrone_sim.onnx`,
`esmoe_n_visdrone.mnn`, the `esmoe_n_visdrone_ncnn/` dir).
