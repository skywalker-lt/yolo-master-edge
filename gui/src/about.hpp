// About / Acknowledgements / License content for the About window — mirrors the Mac runner's
// Info + License pages, with the third-party list adapted to the Windows runtime (ONNX/ncnn/MNN/OpenCV/ImGui).
#pragma once

namespace gui {

inline const char* kAppName    = "YOLO-Master Edge";
inline const char* kAppVersion = "1.0.0";
inline const char* kAppTagline = "On-device YOLO-Master detection & segmentation via ONNX Runtime / ncnn / MNN.";
inline const char* kAuthor     = "(Thomas) Ruiheng Li";
inline const char* kAuthorAffil= "The Hong Kong University of Science and Technology";
inline const char* kAuthorUrl  = "https://github.com/skywalker-lt";
inline const char* kSourceUrl  = "https://github.com/skywalker-lt/yolo-master-edge";

// `logo` is the base name of a PNG under assets/ack/ (nullptr -> lettered placeholder tile).
struct Ack { const char* name; const char* blurb; const char* repo; const char* logo; };

inline const Ack kAcks[] = {
    {"YOLO-Master @ Tencent",
     "The YOLO-Master detector/segmenter family this runner packages. (C) 2026 Tencent - AGPL-3.0.",
     "https://github.com/Tencent/YOLO-Master", "tencent"},
    {"Ultralytics",
     "The YOLO training & inference framework YOLO-Master builds on. (C) 2025 Ultralytics - AGPL-3.0.",
     "https://github.com/ultralytics/ultralytics", "ultralytics"},
    {"ncnn @ Tencent",
     "High-performance NN inference (ncnn backend + Vulkan GPU). (C) Tencent - BSD-3-Clause.",
     "https://github.com/Tencent/ncnn", "tencent"},
    {"ONNX Runtime @ Microsoft",
     "Cross-platform inference runtime (ONNX backend + CUDA EP). (C) Microsoft - MIT License.",
     "https://github.com/microsoft/onnxruntime", nullptr},
    {"MNN @ Alibaba",
     "Lightweight inference engine (MNN backend + OpenCL/Vulkan GPU). (C) Alibaba - Apache-2.0.",
     "https://github.com/alibaba/MNN", nullptr},
    {"OpenCV",
     "Image decoding, preprocessing, and video/camera capture. - Apache-2.0.",
     "https://github.com/opencv/opencv", nullptr},
    {"Dear ImGui",
     "The immediate-mode GUI toolkit behind this interface. (C) 2014-2026 Omar Cornut - MIT License.",
     "https://github.com/ocornut/imgui", nullptr},
};

inline const char* kLicenseText =
R"LIC(================================================================
YOLO-Master Edge
Copyright (C) 2026 (Thomas) RUIHENG LI
(Author affiliated with The Hong Kong University of Science and
Technology)

YOLO-Master Edge is a cross-platform toolchain for running
YOLO-Master models on-device via ONNX Runtime, ncnn, and MNN
(CPU and GPU: CUDA / Vulkan / OpenCL).

This project is licensed under the GNU Affero General Public
License, version 3 (AGPL-3.0). You should have received a copy of
the AGPL-3.0 with this program; if not, see
<https://www.gnu.org/licenses/>.

This project is built upon and incorporates the following
third-party software:

  * YOLO-Master - Copyright (C) 2026 Tencent. Licensed under the
    AGPL-3.0. YOLO-Master includes modifications by Tencent and is
    itself based on ultralytics.

  * ultralytics - Copyright (c) 2025 Ultralytics and authors.
    Licensed under the AGPL-3.0.

  * ONNX Runtime - Copyright (c) Microsoft Corporation. Licensed
    under the MIT License.

  * ncnn - Copyright (c) Tencent. Licensed under the BSD-3-Clause
    License.

  * MNN - Copyright (c) Alibaba Group. Licensed under the
    Apache-2.0 License.

  * OpenCV - Licensed under the Apache-2.0 License.

  * Dear ImGui - Copyright (c) 2014-2026 Omar Cornut. Licensed
    under the MIT License.

Modifications: This project modifies and extends YOLO-Master with a
C++/ONNX/ncnn/MNN export and runtime, plus native macOS and Windows
frontends. See the repository commit history for details and dates.

ACCEPTABLE USE NOTICE (not a license term):
The author(s) develop and release this software exclusively for
lawful, civilian, and peaceful purposes. Any use in violation of
applicable law - including, without limitation, terrorism, the
development or deployment of weapons, or other military applications
prohibited by applicable law - is already prohibited by law and is
emphatically condemned by the author(s). This notice is a statement
of intent; it does not impose additional restrictions under Section
10 of the AGPL-3.0.

NO WARRANTY: This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Affero General Public License for more details.
================================================================)LIC";

} // namespace gui
