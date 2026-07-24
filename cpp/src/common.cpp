// Shared, backend/model-agnostic ops: class tables, letterbox, decode+NMS,
// drawing, model-metadata parsing, and versatile source resolution.
#include "yolomaster.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <filesystem>

namespace fs = std::filesystem;

namespace yolomaster {

const std::vector<std::string>& visdrone_classes() {
    static const std::vector<std::string> c = {
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor"};
    return c;
}
const std::vector<std::string>& sku110k_classes() {
    static const std::vector<std::string> c = {"object"};
    return c;
}

cv::Mat preprocess(const cv::Mat& img, int imgsz, bool stretch, LetterboxInfo& info) {
    info.orig_w = img.cols;
    info.orig_h = img.rows;
    if (stretch) {
        // resize straight to imgsz x imgsz, ignoring aspect -> per-axis scale, no pad.
        info.scale_x = imgsz / static_cast<float>(img.cols);
        info.scale_y = imgsz / static_cast<float>(img.rows);
        info.scale   = info.scale_x;   // ambiguous under stretch; keep x for any legacy reader
        info.pad_x = info.pad_y = 0;
        cv::Mat out;
        cv::resize(img, out, cv::Size(imgsz, imgsz));
        return out;
    }
    // letterbox: min-scale aspect-preserving, 114-gray padded, centered.
    const float r = std::min(imgsz / static_cast<float>(img.cols),
                             imgsz / static_cast<float>(img.rows));
    const int nw = static_cast<int>(std::round(img.cols * r));
    const int nh = static_cast<int>(std::round(img.rows * r));
    info.scale = info.scale_x = info.scale_y = r;
    info.pad_x = (imgsz - nw) / 2;
    info.pad_y = (imgsz - nh) / 2;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::Mat out(imgsz, imgsz, img.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, nw, nh)));
    return out;
}

cv::Mat letterbox(const cv::Mat& img, int imgsz, LetterboxInfo& info) {
    return preprocess(img, imgsz, /*stretch=*/false, info);
}

// greedy per-box NMS (score-descending, IoU suppression) - replaces
// cv::dnn::NMSBoxes; identical semantics (keep is returned score-descending).
static void nms_greedy(const std::vector<cv::Rect2d>& boxes, const std::vector<float>& scores,
                       float conf, float iou_thr, std::vector<int>& keep) {
    std::vector<int> order;
    order.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
        if (scores[i] >= conf) order.push_back(static_cast<int>(i));
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<char> dead(boxes.size(), 0);
    for (size_t m = 0; m < order.size(); ++m) {
        const int i = order[m];
        if (dead[i]) continue;
        keep.push_back(i);
        for (size_t n = m + 1; n < order.size(); ++n) {
            const int j = order[n];
            if (dead[j]) continue;
            const double xx1 = std::max(boxes[i].x, boxes[j].x);
            const double yy1 = std::max(boxes[i].y, boxes[j].y);
            const double xx2 = std::min(boxes[i].x + boxes[i].width,  boxes[j].x + boxes[j].width);
            const double yy2 = std::min(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);
            const double inter = std::max(0.0, xx2 - xx1) * std::max(0.0, yy2 - yy1);
            const double uni = boxes[i].area() + boxes[j].area() - inter;
            if (uni > 0 && inter / uni > iou_thr) dead[j] = 1;
        }
    }
}

// Decode raw output -> pre-NMS candidates (feat_dim = 4 + nc [+ nm mask coeffs]). Box in orig px.
std::vector<RawDet> decode_candidates(const float* out, int feat_dim, int num_anchors,
                                      const Config& cfg, const LetterboxInfo& lb) {
    const int nc = cfg.num_classes() > 0 ? cfg.num_classes() : (feat_dim - 4);
    const int nm = feat_dim - 4 - nc;                // mask-coeff count (0 = detection)
    std::vector<RawDet> cands;
    auto make = [&](int a, int cls, float score) {
        const float cx = out[0 * num_anchors + a];
        const float cy = out[1 * num_anchors + a];
        const float w  = out[2 * num_anchors + a];
        const float h  = out[3 * num_anchors + a];
        const float x0 = (cx - 0.5f * w - lb.pad_x) / lb.scale_x;
        const float y0 = (cy - 0.5f * h - lb.pad_y) / lb.scale_y;
        RawDet d; d.box = cv::Rect2f(x0, y0, w / lb.scale_x, h / lb.scale_y); d.score = score; d.cls = cls;
        if (nm > 0) { d.mask_coeffs.resize(nm);
            for (int k = 0; k < nm; ++k) d.mask_coeffs[k] = out[(4 + nc + k) * num_anchors + a]; }
        cands.push_back(std::move(d));
    };
    for (int a = 0; a < num_anchors; ++a) {
        int best = -1; float bestv = 0.f; bool any = false;
        for (int c = 0; c < nc; ++c) {
            const float v = out[(4 + c) * num_anchors + a];
            if (v > bestv) { bestv = v; best = c; }
            if (cfg.multi_label && v >= cfg.conf_thresh) any = true;
        }
        if (!(cfg.multi_label ? any : (bestv >= cfg.conf_thresh))) continue;
        if (cfg.multi_label) {                       // one candidate per class >= conf
            for (int c = 0; c < nc; ++c) {
                const float v = out[(4 + c) * num_anchors + a];
                if (v >= cfg.conf_thresh) make(a, c, v);
            }
        } else make(a, best, bestv);                 // single best class
    }
    return cands;
}

// Per-class NMS (ultralytics agnostic=False via class offset) + max_det cap + clip-to-frame.
std::vector<Detection> nms_and_cap(const std::vector<RawDet>& cands, const Config& cfg,
                                   int orig_w, int orig_h) {
    std::vector<cv::Rect2d> boxes; std::vector<float> scores; std::vector<int> idx;
    boxes.reserve(cands.size()); scores.reserve(cands.size()); idx.reserve(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) {
        if (cands[i].score < cfg.conf_thresh) continue;
        boxes.emplace_back(cands[i].box.x, cands[i].box.y, cands[i].box.width, cands[i].box.height);
        scores.push_back(cands[i].score); idx.push_back(static_cast<int>(i));
    }
    std::vector<int> keep;
    {
        const double OFF = 8192.0;                   // > any expected image dimension
        std::vector<cv::Rect2d> off = boxes;
        for (size_t k = 0; k < off.size(); ++k) {
            const int cls = cands[idx[k]].cls;
            off[k].x += cls * OFF; off[k].y += cls * OFF;
        }
        nms_greedy(off, scores, cfg.conf_thresh, cfg.iou_thresh, keep);
    }
    std::vector<Detection> dets;
    const cv::Rect2d frame(0, 0, orig_w, orig_h);
    for (int k : keep) {                             // keep is score-descending
        if (static_cast<int>(dets.size()) >= cfg.max_det) break;
        const RawDet& c = cands[idx[k]];
        cv::Rect2d b = cv::Rect2d(c.box.x, c.box.y, c.box.width, c.box.height) & frame;  // clip in float
        if (b.width > 0 && b.height > 0) {
            Detection d; d.class_id = c.cls; d.conf = c.score;
            d.box = cv::Rect2f(static_cast<float>(b.x), static_cast<float>(b.y),
                               static_cast<float>(b.width), static_cast<float>(b.height));
            d.mask_coeffs = c.mask_coeffs;
            dets.push_back(std::move(d));
        }
    }
    return dets;
}

std::vector<Detection> decode(const float* out, int feat_dim, int num_anchors,
                              const Config& cfg, const LetterboxInfo& lb) {
    auto cands = decode_candidates(out, feat_dim, num_anchors, cfg, lb);
    return nms_and_cap(cands, cfg, lb.orig_w, lb.orig_h);
}

// 10-color class palette (RGB 0..1), indexed cls%10 — identical to the GUI and the Mac runner.
static const float kPalette[10][3] = {
    {0.98f,0.26f,0.30f},{0.20f,0.71f,0.98f},{0.16f,0.85f,0.52f},{0.99f,0.79f,0.12f},
    {0.72f,0.40f,0.98f},{0.99f,0.55f,0.18f},{0.10f,0.83f,0.80f},{0.98f,0.36f,0.66f},
    {0.55f,0.82f,0.28f},{0.40f,0.52f,0.98f},
};
const float* class_color(int class_id) { return kPalette[((class_id % 10) + 10) % 10]; }

static inline float smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

cv::Mat seg_overlay(const std::vector<Detection>& dets, const std::vector<float>& proto,
                    int pc, int ph, int pw, const LetterboxInfo& lb, int imgsz,
                    int orig_w, int orig_h, int mask_alpha) {
    cv::Mat rgba(orig_h, orig_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    if (proto.empty() || pc <= 0 || imgsz <= 0) return rgba;
    const size_t plane = static_cast<size_t>(ph) * pw;
    const float sx = lb.scale_x * pw / imgsz, sy = lb.scale_y * ph / imgsz;  // orig px -> mask space
    const float ox0 = lb.pad_x * (float)pw / imgsz, oy0 = lb.pad_y * (float)ph / imgsz;
    std::vector<float> ml(plane);
    for (const auto& d : dets) {
        if (static_cast<int>(d.mask_coeffs.size()) != pc) continue;   // detection-only box: skip
        const float* co = d.mask_coeffs.data();
        // lowres mask = sigmoid(coeffs . protos), computed once per detection
        for (size_t i = 0; i < plane; ++i) {
            float s = 0.f;
            for (int c = 0; c < pc; ++c) s += co[c] * proto[c * plane + i];
            ml[i] = 1.f / (1.f + std::exp(-s));
        }
        const float* col = class_color(d.class_id);
        const int x0 = std::max(0, (int)std::floor(d.box.x));
        const int y0 = std::max(0, (int)std::floor(d.box.y));
        const int x1 = std::min(orig_w, (int)std::ceil(d.box.x + d.box.width));
        const int y1 = std::min(orig_h, (int)std::ceil(d.box.y + d.box.height));
        for (int oy = y0; oy < y1; ++oy) {
            const float my = oy * sy + oy0;
            const int myi = std::clamp((int)my, 0, ph - 1);
            const int myi2 = std::min(myi + 1, ph - 1);
            const float fy = std::clamp(my - myi, 0.f, 1.f);
            uint8_t* dst = rgba.ptr<uint8_t>(oy);
            for (int ox = x0; ox < x1; ++ox) {
                const float mx = ox * sx + ox0;
                const int mxi = std::clamp((int)mx, 0, pw - 1);
                const int mxi2 = std::min(mxi + 1, pw - 1);
                const float fx = std::clamp(mx - mxi, 0.f, 1.f);
                // bilinear sample of the lowres mask
                const float v = ml[myi * pw + mxi]   * (1 - fx) * (1 - fy)
                              + ml[myi * pw + mxi2]  * fx       * (1 - fy)
                              + ml[myi2 * pw + mxi]  * (1 - fx) * fy
                              + ml[myi2 * pw + mxi2] * fx       * fy;
                const float edge = smoothstep(0.5f - 0.14f, 0.5f + 0.14f, v);  // soft anti-aliased border
                if (edge <= 0.f) continue;
                const uint8_t a = (uint8_t)(edge * mask_alpha);
                uint8_t* px = dst + ox * 4;
                if (a > px[3]) {   // overlapping masks: keep the more opaque one
                    px[0] = (uint8_t)(col[0] * 255); px[1] = (uint8_t)(col[1] * 255);
                    px[2] = (uint8_t)(col[2] * 255); px[3] = a;
                }
            }
        }
    }
    return rgba;
}

void draw(cv::Mat& img, const std::vector<Detection>& dets, const Config& cfg) {
    for (const auto& d : dets) {
        const cv::Rect r(cvRound(d.box.x), cvRound(d.box.y), cvRound(d.box.width), cvRound(d.box.height));
        const float* pc = class_color(d.class_id);
        const cv::Scalar color(pc[2] * 255, pc[1] * 255, pc[0] * 255);   // RGB palette -> BGR
        cv::rectangle(img, r, color, 2);
        const std::string name = (d.class_id < cfg.num_classes()) ? cfg.class_names[d.class_id]
                                                                  : std::to_string(d.class_id);
        char buf[80];
        std::snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), d.conf);
        int base = 0;
        cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(img, cv::Rect(r.x, std::max(0, r.y - ts.height - 4),
                                    ts.width + 2, ts.height + 4), color, cv::FILLED);
        cv::putText(img, buf, cv::Point(r.x, std::max(ts.height, r.y - 3)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ---------------- metadata ----------------
namespace meta {

std::vector<std::string> parse_names_dict(const std::string& s) {
    // keys are unquoted ints, values are quoted strings -> extract quoted tokens in order
    std::vector<std::string> names;
    for (size_t i = 0; i < s.size();) {
        const char q = s[i];
        if (q == '\'' || q == '"') {
            size_t j = i + 1; std::string tok;
            while (j < s.size() && s[j] != q) tok += s[j++];
            names.push_back(tok);
            i = j + 1;
        } else ++i;
    }
    return names;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

bool read_ncnn_yaml(const std::string& path, std::vector<std::string>& names, int& imgsz) {
    std::ifstream f(path);
    if (!f) return false;
    std::map<int, std::string> nm;
    imgsz = 0;
    std::string line;
    enum { NONE, NAMES, IMGSZ } sec = NONE;
    while (std::getline(f, line)) {
        const bool indented = !line.empty() && (line[0] == ' ' || line[0] == '\t' || line[0] == '-');
        if (!indented) {                                   // top-level key -> switch/close section
            if (line.rfind("names:", 0) == 0) { sec = NAMES; continue; }
            if (line.rfind("imgsz:", 0) == 0) {
                sec = IMGSZ;
                auto p = line.find('[');                   // inline "imgsz: [640, 640]"
                if (p != std::string::npos) imgsz = std::atoi(line.c_str() + p + 1);
                continue;
            }
            sec = NONE; continue;
        }
        if (sec == NAMES) {                                // "  0: pedestrian"
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                const int idx = std::atoi(trim(line.substr(0, colon)).c_str());
                nm[idx] = trim(line.substr(colon + 1));
            }
        } else if (sec == IMGSZ && imgsz == 0) {           // "- 640"
            auto d = line.find_first_of("0123456789");
            if (d != std::string::npos) imgsz = std::atoi(line.c_str() + d);
        }
    }
    names.clear();
    for (auto& kv : nm) names.push_back(kv.second);
    return !names.empty();
}

} // namespace meta

// ---------------- source ----------------
static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}
static const std::set<std::string> kImageExt = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
static const std::set<std::string> kVideoExt = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

SourceKind classify_source(const std::string& src) {
    std::error_code ec;
    if (fs::is_directory(src, ec)) return SourceKind::Dir;
    const std::string ext = lower(fs::path(src).extension().string());
    if (ext == ".yaml" || ext == ".yml") return SourceKind::Dataset;
    if (kVideoExt.count(ext)) return SourceKind::Video;
    if (kImageExt.count(ext)) return SourceKind::Image;
    return SourceKind::Unknown;
}

static void collect_dir(const std::string& dir, std::vector<std::string>& out) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        if (kImageExt.count(lower(e.path().extension().string())))
            out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
}

// best-effort dataset.yaml -> val image dir
static std::vector<std::string> resolve_dataset(const std::string& yaml) {
    std::ifstream f(yaml);
    std::string path, val, line;
    while (std::getline(f, line)) {
        auto kv = [&](const char* k, std::string& dst) {
            if (line.rfind(k, 0) == 0) {
                std::string v = line.substr(std::strlen(k));
                auto h = v.find('#'); if (h != std::string::npos) v = v.substr(0, h);
                size_t a = v.find_first_not_of(" \t"); size_t b = v.find_last_not_of(" \t\r");
                dst = (a == std::string::npos) ? "" : v.substr(a, b - a + 1);
            }
        };
        kv("path:", path);
        kv("val:", val);
    }
    if (val.empty()) return {};
    const fs::path ydir = fs::path(yaml).parent_path();
    std::vector<fs::path> cands = {
        fs::path(val),                       // absolute val
        fs::path(path) / val,                // path/val (path absolute)
        ydir / path / val,                   // yaml_dir/path/val
        ydir / val,                          // yaml_dir/val
        fs::path("/data/datasets") / path / val,
    };
    std::error_code ec;
    for (auto& c : cands) {
        if (fs::is_directory(c, ec)) { std::vector<std::string> v; collect_dir(c.string(), v); if (!v.empty()) return v; }
        if (fs::is_regular_file(c, ec) && lower(c.extension().string()) == ".txt") {
            std::vector<std::string> v; std::ifstream tf(c); std::string l;
            while (std::getline(tf, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); if (!l.empty()) v.push_back(l); }
            if (!v.empty()) return v;
        }
    }
    return {};
}

std::vector<std::string> gather_images(const std::string& src, int limit) {
    std::vector<std::string> out;
    switch (classify_source(src)) {
        case SourceKind::Image:   out = {src}; break;
        case SourceKind::Dir:     collect_dir(src, out); break;
        case SourceKind::Dataset: out = resolve_dataset(src); break;
        default: break;
    }
    if (limit > 0 && static_cast<int>(out.size()) > limit) out.resize(limit);
    return out;
}

} // namespace yolomaster
