#include "app.hpp"
#include "imgui.h"
#include "backend_factory.hpp"
#include <algorithm>
#include <cstdio>
#include <cmath>

using namespace yolomaster;

namespace gui {

// 10-color palette (RGB 0-1), indexed cls%10 — identical to the Mac runner.
static const float kPalette[10][3] = {
    {0.98f,0.26f,0.30f},{0.20f,0.71f,0.98f},{0.16f,0.85f,0.52f},{0.99f,0.79f,0.12f},
    {0.72f,0.40f,0.98f},{0.99f,0.55f,0.18f},{0.10f,0.83f,0.80f},{0.98f,0.36f,0.66f},
    {0.55f,0.82f,0.28f},{0.40f,0.52f,0.98f},
};
static ImU32 col_of(int cls, float a = 1.f) {
    const float* c = kPalette[((cls % 10) + 10) % 10];
    return ImGui::GetColorU32(ImVec4(c[0], c[1], c[2], a));
}
// black or white text depending on the box color's luminance (readable on any hue).
static ImU32 text_on(int cls) {
    const float* c = kPalette[((cls % 10) + 10) % 10];
    const float lum = 0.299f * c[0] + 0.587f * c[1] + 0.114f * c[2];
    return lum > 0.6f ? IM_COL32(0,0,0,255) : IM_COL32(255,255,255,255);
}

void App::load_model() {
    be_.reset();
    dets_.clear();
    be_err_.clear();
    const char* names[] = {"auto", "onnx", "ncnn", "mnn"};
    std::string resolved, err;
    auto be = make_backend(model_path_, names[backend_sel_], threads_, "cpu", resolved, err);
    if (!be) { be_err_ = err; return; }
    be_ = std::move(be);
    be_name_ = resolved;
    be_ep_ = be_->active_ep;

    // resolve config: model metadata > visdrone fallback (mirrors the CLI).
    if (!be_->meta_names.empty()) cfg_.class_names = be_->meta_names;
    else                          cfg_.class_names = visdrone_classes();
    cfg_.imgsz = be_->fixed_imgsz > 0 ? be_->fixed_imgsz
               : (be_->meta_imgsz > 0 ? be_->meta_imgsz : 640);
    cfg_.max_det = 300;
    cfg_.multi_label = false;
    need_reinfer_ = !img_bgr_.empty();
}

void App::load_image(const std::string& path, const Platform& plat) {
    close_video();                       // leave video mode if we were in it
    load_err_.clear();
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) { load_err_ = "cannot read image: " + path; return; }
    img_bgr_ = bgr;
    img_path_ = path;
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    if (!plat.upload(rgba, img_tex_)) { load_err_ = "GPU texture upload failed"; return; }
    need_reinfer_ = (be_ != nullptr);
}

void App::load_folder(const std::string& dir, const Platform& plat) {
    close_video();
    folder_imgs_ = gather_images(dir, 0);   // sorted image paths
    folder_path_ = dir;
    if (folder_imgs_.empty()) { cur_idx_ = -1; load_err_ = "no images in: " + dir; return; }
    select_index(0, plat);
}

void App::select_index(int i, const Platform& plat) {
    if (folder_imgs_.empty()) return;
    cur_idx_ = std::clamp(i, 0, (int)folder_imgs_.size() - 1);
    scroll_to_cur_ = true;
    load_image(folder_imgs_[cur_idx_], plat);
}

void App::close_video() {
    if (cap_.isOpened()) cap_.release();
    is_video_ = false; playing_ = false;
    frame_idx_ = 0; total_frames_ = 0; play_accum_ = 0.0;
}

void App::show_frame(const cv::Mat& frame, const Platform& plat) {
    img_bgr_ = frame.clone();            // clone: cap_ reuses its internal buffer on the next read
    cv::Mat rgba;
    cv::cvtColor(img_bgr_, rgba, cv::COLOR_BGR2RGBA);
    if (!plat.upload(rgba, img_tex_)) { load_err_ = "GPU texture upload failed"; return; }
    need_reinfer_ = (be_ != nullptr);    // reuse the still-image inference path
}

void App::open_video(const std::string& path, const Platform& plat) {
    load_err_.clear();
    folder_imgs_.clear(); cur_idx_ = -1;         // leave folder mode
    if (cap_.isOpened()) cap_.release();
    if (!cap_.open(path)) { load_err_ = "cannot open video: " + path; is_video_ = false; return; }
    is_video_ = true; playing_ = false; play_accum_ = 0.0;
    video_path_ = path;
    total_frames_ = (int)cap_.get(cv::CAP_PROP_FRAME_COUNT);
    const double fps = cap_.get(cv::CAP_PROP_FPS);
    video_fps_ = (fps > 1.0 && fps < 1000.0) ? fps : 30.0;
    frame_idx_ = 0;
    cv::Mat f;
    if (cap_.read(f) && !f.empty()) show_frame(f, plat);
    else { load_err_ = "video has no readable frames"; close_video(); }
}

void App::seek_video(int idx, const Platform& plat) {
    if (!cap_.isOpened()) return;
    idx = std::clamp(idx, 0, std::max(0, total_frames_ - 1));
    cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx);
    cv::Mat f;
    if (cap_.read(f) && !f.empty()) { frame_idx_ = idx; show_frame(f, plat); }
}

bool App::advance_video(const Platform& plat) {
    if (!cap_.isOpened()) return false;
    cv::Mat f;
    if (!cap_.read(f) || f.empty()) return false;   // end of stream
    frame_idx_++;
    show_frame(f, plat);
    return true;
}

void App::run_inference() {
    if (!be_ || img_bgr_.empty()) return;
    // forward once with a low conf floor so the cached candidates cover the whole
    // slider range; nms_and_cap() then applies the real (display) conf cheaply.
    Config c = cfg_;
    c.conf_thresh = std::min(conf_, kConfFloor);
    c.iou_thresh  = iou_;
    c.stretch     = (prep_ == Preprocess::Stretch);
    cfg_.stretch  = c.stretch;   // keep display cfg in sync (recompute_nms uses cfg_)
    try { dets_ = be_->infer(img_bgr_, c); }
    catch (const std::exception& e) { be_err_ = std::string("inference error: ") + e.what(); return; }
    pre_ms_ = be_->pre_ms; inf_ms_ = be_->infer_ms; post_ms_ = be_->post_ms;
    need_reinfer_ = false;
    need_renms_ = true;   // apply the real conf/iou below
}

void App::recompute_nms() {
    if (!be_) return;
    cfg_.conf_thresh = conf_;
    cfg_.iou_thresh  = iou_;
    dets_ = nms_and_cap(be_->candidates, cfg_, be_->cand_orig_w, be_->cand_orig_h);
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    need_renms_ = false;
}

void App::frame(const Platform& plat) {
    if (need_reinfer_) run_inference();
    if (need_renms_)   recompute_nms();

    // video playback: advance frames paced to real time (or as fast as inference allows).
    if (is_video_ && playing_) {
        play_accum_ += ImGui::GetIO().DeltaTime;
        if (play_accum_ >= 1.0 / video_fps_) {
            play_accum_ = 0.0;                      // no backlog: run at inference speed if slower
            if (!advance_video(plat)) playing_ = false;   // stop at end of stream
        }
    }

    // keyboard shortcuts (ignored while typing in a field).
    if (!ImGui::GetIO().WantTextInput) {
        const bool fwd  = ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(ImGuiKey_DownArrow);
        const bool back = ImGui::IsKeyPressed(ImGuiKey_LeftArrow)  || ImGui::IsKeyPressed(ImGuiKey_UpArrow);
        if (!folder_imgs_.empty()) {
            if (fwd)  select_index(cur_idx_ + 1, plat);
            else if (back) select_index(cur_idx_ - 1, plat);
        } else if (is_video_) {
            if (ImGui::IsKeyPressed(ImGuiKey_Space)) playing_ = !playing_;
            if (!playing_ && fwd)  seek_video(frame_idx_ + 1, plat);   // step frames while paused
            if (!playing_ && back) seek_video(frame_idx_ - 1, plat);
        }
    }

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("##root", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    ImGui::BeginChild("sidebar", ImVec2(300, 0), true);
    draw_sidebar(plat);
    ImGui::EndChild();

    if (!folder_imgs_.empty()) {
        ImGui::SameLine();
        ImGui::BeginChild("filelist", ImVec2(230, 0), true);
        draw_filelist(plat);
        ImGui::EndChild();
    }

    ImGui::SameLine();
    ImGui::BeginChild("preview", ImVec2(0, 0), true);
    draw_preview(plat);
    ImGui::EndChild();

    ImGui::End();
}

void App::draw_sidebar(const Platform& plat) {
    ImGui::TextUnformatted("YOLO-Master Edge");
    ImGui::Separator();

    // ---- Model ----
    ImGui::SeparatorText("Model");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##model", ".onnx / ncnn dir / .mnn", model_path_, sizeof(model_path_));
    if (ImGui::Button("Browse##model")) {
        std::string p = plat.open_file("Select model", "Models\0*.onnx;*.mnn;*.param\0All\0*.*\0");
        if (!p.empty()) { std::snprintf(model_path_, sizeof(model_path_), "%s", p.c_str()); }
    }
    ImGui::SameLine();
    const char* backends[] = {"auto", "onnx", "ncnn", "mnn"};
    ImGui::SetNextItemWidth(90);
    ImGui::Combo("##backend", &backend_sel_, backends, IM_ARRAYSIZE(backends));
    ImGui::SameLine();
    if (ImGui::Button("Load##model")) load_model();
    if (be_) {
        ImGui::TextColored(ImVec4(0.5f,0.85f,0.4f,1), "%s | %s | nc=%d | %dpx",
                           be_name_.c_str(), be_ep_.c_str(), cfg_.num_classes(), cfg_.imgsz);
    } else if (!be_err_.empty()) {
        ImGui::TextColored(ImVec4(0.98f,0.4f,0.4f,1), "%s", be_err_.c_str());
    }

    // ---- Source ----
    ImGui::SeparatorText("Source");
    if (ImGui::Button("Open image...")) {
        std::string p = plat.open_file("Select image", "Images\0*.jpg;*.jpeg;*.png;*.bmp\0All\0*.*\0");
        if (!p.empty()) {
            folder_imgs_.clear(); cur_idx_ = -1; folder_path_.clear();   // leave folder mode
            load_image(p, plat);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Open folder...")) {
        std::string d = plat.open_folder("Select image folder");
        if (!d.empty()) load_folder(d, plat);
    }
    if (ImGui::Button("Open video...")) {
        std::string p = plat.open_file("Select video", "Videos\0*.mp4;*.avi;*.mov;*.mkv\0All\0*.*\0");
        if (!p.empty()) open_video(p, plat);
    }
    if (!img_bgr_.empty())
        ImGui::Text("%dx%d", img_bgr_.cols, img_bgr_.rows);
    if (!load_err_.empty())
        ImGui::TextColored(ImVec4(0.98f,0.4f,0.4f,1), "%s", load_err_.c_str());

    // ---- Tuning (cheap: conf/iou only re-run NMS) ----
    ImGui::SeparatorText("Detection");
    if (ImGui::SliderFloat("Conf", &conf_, 0.05f, 0.95f, "%.2f")) need_renms_ = true;
    if (ImGui::SliderFloat("IoU",  &iou_,  0.10f, 0.90f, "%.2f")) need_renms_ = true;
    ImGui::SetNextItemWidth(120);
    if (ImGui::SliderInt("Threads", &threads_, 1, 16)) { /* applied on next Load */ }

    // ---- Appearance (free: pure redraw) ----
    ImGui::SeparatorText("Appearance");
    int st = (int)style_;
    const char* styles[] = {"hud", "solid", "neon"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("Box style", &st, styles, IM_ARRAYSIZE(styles))) style_ = (BoxStyle)st;
    int lm = (int)labels_;
    const char* lms[] = {"full", "min", "off"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("Labels", &lm, lms, IM_ARRAYSIZE(lms))) labels_ = (LabelMode)lm;
    int pp = (int)prep_;
    const char* pps[] = {"letterbox", "stretch"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("Preprocess", &pp, pps, IM_ARRAYSIZE(pps))) {
        if ((Preprocess)pp != prep_) { prep_ = (Preprocess)pp; need_reinfer_ = true; }
    }

    // ---- Stats ----
    ImGui::SeparatorText("Inference");
    if (be_ && !img_bgr_.empty()) {
        const double total = pre_ms_ + inf_ms_ + post_ms_;
        ImGui::Text("model  %.1f ms", inf_ms_);
        ImGui::Text("total  %.1f ms  (%.0f fps)", total, total > 0 ? 1000.0 / total : 0.0);
        ImGui::Text("dets   %d", (int)dets_.size());
        if (!class_counts_.empty()) {
            ImGui::Separator();
            for (int i = 0; i < (int)class_counts_.size(); ++i) {
                if (!class_counts_[i]) continue;
                const float* c = kPalette[i % 10];
                ImGui::ColorButton("##c", ImVec4(c[0],c[1],c[2],1),
                    ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop, ImVec2(12,12));
                ImGui::SameLine();
                const char* nm = i < cfg_.num_classes() ? cfg_.class_names[i].c_str() : "?";
                ImGui::Text("%-14s %d", nm, class_counts_[i]);
            }
        }
    } else {
        ImGui::TextDisabled("load a model and an image");
    }
}

void App::draw_filelist(const Platform& plat) {
    ImGui::Text("%d/%d", cur_idx_ + 1, (int)folder_imgs_.size());
    ImGui::SameLine();
    ImGui::TextDisabled("(<- ->)");
    ImGui::Separator();
    ImGui::BeginChild("files", ImVec2(0, 0), false);
    for (int i = 0; i < (int)folder_imgs_.size(); ++i) {
        const std::string name = folder_imgs_[i].substr(folder_imgs_[i].find_last_of("/\\") + 1);
        const bool sel = (i == cur_idx_);
        if (ImGui::Selectable(name.c_str(), sel) && !sel) select_index(i, plat);
        if (sel && scroll_to_cur_) { ImGui::SetScrollHereY(0.5f); scroll_to_cur_ = false; }
    }
    ImGui::EndChild();
}

// draw one detection box in the chosen style onto `dl`, mapped orig-px -> screen.
static void draw_box(ImDrawList* dl, const Detection& d, BoxStyle style, LabelMode labels,
                     const Config& cfg, ImVec2 origin, float scale) {
    const float x0 = origin.x + d.box.x * scale;
    const float y0 = origin.y + d.box.y * scale;
    const float x1 = x0 + d.box.width  * scale;
    const float y1 = y0 + d.box.height * scale;
    const ImU32 col = col_of(d.class_id);

    if (style == BoxStyle::Solid) {
        dl->AddRect(ImVec2(x0,y0), ImVec2(x1,y1), col, 2.f, 0, 2.f);
    } else if (style == BoxStyle::Neon) {
        dl->AddRect(ImVec2(x0-1,y0-1), ImVec2(x1+1,y1+1), col_of(d.class_id, 0.35f), 3.f, 0, 5.f);
        dl->AddRect(ImVec2(x0,y0), ImVec2(x1,y1), col, 3.f, 0, 2.f);
    } else { // Hud: corner brackets
        const float len = std::min({ (x1-x0), (y1-y0), 22.f }) * 0.35f + 4.f;
        const float t = 2.5f;
        // TL
        dl->AddLine(ImVec2(x0,y0), ImVec2(x0+len,y0), col, t);
        dl->AddLine(ImVec2(x0,y0), ImVec2(x0,y0+len), col, t);
        // TR
        dl->AddLine(ImVec2(x1,y0), ImVec2(x1-len,y0), col, t);
        dl->AddLine(ImVec2(x1,y0), ImVec2(x1,y0+len), col, t);
        // BL
        dl->AddLine(ImVec2(x0,y1), ImVec2(x0+len,y1), col, t);
        dl->AddLine(ImVec2(x0,y1), ImVec2(x0,y1-len), col, t);
        // BR
        dl->AddLine(ImVec2(x1,y1), ImVec2(x1-len,y1), col, t);
        dl->AddLine(ImVec2(x1,y1), ImVec2(x1,y1-len), col, t);
    }

    if (labels == LabelMode::Off) return;
    char buf[96];
    const char* nm = d.class_id < cfg.num_classes() ? cfg.class_names[d.class_id].c_str() : "?";
    if (labels == LabelMode::Full) std::snprintf(buf, sizeof(buf), "%s %.2f", nm, d.conf);
    else                           std::snprintf(buf, sizeof(buf), "%s", nm);
    const ImVec2 ts = ImGui::CalcTextSize(buf);
    const float pad = 3.f;
    ImVec2 lp(x0, y0 - ts.y - 2 * pad);
    if (lp.y < origin.y) lp.y = y0;                        // flip inside if it would clip off-top
    dl->AddRectFilled(lp, ImVec2(lp.x + ts.x + 2*pad, lp.y + ts.y + 2*pad), col, 2.f);
    dl->AddText(ImVec2(lp.x + pad, lp.y + pad), text_on(d.class_id), buf);
}

void App::draw_transport(const Platform& plat) {
    if (ImGui::Button(playing_ ? "Pause" : "Play ")) playing_ = !playing_;
    ImGui::SameLine();
    int f = frame_idx_;
    ImGui::SetNextItemWidth(-150);
    if (ImGui::SliderInt("##seek", &f, 0, std::max(0, total_frames_ - 1), "frame %d")) {
        playing_ = false; seek_video(f, plat);
    }
    ImGui::SameLine();
    ImGui::Text("%d/%d  %.0ffps", frame_idx_ + 1, total_frames_, video_fps_);
}

void App::draw_preview(const Platform& plat) {
    if (!img_tex_.id) {
        ImGui::TextDisabled(is_video_ ? "No frame." : "Open an image, folder, or video to begin.");
        return;
    }
    const ImVec2 cur = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float ctrlH = is_video_ ? ImGui::GetFrameHeightWithSpacing() : 0.f;
    const float imgH = std::max(1.f, avail.y - ctrlH);
    const float scale = std::min(avail.x / img_tex_.w, imgH / img_tex_.h);
    const ImVec2 disp(img_tex_.w * scale, img_tex_.h * scale);
    const ImVec2 origin(cur.x + (avail.x - disp.x) * 0.5f, cur.y + (imgH - disp.y) * 0.5f);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)img_tex_.id, origin, ImVec2(origin.x + disp.x, origin.y + disp.y));
    for (const auto& d : dets_)
        draw_box(dl, d, style_, labels_, cfg_, origin, scale);

    if (is_video_) {
        ImGui::SetCursorScreenPos(ImVec2(cur.x, cur.y + imgH));
        draw_transport(plat);
    }
}

} // namespace gui
