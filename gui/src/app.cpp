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

App::~App() { close_camera(); close_video(); }   // join the webcam worker + video pre-infer thread

std::string App::gpu_device_str() const {
    return device_ == Device::GPU ? "gpu" : "cpu";   // make_backend maps "gpu" per backend
}

void App::load_model(const Platform& plat) {
    const bool cam = is_cam_;
    const bool vid = is_video_;
    const std::string vpath = video_path_;
    stop_worker();                                  // stop any camera worker holding the old backend
    if (vid) { vinfer_cancel_ = true; if (vinfer_.joinable()) vinfer_.join(); vinfer_cancel_ = false; }
    be_.reset();
    dets_.clear();
    be_err_.clear();
    const char* names[] = {"auto", "onnx", "ncnn", "mnn"};
    std::string resolved, err;
    auto be = make_backend(model_path_, names[backend_sel_], threads_, gpu_device_str(), resolved, err);
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
    if (cam)                     { start_worker(); submit_job(img_bgr_); }   // resume live feed
    else if (vid && !vpath.empty()) open_video(vpath, plat);                 // re-infer the clip
    else                         need_reinfer_ = !img_bgr_.empty();
}

void App::open_media(const std::string& path, const Platform& plat) {
    switch (classify_source(path)) {     // autodetect image / video / folder by extension
        case SourceKind::Video: open_video(path, plat); break;
        case SourceKind::Dir:   load_folder(path, plat); break;
        case SourceKind::Image:
            folder_imgs_.clear(); cur_idx_ = -1; folder_path_.clear();
            load_image(path, plat); break;
        default: load_err_ = "unsupported file type: " + path;
    }
}

void App::load_image(const std::string& path, const Platform& plat) {
    close_camera();
    close_video();                       // leave video/camera mode if we were in it
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
    close_camera();
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
    vinfer_cancel_ = true;                // stop any in-flight pre-inference
    if (vinfer_.joinable()) vinfer_.join();
    vinfer_cancel_ = false;
    if (cap_.isOpened()) cap_.release();
    is_video_ = false; playing_ = false; video_ready_ = false;
    frame_idx_ = 0; total_frames_ = 0; play_accum_ = 0.0;
    vcache_.clear(); vproto_.clear(); vprogress_ = 0; vinfer_done_ = false;
}

void App::show_frame(const cv::Mat& frame, const Platform& plat) {
    img_bgr_ = frame.clone();            // clone: cap_ reuses its internal buffer on the next read
    cv::Mat rgba;
    cv::cvtColor(img_bgr_, rgba, cv::COLOR_BGR2RGBA);
    plat.upload(rgba, img_tex_);
}

// Background: forward-pass EVERY frame once, caching per-frame candidates (+ proto for seg).
// Runs on its own thread + its own capture; the UI shows a progress bar meanwhile.
void App::video_preinfer(std::string path, Config c) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { vinfer_done_ = true; return; }
    double sum = 0; int cnt = 0;
    cv::Mat f;
    while (!vinfer_cancel_ && cap.read(f) && !f.empty()) {
        try { be_->infer(f, c); } catch (...) { break; }
        vcache_.push_back(be_->candidates);
        if (be_->is_seg()) {
            if (vproto_.size() < vcache_.size() - 1) vproto_.resize(vcache_.size() - 1);
            vproto_.push_back(be_->proto);
            vproto_c_ = be_->proto_c; vproto_h_ = be_->proto_h; vproto_w_ = be_->proto_w;
        }
        vlb_ = be_->cand_lb; vorig_w_ = be_->cand_orig_w; vorig_h_ = be_->cand_orig_h;
        sum += be_->infer_ms; cnt++;
        vprogress_ = (int)vcache_.size();
    }
    total_frames_ = (int)vcache_.size();
    vinfer_ms_ = cnt ? sum / cnt : 0.0;
    vinfer_done_ = true;
}

void App::open_video(const std::string& path, const Platform& plat) {
    close_camera();
    close_video();
    load_err_.clear();
    folder_imgs_.clear(); cur_idx_ = -1;         // leave folder mode
    if (!be_) { load_err_ = "load a model first"; return; }
    if (!cap_.open(path)) { load_err_ = "cannot open video: " + path; return; }
    is_video_ = true; playing_ = false; play_accum_ = 0.0; frame_idx_ = 0;
    video_ready_ = false; vprogress_ = 0; vinfer_done_ = false;
    video_path_ = path;
    total_frames_ = (int)cap_.get(cv::CAP_PROP_FRAME_COUNT);
    const double fps = cap_.get(cv::CAP_PROP_FPS);
    video_fps_ = (fps > 1.0 && fps < 1000.0) ? fps : 30.0;
    cv::Mat f0;                                  // show frame 0 immediately (no boxes yet)
    if (cap_.read(f0) && !f0.empty()) show_frame(f0, plat);
    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
    Config c = cfg_;                             // cache to the conf floor; per-frame NMS is cheap
    c.conf_thresh = kConfFloor; c.iou_thresh = iou_; c.stretch = (prep_ == Preprocess::Stretch);
    vinfer_ = std::thread(&App::video_preinfer, this, path, c);
}

// re-NMS + mask for a cached frame's candidates (no decode) — used on conf/IoU change
void App::overlay_from_cache(int idx, const Platform& plat) {
    if (idx < 0 || idx >= (int)vcache_.size()) return;
    seg_model_ = !vproto_.empty();
    Config c = cfg_; c.conf_thresh = conf_; c.iou_thresh = iou_;
    dets_ = nms_and_cap(vcache_[idx], c, vorig_w_, vorig_h_);
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    if (seg_model_ && idx < (int)vproto_.size())
        build_overlay(dets_, vproto_[idx], vproto_c_, vproto_h_, vproto_w_, vlb_, vorig_w_, vorig_h_, plat);
    else has_mask_ = false;
}

// decode the pixels for frame idx (sequential read when possible, else seek), display + overlay
void App::render_video_frame(int idx, const Platform& plat) {
    if (!cap_.isOpened() || total_frames_ <= 0) return;
    idx = std::clamp(idx, 0, total_frames_ - 1);
    if (idx != frame_idx_ + 1) cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx);  // seek if non-sequential
    cv::Mat f;
    if (!cap_.read(f) || f.empty()) { cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx);
                                      if (!cap_.read(f) || f.empty()) return; }
    frame_idx_ = idx;
    show_frame(f, plat);
    overlay_from_cache(idx, plat);
    inf_ms_ = vinfer_ms_;
}

void App::seek_video(int idx, const Platform& plat) {
    if (video_ready_) render_video_frame(idx, plat);
}

// ---------------- webcam ----------------
void App::open_camera(const Platform& plat) {
    load_err_.clear();
    close_video();
    folder_imgs_.clear(); cur_idx_ = -1;
    if (!be_) { load_err_ = "load a model first"; return; }
    if (!cam_.open(0)) { load_err_ = "cannot open webcam (device 0)"; return; }
    cam_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);      // 720p is plenty; we downscale to imgsz anyway
    cam_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    is_cam_ = true; cam_fps_ema_ = cam_ms_ema_ = 0.0;
    start_worker();
}

void App::close_camera() {
    stop_worker();
    if (cam_.isOpened()) cam_.release();
    is_cam_ = false;
}

// ---------------- async inference worker (video + webcam) ----------------
void App::start_worker() {
    if (worker_.joinable()) return;
    worker_quit_ = false; job_pending_ = false; worker_busy_ = false;
    async_mode_ = true;
    worker_ = std::thread(&App::worker_loop, this);
}

void App::stop_worker() {
    async_mode_ = false;
    if (!worker_.joinable()) return;
    { std::lock_guard<std::mutex> lk(job_mtx_); worker_quit_ = true; }
    job_cv_.notify_all();
    worker_.join();
    { std::lock_guard<std::mutex> lk(res_mtx_); ares_dirty_ = false; }
    worker_busy_ = false; job_pending_ = false;
}

void App::submit_job(const cv::Mat& frame) {
    if (worker_busy_ || frame.empty()) return;      // drop-late: skip while the worker is inferring
    {
        std::lock_guard<std::mutex> lk(job_mtx_);
        job_frame_ = frame.clone();
        job_cfg_ = cfg_;
        job_cfg_.conf_thresh = conf_;
        job_cfg_.iou_thresh  = iou_;
        job_cfg_.stretch     = (prep_ == Preprocess::Stretch);
        job_pending_ = true;
    }
    job_cv_.notify_one();
}

void App::worker_loop() {
    for (;;) {
        cv::Mat frame; Config c;
        {
            std::unique_lock<std::mutex> lk(job_mtx_);
            job_cv_.wait(lk, [&] { return job_pending_ || worker_quit_; });
            if (worker_quit_) return;
            frame = job_frame_; c = job_cfg_; job_pending_ = false;
        }
        if (!be_ || frame.empty()) continue;
        worker_busy_ = true;
        Config cc = c; cc.conf_thresh = std::min(c.conf_thresh, kConfFloor);   // cache to the floor
        bool ok = true; double inf = 0;
        try { be_->infer(frame, cc); inf = be_->infer_ms; }
        catch (...) { ok = false; }
        if (ok) {
            std::lock_guard<std::mutex> lk(res_mtx_);
            ares_.cands  = be_->candidates;
            ares_.dets   = nms_and_cap(be_->candidates, c, be_->cand_orig_w, be_->cand_orig_h);
            ares_.proto  = be_->proto;
            ares_.pc = be_->proto_c; ares_.ph = be_->proto_h; ares_.pw = be_->proto_w;
            ares_.lb = be_->cand_lb; ares_.ow = be_->cand_orig_w; ares_.oh = be_->cand_orig_h;
            ares_.inf_ms = inf; ares_.seg = be_->is_seg();
            ares_dirty_ = true;
        }
        worker_busy_ = false;
    }
}

void App::consume_async(const Platform& plat) {
    AsyncResult r;
    {
        std::lock_guard<std::mutex> lk(res_mtx_);
        if (!ares_dirty_) return;
        r = ares_;                 // copy out (proto included) so drawing is lock-free
        ares_dirty_ = false;
    }
    dets_ = std::move(r.dets);
    inf_ms_ = r.inf_ms; pre_ms_ = 0; post_ms_ = 0;
    seg_model_ = r.seg;
    if (is_cam_ && r.inf_ms > 0)
        cam_ms_ema_ = (cam_ms_ema_ <= 0) ? r.inf_ms : cam_ms_ema_ * 0.8 + r.inf_ms * 0.2;
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    if (r.seg) build_overlay(dets_, r.proto, r.pc, r.ph, r.pw, r.lb, r.ow, r.oh, plat);
    else has_mask_ = false;
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
    seg_model_ = be_->is_seg();
    need_reinfer_ = false;
    need_renms_ = true;   // apply the real conf/iou below
}

void App::recompute_nms() {
    if (!be_) return;
    seg_model_ = be_->is_seg();
    cfg_.conf_thresh = conf_;
    cfg_.iou_thresh  = iou_;
    dets_ = nms_and_cap(be_->candidates, cfg_, be_->cand_orig_w, be_->cand_orig_h);
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    need_renms_ = false;
    need_overlay_ = true;   // dets changed -> seg overlay is stale
}

void App::build_overlay(const std::vector<Detection>& dets, const std::vector<float>& proto,
                        int pc, int ph, int pw, const LetterboxInfo& lb,
                        int ow, int oh, const Platform& plat) {
    has_mask_ = false;
    if (proto.empty() || pc <= 0 || dets.empty()) return;
    cv::Mat ov = seg_overlay(dets, proto, pc, ph, pw, lb, cfg_.imgsz, ow, oh);
    if (plat.upload(ov, mask_tex_)) has_mask_ = true;
}

void App::rebuild_overlay(const Platform& plat) {
    need_overlay_ = false;
    if (!be_) { has_mask_ = false; return; }
    build_overlay(dets_, be_->proto, be_->proto_c, be_->proto_h, be_->proto_w,
                  be_->cand_lb, be_->cand_orig_w, be_->cand_orig_h, plat);
}

void App::frame(const Platform& plat) {
    // sync inference for still image / folder (never for video/camera — those own the backend)
    if (!async_mode_ && !is_video_) {
        if (need_reinfer_) run_inference();
        if (need_renms_)   recompute_nms();
        if (need_overlay_) rebuild_overlay(plat);
    }

    // webcam: grab + display every UI frame at feed rate; infer off-thread (drops late frames)
    if (is_cam_) {
        cv::Mat f;
        if (cam_.read(f) && !f.empty()) {
            if (cam_mirror_) cv::flip(f, f, 1);
            const double dt = ImGui::GetIO().DeltaTime;
            if (dt > 0) cam_fps_ema_ = (cam_fps_ema_ <= 0) ? 1.0 / dt : cam_fps_ema_ * 0.9 + (1.0 / dt) * 0.1;
            show_frame(f, plat);
            submit_job(img_bgr_);
        }
        consume_async(plat);
        if (need_renms_) need_renms_ = false;   // camera picks up new conf/iou on the next frame
    }

    // video: pre-infer all frames (progress bar), then play the raw video from cache — NO inference
    // on the playback path, so it runs at true source fps.
    if (is_video_) {
        if (!video_ready_) {
            if (vinfer_done_) {
                if (vinfer_.joinable()) vinfer_.join();
                video_ready_ = true;
                render_video_frame(0, plat);            // first frame with its cached overlay
            }
        } else {
            if (playing_) {
                play_accum_ += ImGui::GetIO().DeltaTime;
                const double iv = 1.0 / video_fps_;
                if (play_accum_ >= iv) {
                    play_accum_ -= iv;                  // carry remainder -> real-time pacing
                    if (frame_idx_ + 1 >= total_frames_) frame_idx_ = -1;   // loop
                    render_video_frame(frame_idx_ + 1, plat);
                }
            }
            if (need_renms_) { overlay_from_cache(frame_idx_, plat); need_renms_ = false; }
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

    const float ui = ImGui::GetFontSize() / 17.0f;   // ~DPI scale (font is 17*dpi on Windows)
    ImGui::BeginChild("sidebar", ImVec2(300 * ui, 0), true);
    draw_sidebar(plat);
    ImGui::EndChild();

    if (!folder_imgs_.empty()) {
        ImGui::SameLine();
        ImGui::BeginChild("filelist", ImVec2(230 * ui, 0), true);
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
    ImGui::SetNextItemWidth(80);
    ImGui::Combo("##backend", &backend_sel_, backends, IM_ARRAYSIZE(backends));
    ImGui::SameLine();
    int dv = (int)device_;
    const char* devs[] = {"CPU", "GPU"};
    ImGui::SetNextItemWidth(64);
    if (ImGui::Combo("##device", &dv, devs, IM_ARRAYSIZE(devs)) && (Device)dv != device_) {
        device_ = (Device)dv;
        if (be_) load_model(plat);   // rebuild on the new device (onnx:CUDA / ncnn:Vulkan / mnn:OpenCL)
    }
    ImGui::SameLine();
    if (ImGui::Button("Load##model")) load_model(plat);
    if (be_) {
        ImGui::TextColored(ImVec4(0.5f,0.85f,0.4f,1), "%s | %s | nc=%d | %dpx",
                           be_name_.c_str(), be_ep_.c_str(), cfg_.num_classes(), cfg_.imgsz);
    } else if (!be_err_.empty()) {
        ImGui::TextColored(ImVec4(0.98f,0.4f,0.4f,1), "%s", be_err_.c_str());
    }

    // ---- Source ----
    ImGui::SeparatorText("Source");
    if (ImGui::Button("Open...")) {   // one picker: image or video, autodetected by extension
        std::string p = plat.open_file("Open image or video",
            "Media\0*.jpg;*.jpeg;*.png;*.bmp;*.mp4;*.avi;*.mov;*.mkv\0All\0*.*\0");
        if (!p.empty()) open_media(p, plat);
    }
    ImGui::SameLine();
    if (ImGui::Button("Open folder...")) {
        std::string d = plat.open_folder("Select image folder");
        if (!d.empty()) load_folder(d, plat);
    }
    ImGui::SameLine();
    if (is_cam_) {
        if (ImGui::Button("Stop Camera")) close_camera();
    } else if (ImGui::Button("Webcam")) {
        open_camera(plat);
    }
    if (is_cam_) ImGui::Checkbox("Mirror", &cam_mirror_);
    if (!img_bgr_.empty())
        ImGui::Text("%dx%d%s", img_bgr_.cols, img_bgr_.rows, is_cam_ ? "  (live)" : "");
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
    if (seg_model_) {                               // segmentation: masks / boxes / both
        int ov = (int)overlay_;
        const char* ovs[] = {"both", "masks", "boxes"};
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("Overlay", &ov, ovs, IM_ARRAYSIZE(ovs));
        overlay_ = (Overlay)ov;
    }
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
        if ((Preprocess)pp != prep_) {
            prep_ = (Preprocess)pp;
            if (is_video_)       open_video(video_path_, plat);   // re-infer the clip
            else if (!is_cam_)   need_reinfer_ = true;            // camera picks it up on the next frame
        }
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
    } else { // Hud: thin full border + thick corner brackets
        dl->AddRect(ImVec2(x0,y0), ImVec2(x1,y1), col_of(d.class_id, 0.55f), 0.f, 0, 1.2f);  // thin sides
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

void App::draw_camera_hud() {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 wp = ImGui::GetWindowPos();
    const ImVec2 p(wp.x + 14, wp.y + 12);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%.0f fps   %.0f ms   %d obj",
                  cam_fps_ema_, cam_ms_ema_, (int)dets_.size());
    const ImVec2 ts = ImGui::CalcTextSize(buf);
    dl->AddRectFilled(ImVec2(p.x - 7, p.y - 5), ImVec2(p.x + ts.x + 9, p.y + ts.y + 6),
                      IM_COL32(0, 0, 0, 140), 6.f);
    dl->AddCircleFilled(ImVec2(p.x + 3, p.y + ts.y * 0.5f), 4.f, IM_COL32(240, 70, 70, 255));  // LIVE dot
    dl->AddText(ImVec2(p.x + 12, p.y), IM_COL32(255, 255, 255, 255), buf);
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

    const bool show_masks = seg_model_ && overlay_ != Overlay::Boxes && has_mask_;
    const bool show_boxes = !seg_model_ || overlay_ != Overlay::Masks;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 br(origin.x + disp.x, origin.y + disp.y);
    dl->AddImage((ImTextureID)img_tex_.id, origin, br);
    if (show_masks)   // seg overlay is same dims as the image -> same rect, alpha-blended by ImGui
        dl->AddImage((ImTextureID)mask_tex_.id, origin, br);
    if (show_boxes)
        for (const auto& d : dets_)
            draw_box(dl, d, style_, labels_, cfg_, origin, scale);

    if (is_cam_) draw_camera_hud();

    if (is_video_) {
        ImGui::SetCursorScreenPos(ImVec2(cur.x, cur.y + imgH));
        if (video_ready_) {
            draw_transport(plat);
        } else {                                    // pre-inference progress
            const int done = vprogress_.load();
            const int tot = total_frames_;
            char ov[64];
            if (tot > 0) std::snprintf(ov, sizeof(ov), "Inferring  %d / %d", done, tot);
            else         std::snprintf(ov, sizeof(ov), "Inferring frame %d", done);
            ImGui::ProgressBar(tot > 0 ? (float)done / tot : -1.0f * (float)ImGui::GetTime(),
                               ImVec2(-1, 0), ov);
        }
    }
}

} // namespace gui
