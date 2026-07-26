#include "app.hpp"
#include "imgui.h"
#include "about.hpp"
#include "backend_factory.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace yolomaster;

namespace gui {

// 10-color palette (RGB 0-1), indexed cls%10 - identical to the Mac runner.
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

App::~App() { close_camera(); close_video(); close_folder(); }   // join all background threads

std::string App::gpu_device_str() const {
    return device_ == Device::GPU ? "gpu" : "cpu";   // make_backend maps "gpu" per backend
}

void App::load_model(const Platform& plat) {
    const bool cam = is_cam_;
    const bool vid = is_video_;
    const bool fold = !folder_imgs_.empty();
    const std::string vpath = video_path_, fdir = folder_path_;
    stop_worker();                                  // stop any camera worker holding the old backend
    if (vid)  { vinfer_cancel_ = true; if (vinfer_.joinable()) vinfer_.join(); vinfer_cancel_ = false; }
    if (fold) { finfer_cancel_ = true; if (finfer_.joinable()) finfer_.join(); finfer_cancel_ = false; }
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
    be_note_ = be_->ep_note;

    // resolve config: model metadata > visdrone fallback (mirrors the CLI).
    if (!be_->meta_names.empty()) cfg_.class_names = be_->meta_names;
    else                          cfg_.class_names = visdrone_classes();
    cfg_.imgsz = be_->fixed_imgsz > 0 ? be_->fixed_imgsz
               : (be_->meta_imgsz > 0 ? be_->meta_imgsz : 640);
    cfg_.max_det = 300;
    cfg_.multi_label = false;
    if (cam)                     { start_worker(); submit_job(img_bgr_); }   // resume live feed
    else if (vid && !vpath.empty()) open_video(vpath, plat);                 // re-infer the clip
    else if (fold && !fdir.empty()) load_folder(fdir, plat);                 // re-infer the folder
    else                         need_reinfer_ = !img_bgr_.empty();
}

void App::open_media(const std::string& path, const Platform& plat) {
    switch (classify_source(path)) {     // autodetect image / video / folder by extension
        case SourceKind::Video: open_video(path, plat); break;
        case SourceKind::Dir:   load_folder(path, plat); break;
        case SourceKind::Image:
            load_image(path, plat); break;   // load_image exits folder/video/camera mode
        default: load_err_ = "unsupported file type: " + path;
    }
}

void App::load_image(const std::string& path, const Platform& plat) {
    close_camera();
    close_video();                       // leave video/camera/folder mode if we were in it
    close_folder();
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

void App::close_folder() {
    finfer_cancel_ = true;
    if (finfer_.joinable()) finfer_.join();
    finfer_cancel_ = false;
    folder_imgs_.clear(); fcache_.clear();
    cur_idx_ = -1; folder_ready_ = false; fprogress_ = 0; finfer_done_ = false;
}

// Background: forward-pass EVERY image once, caching per-image candidates (+ proto for seg)
// and timings (per-image model ms, plus folder mean/min/max and wall clock).
void App::folder_preinfer(std::vector<std::string> paths, Config c) {
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    double sum = 0, lo = 0, hi = 0; int n = 0;
    for (int i = 0; i < (int)paths.size() && !finfer_cancel_; ++i) {
        cv::Mat bgr = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (!bgr.empty()) {
            try {
                be_->infer(bgr, c);
                FolderItem it;
                it.cands = be_->candidates; it.lb = be_->cand_lb;
                it.ow = be_->cand_orig_w;   it.oh = be_->cand_orig_h;
                if (be_->is_seg()) { it.proto = be_->proto; it.pc = be_->proto_c;
                                     it.ph = be_->proto_h; it.pw = be_->proto_w; }
                it.ms = be_->infer_ms;
                it.done = true;
                fcache_[i] = std::move(it);
                sum += be_->infer_ms; ++n;
                if (n == 1 || be_->infer_ms < lo) lo = be_->infer_ms;
                if (n == 1 || be_->infer_ms > hi) hi = be_->infer_ms;
                fmean_ms_ = sum / n;              // publish live so the UI can show progress stats
                fmin_ms_ = lo; fmax_ms_ = hi; fcount_ = n;
                fwall_s_ = std::chrono::duration<double>(clk::now() - t_start).count();
            } catch (...) {}
        }
        fprogress_ = i + 1;
    }
    fwall_s_ = std::chrono::duration<double>(clk::now() - t_start).count();
    finfer_done_ = true;
}

void App::load_folder(const std::string& dir, const Platform& plat) {
    close_camera();
    close_video();
    close_folder();
    load_err_.clear();
    folder_imgs_ = gather_images(dir, 0);   // sorted image paths
    folder_path_ = dir;
    if (folder_imgs_.empty()) { cur_idx_ = -1; load_err_ = "no images in: " + dir; return; }
    if (!be_) { folder_imgs_.clear(); load_err_ = "load a model first"; return; }
    fcache_.assign(folder_imgs_.size(), FolderItem{});
    folder_ready_ = false; fprogress_ = 0; finfer_done_ = false;
    cur_idx_ = 0; scroll_to_cur_ = true;
    show_folder_item(0, plat);              // show first image now; boxes appear when pre-infer finishes
    Config c = cfg_;
    c.conf_thresh = kConfFloor; c.iou_thresh = iou_; c.stretch = (prep_ == Preprocess::Stretch);
    finfer_ = std::thread(&App::folder_preinfer, this, folder_imgs_, c);
}

// re-NMS + mask for a cached image (no decode) - used on conf/IoU change
void App::overlay_folder_item(int idx, const Platform& plat) {
    if (idx < 0 || idx >= (int)fcache_.size() || !fcache_[idx].done) {
        dets_.clear(); class_counts_.clear(); has_mask_ = false; return;
    }
    const FolderItem& it = fcache_[idx];
    seg_model_ = !it.proto.empty();
    inf_ms_ = it.ms; pre_ms_ = 0; post_ms_ = 0;   // this image's model-only time
    Config c = cfg_; c.conf_thresh = conf_; c.iou_thresh = iou_;
    dets_ = nms_and_cap(it.cands, c, it.ow, it.oh);
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    if (seg_model_) build_overlay(dets_, it.proto, it.pc, it.ph, it.pw, it.lb, it.ow, it.oh, plat);
    else has_mask_ = false;
}

// decode the image pixels + show cached overlay (if the pre-infer has finished)
void App::show_folder_item(int idx, const Platform& plat) {
    if (idx < 0 || idx >= (int)folder_imgs_.size()) return;
    cv::Mat bgr = cv::imread(folder_imgs_[idx], cv::IMREAD_COLOR);
    if (bgr.empty()) { load_err_ = "cannot read: " + folder_imgs_[idx]; return; }
    img_bgr_ = bgr; img_path_ = folder_imgs_[idx];
    cv::Mat rgba; cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    plat.upload(rgba, img_tex_);
    if (folder_ready_) overlay_folder_item(idx, plat);
    else { dets_.clear(); has_mask_ = false; }
}

void App::select_index(int i, const Platform& plat) {
    if (folder_imgs_.empty()) return;
    cur_idx_ = std::clamp(i, 0, (int)folder_imgs_.size() - 1);
    scroll_to_cur_ = true;
    show_folder_item(cur_idx_, plat);
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
    close_folder();                              // leave folder mode
    load_err_.clear();
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

// re-NMS + mask for a cached frame's candidates (no decode) - used on conf/IoU change
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

// Show frame `idx`: display its pixels + cached overlay. Forward jumps grab-skip the
// intermediate frames (decode-skip, no upload) so playback stays real-time when the UI
// can't render every frame; backward jumps seek.
void App::render_video_frame(int idx, const Platform& plat) {
    if (!cap_.isOpened() || total_frames_ <= 0) return;
    idx = std::clamp(idx, 0, total_frames_ - 1);
    if (idx != frame_idx_) {
        if (idx > frame_idx_ + 1) {                      // forward skip: grab (no decode-to-Mat)
            for (int i = frame_idx_ + 1; i < idx; ++i)
                if (!cap_.grab()) { cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx); break; }
        } else if (idx < frame_idx_ + 1) {               // backward / wrap: seek
            cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx);
        }
        cv::Mat f;
        if (!cap_.read(f) || f.empty()) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx);
            if (!cap_.read(f) || f.empty()) return;
        }
        frame_idx_ = idx;
        show_frame(f, plat);
    }
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
    close_folder();
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
    // sync inference for a single still image only (video/camera/folder own the backend elsewhere)
    if (!async_mode_ && !is_video_ && folder_imgs_.empty()) {
        if (need_reinfer_) run_inference();
        if (need_renms_)   recompute_nms();
        if (need_overlay_) rebuild_overlay(plat);
    }

    // folder: pre-infer every image (progress bar), then browse instantly from cache
    if (!folder_imgs_.empty()) {
        if (!folder_ready_) {
            if (finfer_done_) {
                if (finfer_.joinable()) finfer_.join();
                folder_ready_ = true;
                overlay_folder_item(cur_idx_, plat);   // reveal boxes for the shown image
            }
        } else if (need_renms_) {
            overlay_folder_item(cur_idx_, plat);       // conf/IoU re-NMS from cache
            need_renms_ = false;
        }
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

    // video: pre-infer all frames (progress bar), then play the raw video from cache - NO inference
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
                int steps = 0;                          // how many source frames elapsed this UI tick
                while (play_accum_ >= iv) {
                    play_accum_ -= iv;
                    if (++steps > (int)video_fps_) { play_accum_ = 0.0; break; }  // cap catch-up to ~1s
                }
                if (steps > 0) {                        // advance real-time; render_video_frame skips
                    int next = frame_idx_ + steps;
                    while (next >= total_frames_) next -= total_frames_;   // loop
                    render_video_frame(next, plat);
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

    if (show_about_) draw_about(plat);
}

// A rounded, padded, bordered "card" section - mirrors the macOS runner's grouped panels.
static void begin_card(const char* title) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.155f, 0.166f, 0.198f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border,  ImVec4(1.0f, 1.0f, 1.0f, 0.05f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 11.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14, 12));
    ImGui::BeginChild(title, ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.56f, 0.61f, 0.68f, 1.0f));   // section header
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::Spacing();
}
static void end_card() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    ImGui::Dummy(ImVec2(0, 5));   // gap between cards
}
// small dim caption above a full-width control
static void field_label(const char* t) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.60f, 0.64f, 0.70f, 1.0f));
    ImGui::TextUnformatted(t);
    ImGui::PopStyleColor();
}

void App::draw_sidebar(const Platform& plat) {
    const float ui = ImGui::GetFontSize() / 17.0f;   // DPI scale for fixed pixel offsets
    // ---- header ----
    ImGui::AlignTextToFramePadding();   // vertically center the title against the "?" button
    ImGui::TextUnformatted("YOLO-Master");
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight());
    if (ImGui::Button("?", ImVec2(ImGui::GetFrameHeight(), 0))) show_about_ = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("About / Licenses");
    ImGui::TextDisabled("Windows runner (GUI) - ONNX / ncnn / MNN");
    ImGui::Dummy(ImVec2(0, 6));

    // ---- MODEL (auto-loads; no Load button) ----
    begin_card("MODEL");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint("##model", ".onnx / ncnn dir / .mnn", model_path_, sizeof(model_path_),
                                 ImGuiInputTextFlags_EnterReturnsTrue))
        load_model(plat);
    if (ImGui::Button("Browse...", ImVec2(-1, 0))) {
        std::string p = plat.open_file("Select model", "Models\0*.onnx;*.mnn;*.param\0All\0*.*\0");
        if (!p.empty()) { std::snprintf(model_path_, sizeof(model_path_), "%s", p.c_str()); load_model(plat); }
    }
    field_label("Backend");
    const char* backends[] = {"auto", "onnx", "ncnn", "mnn"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##backend", &backend_sel_, backends, IM_ARRAYSIZE(backends)) && model_path_[0])
        load_model(plat);
    if (be_) {
        ImGui::TextColored(ImVec4(0.5f,0.85f,0.4f,1), "%s  \xc2\xb7  %s  \xc2\xb7  nc %d  \xc2\xb7  %dpx",
                           be_name_.c_str(), be_ep_.c_str(), cfg_.num_classes(), cfg_.imgsz);
        if (!be_note_.empty())
            ImGui::TextWrapped("%s", be_note_.c_str());
    } else if (!be_err_.empty()) {
        ImGui::TextColored(ImVec4(0.98f,0.4f,0.4f,1), "%s", be_err_.c_str());
    }
    end_card();

    // ---- SOURCE ----
    begin_card("SOURCE");
    if (ImGui::Button("Open image / video...", ImVec2(-1, 0))) {   // autodetect by extension
        std::string p = plat.open_file("Open image or video",
            "Media\0*.jpg;*.jpeg;*.png;*.bmp;*.mp4;*.avi;*.mov;*.mkv\0All\0*.*\0");
        if (!p.empty()) open_media(p, plat);
    }
    if (ImGui::Button("Open folder...", ImVec2(-1, 0))) {
        std::string d = plat.open_folder("Select image folder");
        if (!d.empty()) load_folder(d, plat);
    }
    if (is_cam_) {
        if (ImGui::Button("Stop Camera", ImVec2(-1, 0))) close_camera();
        ImGui::Checkbox("Mirror", &cam_mirror_);
    } else if (ImGui::Button("Live Webcam", ImVec2(-1, 0))) {
        open_camera(plat);
    }
    if (!img_bgr_.empty())
        ImGui::TextDisabled("%d x %d%s", img_bgr_.cols, img_bgr_.rows, is_cam_ ? "   live" : "");
    if (!load_err_.empty())
        ImGui::TextColored(ImVec4(0.98f,0.4f,0.4f,1), "%s", load_err_.c_str());
    end_card();

    // ---- PREPROCESS ----
    begin_card("PREPROCESS");
    field_label("Input fit");
    int pp = (int)prep_;
    const char* pps[] = {"letterbox", "stretch"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##prep", &pp, pps, IM_ARRAYSIZE(pps)) && (Preprocess)pp != prep_) {
        prep_ = (Preprocess)pp;
        if (is_video_)                  open_video(video_path_, plat);   // re-infer the clip
        else if (!folder_imgs_.empty()) load_folder(folder_path_, plat); // re-infer the folder
        else if (!is_cam_)              need_reinfer_ = true;            // camera: next frame
    }
    end_card();

    // ---- DETECTION ----
    begin_card("DETECTION");
    field_label("Confidence");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##conf", &conf_, 0.05f, 0.95f, "%.2f")) need_renms_ = true;
    field_label("IoU (NMS)");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##iou", &iou_, 0.10f, 0.90f, "%.2f")) need_renms_ = true;
    end_card();

    // ---- APPEARANCE ----
    begin_card("APPEARANCE");
    if (seg_model_) {
        field_label("Overlay");
        int ov = (int)overlay_;
        const char* ovs[] = {"both", "masks", "boxes"};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##overlay", &ov, ovs, IM_ARRAYSIZE(ovs))) overlay_ = (Overlay)ov;
    }
    if (!(seg_model_ && overlay_ == Overlay::Masks)) {   // box style is irrelevant in masks-only
        field_label("Box style");
        int st = (int)style_;
        const char* styles[] = {"hud", "solid", "neon"};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##style", &st, styles, IM_ARRAYSIZE(styles))) style_ = (BoxStyle)st;
    }
    field_label("Labels");
    int lm = (int)labels_;
    const char* lms[] = {"full", "min", "off"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##labels", &lm, lms, IM_ARRAYSIZE(lms))) labels_ = (LabelMode)lm;
    end_card();

    // ---- DEVICE ----
    begin_card("DEVICE");
    field_label("Compute");
    int dv = (int)device_;
    const char* devs[] = {"CPU", "GPU"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##device", &dv, devs, IM_ARRAYSIZE(devs)) && (Device)dv != device_) {
        device_ = (Device)dv;
        if (be_ || model_path_[0]) load_model(plat);
    }
    ImGui::TextDisabled("GPU: onnx CUDA \xc2\xb7 ncnn Vulkan \xc2\xb7 mnn OpenCL");
    end_card();

    // ---- INFERENCE ----
    begin_card("INFERENCE");
    if (be_ && !img_bgr_.empty()) {
        const bool folder = !folder_imgs_.empty();
        // "This image / frame" model-only time
        ImGui::Text(folder ? "This image" : "Model");
        ImGui::SameLine(110 * ui); ImGui::TextDisabled("%.1f ms", inf_ms_);
        if (folder) {
            // folder-wide: mean model ms + throughput (images/sec), plus min/max and wall time
            ImGui::Text("Folder avg"); ImGui::SameLine(110 * ui);
            ImGui::TextDisabled("%.1f ms  \xc2\xb7  %.1f img/s", fmean_ms_,
                                fmean_ms_ > 0 ? 1000.0 / fmean_ms_ : 0.0);
            ImGui::Text("Throughput"); ImGui::SameLine(110 * ui);
            ImGui::TextDisabled("%.1f img/s  (wall)", fwall_s_ > 0 ? fcount_ / fwall_s_ : 0.0);
            if (fcount_ > 1) {
                ImGui::Text("Min / max"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%.1f / %.1f ms", fmin_ms_, fmax_ms_);
                ImGui::Text("Total"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%d imgs  \xc2\xb7  %.2f s", fcount_, fwall_s_);
            }
        } else {
            const double total = pre_ms_ + inf_ms_ + post_ms_;
            ImGui::Text("Overall"); ImGui::SameLine(110 * ui);
            ImGui::TextDisabled("%.1f ms  \xc2\xb7  %.0f fps", total, total > 0 ? 1000.0 / total : 0.0);
        }
        ImGui::Text("Detections"); ImGui::SameLine(110 * ui); ImGui::TextDisabled("%d", (int)dets_.size());
        if (!class_counts_.empty()) {
            ImGui::Spacing();
            for (int i = 0; i < (int)class_counts_.size(); ++i) {
                if (!class_counts_[i]) continue;
                const float* c = kPalette[i % 10];
                ImGui::ColorButton("##c", ImVec4(c[0],c[1],c[2],1),
                    ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop, ImVec2(11*ui,11*ui));
                ImGui::SameLine();
                const char* nm = i < cfg_.num_classes() ? cfg_.class_names[i].c_str() : "?";
                ImGui::Text("%s", nm); ImGui::SameLine(110 * ui); ImGui::TextDisabled("%d", class_counts_[i]);
            }
        }
    } else {
        ImGui::TextDisabled("Load a model and a source to see stats.");
    }
    end_card();

    ImGui::Dummy(ImVec2(0, 2));
    ImGui::TextDisabled("(c) 2026 Thomas Li");
}

void App::draw_filelist(const Platform& plat) {
    if (!folder_ready_) {                        // pre-inference progress
        const int done = fprogress_.load(), tot = (int)folder_imgs_.size();
        ImGui::Text("Inferring %d / %d", done, tot);
        ImGui::ProgressBar(tot ? (float)done / tot : 0.f, ImVec2(-1, 0));
    } else {
        ImGui::Text("%d/%d", cur_idx_ + 1, (int)folder_imgs_.size());
        ImGui::SameLine();
        ImGui::TextDisabled("(<- ->)");
    }
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

// Draw one detection onto `dl`, mapped orig-px -> screen. `draw_shape=false` skips the box
// outline and draws only the label (masks-only overlay mode).
static void draw_box(ImDrawList* dl, const Detection& d, BoxStyle style, LabelMode labels,
                     const Config& cfg, ImVec2 origin, float scale, bool draw_shape = true) {
    const float x0 = origin.x + d.box.x * scale;
    const float y0 = origin.y + d.box.y * scale;
    const float x1 = x0 + d.box.width  * scale;
    const float y1 = y0 + d.box.height * scale;
    const ImU32 col = col_of(d.class_id);

    if (!draw_shape) {
        // no box outline; fall through to the label block below
    } else if (style == BoxStyle::Solid) {
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
    const float padX = ts.y * 0.5f, padY = ts.y * 0.22f;   // roomy chip padding
    const float chipH = ts.y + 2 * padY, chipW = ts.x + 2 * padX;
    const float rounding = chipH * 0.34f;                  // rounded pill
    float lx = x0, ly = y0 - chipH - 1.f;                  // sit just above the box top edge
    if (ly < origin.y) ly = y0 + 1.f;                      // flip inside if it would clip off-top
    const float a = (labels == LabelMode::Min) ? 0.72f : 0.86f;
    dl->AddRectFilled(ImVec2(lx, ly), ImVec2(lx + chipW, ly + chipH), col_of(d.class_id, a), rounding);
    dl->AddText(ImVec2(lx + padX, ly + padY), text_on(d.class_id), buf);
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
    const float th = ImGui::GetTextLineHeight();
    const float padX = 12.f, padY = 7.f, gap = th * 0.9f, kern = 4.f;
    const float x0 = wp.x + 16, y0 = wp.y + 16;   // top-left inset

    char fps[16], ms[16], obj[16];
    std::snprintf(fps, sizeof(fps), "%.0f", cam_fps_ema_);
    std::snprintf(ms,  sizeof(ms),  "%.0f", cam_ms_ema_);
    std::snprintf(obj, sizeof(obj), "%d",  (int)dets_.size());
    auto w = [](const char* s) { return ImGui::CalcTextSize(s).x; };

    const float dotR = th * 0.22f, dotAdv = dotR * 2 + 7;
    const float total = dotAdv + w("LIVE") + gap
                      + w(fps) + kern + w("fps") + gap
                      + w(ms)  + kern + w("ms")  + gap
                      + w(obj) + kern + w("obj");

    // pill background
    dl->AddRectFilled(ImVec2(x0 - padX, y0 - padY),
                      ImVec2(x0 + total + padX, y0 + th + padY),
                      IM_COL32(15, 17, 20, 205), (th + 2 * padY) * 0.5f);

    const ImU32 white  = IM_COL32(236, 239, 243, 255), dim = IM_COL32(146, 152, 160, 255);
    const ImU32 green  = IM_COL32(88, 214, 128, 255),  cyan   = IM_COL32(64, 206, 218, 255);
    const ImU32 orange = IM_COL32(250, 162, 74, 255),  red    = IM_COL32(244, 76, 76, 255);

    float x = x0;
    dl->AddCircleFilled(ImVec2(x + dotR, y0 + th * 0.5f), dotR, red);
    x += dotAdv;
    dl->AddText(ImVec2(x, y0), white, "LIVE");                 x += w("LIVE") + gap;
    dl->AddText(ImVec2(x, y0), green,  fps); x += w(fps) + kern;
    dl->AddText(ImVec2(x, y0), dim,    "fps");                 x += w("fps") + gap;
    dl->AddText(ImVec2(x, y0), cyan,   ms);  x += w(ms) + kern;
    dl->AddText(ImVec2(x, y0), dim,    "ms");                  x += w("ms") + gap;
    dl->AddText(ImVec2(x, y0), orange, obj); x += w(obj) + kern;
    dl->AddText(ImVec2(x, y0), dim,    "obj");
}

void App::load_about_assets(const Platform& plat) {
    assets_loaded_ = true;   // attempt once; missing files just leave a lettered placeholder
    auto load = [&](const char* file, Texture& tex) {
        const std::string path = plat.exe_dir + "/assets/ack/" + file;
        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) return;
        cv::Mat rgba;
        if (img.channels() == 4)      cv::cvtColor(img, rgba, cv::COLOR_BGRA2RGBA);
        else if (img.channels() == 3) cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
        else if (img.channels() == 1) cv::cvtColor(img, rgba, cv::COLOR_GRAY2RGBA);
        else return;
        plat.upload(rgba, tex);
    };
    load("skywalker-lt.png", avatar_tex_);
    const char* keys[] = {"tencent", "ultralytics", "onnx", "mnn", "opencv", "imgui"};
    for (const char* k : keys) load((std::string(k) + ".png").c_str(), logos_[k]);
}

void App::draw_about(const Platform& plat) {
    if (!assets_loaded_) load_about_assets(plat);
    const float ui = ImGui::GetFontSize() / 17.0f;
    ImGui::SetNextWindowSize(ImVec2(560 * ui, 640 * ui), ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::Begin("About", &show_about_, ImGuiWindowFlags_NoCollapse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        auto link = [&](const char* label, const char* url) {
            ImGui::TextColored(ImVec4(0.37f, 0.64f, 0.88f, 1.f), "%s", label);
            if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            if (ImGui::IsItemClicked() && plat.open_url) plat.open_url(url);
        };
        auto logo_of = [&](const char* key) -> Texture* {
            if (!key) return nullptr;
            auto it = logos_.find(key);
            return (it != logos_.end() && it->second.id) ? &it->second : nullptr;
        };

        ImGui::TextUnformatted(kAppName);
        ImGui::SameLine();
        ImGui::TextDisabled("%s", kAppSubtitle);
        ImGui::TextDisabled("Version %s", kAppVersion);
        ImGui::TextWrapped("%s", kAppTagline);

        ImGui::Dummy(ImVec2(0, 6 * ui));
        // author: round avatar + name/affiliation/link
        const float av = 64 * ui;
        if (avatar_tex_.id) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            dl->AddImageRounded((ImTextureID)avatar_tex_.id, p, ImVec2(p.x + av, p.y + av),
                                ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, av * 0.5f);
            dl->AddCircle(ImVec2(p.x + av * 0.5f, p.y + av * 0.5f), av * 0.5f, IM_COL32(255, 255, 255, 45), 0, 1.5f);
            ImGui::Dummy(ImVec2(av + 10 * ui, av));
            ImGui::SameLine();
        }
        ImGui::BeginGroup();
        ImGui::TextUnformatted(kAuthor);
        ImGui::TextDisabled("%s", kAuthorAffil);
        link("github.com/skywalker-lt", kAuthorUrl);
        ImGui::EndGroup();
        link("Source - github.com/skywalker-lt/yolo-master-edge", kSourceUrl);

        ImGui::Dummy(ImVec2(0, 6 * ui));
        ImGui::SeparatorText("Acknowledgements");
        for (const auto& a : kAcks) {
            const float ls = 40 * ui;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            Texture* lt = logo_of(a.logo);
            if (lt && lt->id) {
                dl->AddImageRounded((ImTextureID)lt->id, p, ImVec2(p.x + ls, p.y + ls),
                                    ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 7 * ui);
            } else {   // lettered placeholder tile
                dl->AddRectFilled(p, ImVec2(p.x + ls, p.y + ls), IM_COL32(58, 64, 76, 255), 7 * ui);
                const char L[2] = { a.name[0], 0 };
                const ImVec2 tz = ImGui::CalcTextSize(L);
                dl->AddText(ImVec2(p.x + (ls - tz.x) * 0.5f, p.y + (ls - tz.y) * 0.5f),
                            IM_COL32(176, 182, 192, 255), L);
            }
            ImGui::Dummy(ImVec2(ls + 10 * ui, ls));
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::TextUnformatted(a.name);
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextWrapped("%s", a.blurb);
            ImGui::PopTextWrapPos();
            ImGui::PopStyleColor();
            link(a.repo, a.repo);
            ImGui::EndGroup();
            ImGui::Dummy(ImVec2(0, 6 * ui));
        }

        ImGui::Dummy(ImVec2(0, 4 * ui));
        if (ImGui::CollapsingHeader("License (AGPL-3.0)")) {
            ImGui::BeginChild("lic", ImVec2(0, 240 * ui), true,
                              ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::TextUnformatted(kLicenseText);
            ImGui::EndChild();
        }

        ImGui::Dummy(ImVec2(0, 4 * ui));
        if (ImGui::Button("Close", ImVec2(-1, 0))) show_about_ = false;
    }
    ImGui::End();
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
    // masks-only hides the box outline but keeps the labels
    const bool box_shape  = !seg_model_ || overlay_ != Overlay::Masks;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 br(origin.x + disp.x, origin.y + disp.y);
    dl->AddImage((ImTextureID)img_tex_.id, origin, br);
    if (show_masks)   // seg overlay is same dims as the image -> same rect, alpha-blended by ImGui
        dl->AddImage((ImTextureID)mask_tex_.id, origin, br);
    for (const auto& d : dets_)
        draw_box(dl, d, style_, labels_, cfg_, origin, scale, box_shape);

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
