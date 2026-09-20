#include "http_api.hpp"
#include "serialize.hpp"
#include "App.h"
#include "Multipart.h"
#include <future>
#include <map>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>
#ifdef HAVE_VIDEOIO
#include <opencv2/videoio.hpp>
#endif

namespace yolomaster::server {

namespace {

using Res = uWS::HttpResponse<false>;
using Req = uWS::HttpRequest;

std::string now_iso() {
    const auto t = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(t);
    char b[40]; std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", std::gmtime(&tt));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count() % 1000;
    char o[48]; std::snprintf(o, sizeof o, "%s.%03dZ", b, int(ms)); return o;
}

std::string make_rid(ServerState& st) {
    char b[32]; std::snprintf(b, sizeof b, "%016llx", (unsigned long long)(++st.request_seq)); return b;
}

// A response that may be finished from another thread's completion via loop->defer.
struct Pending {
    Res* res; uWS::Loop* loop; std::atomic<bool> aborted{false};
    std::string rid, method, path; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ServerState* st;
    Pending(Res* r, uWS::Loop* l, ServerState* s) : res(r), loop(l), st(s) {}
};

void access_log(ServerState& st, const Pending& p, int status, size_t bytes) {
    if (!st.cfg.access_log) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p.t0).count();
    std::fprintf(stderr, "%s access rid=%s %s %s %d %zuB %.2fms\n", now_iso().c_str(), p.rid.c_str(),
                 p.method.c_str(), p.path.c_str(), status, bytes, ms);
}

void cors_headers(Res* res, const ServerConfig& cfg) {
    if (!cfg.cors) return;
    res->writeHeader("Access-Control-Allow-Origin", "*");
    res->writeHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res->writeHeader("Access-Control-Allow-Headers", "Content-Type, X-Request-Id");
}

const char* status_text(int code) {
    switch (code) {
        case 200: return "200 OK"; case 202: return "202 Accepted"; case 204: return "204 No Content";
        case 400: return "400 Bad Request"; case 404: return "404 Not Found"; case 405: return "405 Method Not Allowed";
        case 409: return "409 Conflict"; case 413: return "413 Payload Too Large"; case 415: return "415 Unsupported Media Type";
        case 422: return "422 Unprocessable Entity"; case 429: return "429 Too Many Requests";
        case 500: return "500 Internal Server Error"; case 501: return "501 Not Implemented";
        case 503: return "503 Service Unavailable"; case 504: return "504 Gateway Timeout";
        default: return "500 Internal Server Error";
    }
}

// Must run on the loop thread.
void send(Pending& p, int code, const std::string& body, const char* ctype, const std::string& extra_hdr = "", const std::string& extra_val = "") {
    if (p.aborted) return;
    p.res->cork([&] {
        p.res->writeStatus(status_text(code));
        p.res->writeHeader("Content-Type", ctype);
        p.res->writeHeader("X-Request-Id", p.rid);
        if (!extra_hdr.empty()) p.res->writeHeader(extra_hdr, extra_val);
        cors_headers(p.res, p.st->cfg);
        p.res->end(body);
    });
    access_log(*p.st, p, code, body.size());
}
void send_json(Pending& p, int code, const nlohmann::json& j, const std::string& eh = "", const std::string& ev = "") {
    send(p, code, j.dump(), "application/json", eh, ev);
}
void send_error(Pending& p, int code, const std::string& msg, const std::string& eh = "", const std::string& ev = "") {
    send_json(p, code, {{"error", msg}, {"code", code}, {"request_id", p.rid}}, eh, ev);
}
// Deliver from any thread.
void deliver(const std::shared_ptr<Pending>& p, std::function<void(Pending&)> fn) {
    if (p->aborted) return;
    p->loop->defer([p, fn = std::move(fn)]() { if (!p->aborted) fn(*p); });
}

float qf(Req* req, const char* k, float def) { auto v = req->getQuery(k); if (v.empty()) return def; try { return std::stof(std::string(v)); } catch (...) { return def; } }
int qi(Req* req, const char* k, int def) { auto v = req->getQuery(k); if (v.empty()) return def; try { return std::stoi(std::string(v)); } catch (...) { return def; } }
std::string qs(Req* req, const char* k, const std::string& def = "") { auto v = req->getQuery(k); return v.empty() ? def : std::string(v); }
bool qb(Req* req, const char* k, bool def) { auto v = req->getQuery(k); if (v.empty()) return def; return v == "1" || v == "true" || v == "yes"; }

InferParams params_from(Req* req, const std::string& ret) {
    InferParams p;
    p.conf = qf(req, "conf", -1); p.iou = qf(req, "iou", -1); p.max_det = qi(req, "max_det", -1);
    auto ml = req->getQuery("multi_label"); if (!ml.empty()) p.multi_label = (ml == "1" || ml == "true") ? 1 : 0;
    p.slicing = qs(req, "slicing"); p.tile_size = qi(req, "tile_size", -1);
    p.annotated = ret == "annotated"; p.mask_overlay = qs(req, "masks") == "overlay"; p.mask_coeffs = qb(req, "mask_coeffs", false);
    p.jpeg_quality = qi(req, "quality", 90);
    return p;
}

// Collect the request body (bounded), then call `cb(body)` on the loop thread.
// Oversized bodies are drained (discarded) to the end and then answered 413: ending the response
// while the client is still uploading makes the kernel RST the socket and the client never sees
// the status.
void read_body(const std::shared_ptr<Pending>& p, size_t max_bytes, std::function<void(std::string&&)> cb) {
    auto buf = std::make_shared<std::string>();
    auto too_big = std::make_shared<bool>(false);
    p->res->onAborted([p] { p->aborted = true; });
    p->res->onData([p, buf, too_big, max_bytes, cb = std::move(cb)](std::string_view chunk, bool last) mutable {
        if (p->aborted) return;
        if (!*too_big) {
            if (buf->size() + chunk.size() > max_bytes) { *too_big = true; std::string().swap(*buf); }
            else buf->append(chunk.data(), chunk.size());
        }
        if (!last) return;
        if (*too_big) send_error(*p, 413, "body exceeds max_body_mb=" + std::to_string(p->st->cfg.max_body_mb));
        else cb(std::move(*buf));
    });
}

// First file part of a multipart body (or the raw body when not multipart).
bool extract_image(const std::string& content_type, const std::string& body, std::string& out, std::string& filename) {
    if (content_type.rfind("multipart/form-data", 0) == 0) {
        uWS::MultipartParser mp(content_type);
        mp.setBody(body);
        std::pair<std::string_view, std::string_view> headers[10];
        while (auto part = mp.getNextPart(headers)) {
            std::string disp;
            for (auto& h : headers) { if (h.first.empty()) break; if (h.first == "content-disposition") disp = std::string(h.second); }
            const auto fn = disp.find("filename=\"");
            if (fn != std::string::npos) { const auto e = disp.find('"', fn + 10); filename = disp.substr(fn + 10, e == std::string::npos ? std::string::npos : e - fn - 10); }
            if (fn != std::string::npos || disp.find("name=\"file\"") != std::string::npos || disp.find("name=\"image\"") != std::string::npos) {
                out.assign(part->data(), part->size()); return true;
            }
        }
        return false;
    }
    out = body; return !out.empty();
}

// Every file part of a multipart body, in order.
std::vector<std::pair<std::string, std::string>> extract_all(const std::string& content_type, const std::string& body) {
    std::vector<std::pair<std::string, std::string>> v;
    if (content_type.rfind("multipart/form-data", 0) != 0) return v;
    uWS::MultipartParser mp(content_type);
    mp.setBody(body);
    std::pair<std::string_view, std::string_view> headers[10];
    int i = 0;
    while (auto part = mp.getNextPart(headers)) {
        std::string disp, name = "part" + std::to_string(i++);
        for (auto& h : headers) { if (h.first.empty()) break; if (h.first == "content-disposition") disp = std::string(h.second); }
        const auto fn = disp.find("filename=\"");
        if (fn == std::string::npos) continue;
        const auto e = disp.find('"', fn + 10);
        name = disp.substr(fn + 10, e == std::string::npos ? std::string::npos : e - fn - 10);
        v.emplace_back(name, std::string(part->data(), part->size()));
    }
    return v;
}

WorkerPool* resolve_pool(ServerState& st, Req* req, std::shared_ptr<Pending>& p, std::string& model_id) {
    model_id = qs(req, "model");
    if (model_id.empty()) {
        // single-model convenience: the only configured model is the default
        auto specs = st.reg.specs();
        if (specs.size() == 1) model_id = specs[0].id;
        else { send_error(*p, 400, "missing ?model=<id>"); return nullptr; }
    }
    WorkerPool* pool = st.reg.get(model_id);
    if (!pool) {
        if (!st.reg.has_spec(model_id)) { send_error(*p, 404, "unknown model: " + model_id); return nullptr; }
        std::string err;
        if (!st.reg.load(model_id, err)) { send_error(*p, 500, err); return nullptr; }
        pool = st.reg.get(model_id);
    }
    if (pool->failed()) { send_error(*p, 503, "model failed to load: " + pool->init_error()); return nullptr; }
    if (!pool->ready()) { send_error(*p, 503, "model warming up: " + model_id, "Retry-After", "2"); return nullptr; }
    return pool;
}

// Serialize one result per the `return` mode; runs on the loop thread.
void respond_result(Pending& p, InferResult& r, const std::string& model_id, const std::string& ret, int image_id, bool coco91, bool names) {
    if (r.http_status != 200) { send_error(p, r.http_status, r.error, r.http_status == 503 ? "Retry-After" : "", r.http_status == 503 ? "1" : ""); return; }
    p.res->writeHeader("X-Infer-Ms", std::to_string(r.infer_ms));
    if (ret == "txt") send(p, 200, result_txt(r), "text/plain");
    else if (ret == "coco") send_json(p, 200, result_coco(r, image_id, coco91));
    else if (ret == "annotated") send(p, 200, std::string(r.annotated_jpg.begin(), r.annotated_jpg.end()), "image/jpeg",
                                      "X-Detections", std::to_string(r.dets.size()));
    else send_json(p, 200, result_json(r, model_id, p.rid, names));
}

Job make_job(ServerState& st, const std::shared_ptr<Pending>& p, std::string&& image, const InferParams& params) {
    Job j; j.request_id = p->rid; j.image = std::move(image); j.params = params;
    j.enqueued = std::chrono::steady_clock::now();
    j.deadline = j.enqueued + std::chrono::milliseconds(st.cfg.request_timeout_ms);
    return j;
}

// ---- WebSocket stream: keep-latest backpressure per connection ----
struct WsSession {
    std::string model_id; InferParams params; std::string ret = "json";
    std::atomic<bool> busy{false}, closed{false};
    std::string pending; bool has_pending = false; uint64_t seq = 0, pending_seq = 0, dropped = 0;
    uWS::WebSocket<false, true, WsSession*>* ws = nullptr; uWS::Loop* loop = nullptr;
};
using Ws = uWS::WebSocket<false, true, WsSession*>;

void ws_submit(ServerState& st, std::shared_ptr<WsSession> s, std::string&& frame, uint64_t seq);

void ws_on_result(ServerState& st, std::shared_ptr<WsSession> s, uint64_t seq, InferResult&& r) {
    // on the loop thread
    if (s->closed) return;
    s->busy = false;
    nlohmann::json j = r.http_status == 200 ? result_json(r, s->model_id, std::to_string(seq), true)
                                            : nlohmann::json{{"error", r.error}, {"code", r.http_status}};
    j["seq"] = seq; j["dropped"] = s->dropped;
    if (s->ret == "annotated" && r.http_status == 200 && !r.annotated_jpg.empty())
        s->ws->send(std::string_view(reinterpret_cast<const char*>(r.annotated_jpg.data()), r.annotated_jpg.size()), uWS::OpCode::BINARY);
    s->ws->send(j.dump(), uWS::OpCode::TEXT);
    if (s->has_pending) { s->has_pending = false; ws_submit(st, s, std::move(s->pending), s->pending_seq); }
}

void ws_submit(ServerState& st, std::shared_ptr<WsSession> s, std::string&& frame, uint64_t seq) {
    WorkerPool* pool = st.reg.get(s->model_id);
    if (!pool || !pool->ready()) { s->ws->send(nlohmann::json{{"error", "model not ready"}, {"seq", seq}}.dump(), uWS::OpCode::TEXT); return; }
    s->busy = true;
    Job j; j.request_id = "ws-" + std::to_string(seq); j.image = std::move(frame); j.params = s->params;
    j.enqueued = std::chrono::steady_clock::now(); j.deadline = j.enqueued + std::chrono::milliseconds(st.cfg.request_timeout_ms);
    uWS::Loop* loop = s->loop; ServerState* stp = &st;
    j.done = [stp, s, seq, loop](InferResult&& r) mutable {
        auto rp = std::make_shared<InferResult>(std::move(r));
        loop->defer([stp, s, seq, rp]() { ws_on_result(*stp, s, seq, std::move(*rp)); });
    };
    if (!pool->submit(std::move(j))) {
        s->busy = false; pool->metrics.dropped_frames.fetch_add(1);
        s->ws->send(nlohmann::json{{"error", "queue full"}, {"code", 503}, {"seq", seq}}.dump(), uWS::OpCode::TEXT);
    }
}

} // namespace

void request_stop(ServerState& st) {
    st.stopping = true;
    std::lock_guard<std::mutex> g(st.loops_m);
    for (auto& c : st.closers) { auto fn = c.second; c.first->defer([fn]() { fn(); }); }
}

bool run_http_loop(ServerState& st, int loop_index, std::atomic<int>& bound_count) {
    const ServerConfig& cfg = st.cfg;
    const size_t max_body = size_t(cfg.max_body_mb) << 20;
    uWS::App app;
    uWS::Loop* loop = uWS::Loop::get();

    auto begin = [&](Res* res, Req* req) {
        auto p = std::make_shared<Pending>(res, loop, &st);
        p->rid = make_rid(st); p->method = std::string(req->getMethod()); p->path = std::string(req->getUrl());
        res->onAborted([p] { p->aborted = true; });
        return p;
    };

    app.get("/", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        send_json(*p, 200, {{"service", "yolomaster-edge api"}, {"version", YM_SERVER_VERSION}, {"runtime", YM_VERSION}, {"commit", YM_GIT_COMMIT}, {"docs", "/docs/API.md"},
                            {"endpoints", {"/healthz", "/readyz", "/v1/models", "/v1/infer", "/v1/infer/batch", "/v1/video", "/v1/stream (ws)", "/v1/stats", "/metrics"}}});
    });
    app.get("/healthz", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        send_json(*p, st.stopping ? 503 : 200, {{"status", st.stopping ? "stopping" : "ok"}, {"uptime_s", st.uptime_s()}});
    });
    app.get("/readyz", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        std::string why; const bool ok = !st.stopping && st.reg.ready(why);
        send_json(*p, ok ? 200 : 503, {{"ready", ok}, {"reason", ok ? "" : why}});
    });
    app.get("/v1/models", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& s : st.reg.specs()) {
            WorkerPool* pool = st.reg.get(s.id);
            nlohmann::json j = pool ? pool->info() : nlohmann::json(s);
            j["loaded"] = pool != nullptr;
            arr.push_back(j);
        }
        send_json(*p, 200, {{"models", arr}});
    });
    app.post("/v1/models/:id/load", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        const std::string id(req->getParameter("id"));
        std::string err;
        if (!st.reg.load(id, err)) { send_error(*p, 404, err); return; }
        send_json(*p, 202, {{"model", id}, {"loaded", true}, {"ready", st.reg.get(id)->ready()}});
    });
    app.post("/v1/models/:id/unload", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        const std::string id(req->getParameter("id"));
        if (!st.reg.unload(id)) { send_error(*p, 404, "model not loaded: " + id); return; }
        send_json(*p, 200, {{"model", id}, {"loaded", false}});
    });
    app.get("/v1/stats", [&](Res* res, Req* req) { auto p = begin(res, req); send_json(*p, 200, stats_json(st.reg, st.uptime_s())); });
    app.get("/metrics", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        send(*p, 200, prometheus_text(st.reg, st.cfg, st.uptime_s()), "text/plain; version=0.0.4");
    });
    app.options("/*", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        res->writeHeader("Access-Control-Max-Age", "86400");
        send(*p, 204, "", "text/plain");
    });

    // ---- single image ----
    app.post("/v1/infer", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        std::string model_id;
        WorkerPool* pool = resolve_pool(st, req, p, model_id);
        if (!pool) return;
        const std::string ctype(req->getHeader("content-type"));
        const std::string ret = qs(req, "return", "json");
        if (ret != "json" && ret != "txt" && ret != "coco" && ret != "annotated") { send_error(*p, 400, "return must be json|txt|coco|annotated"); return; }
        const InferParams params = params_from(req, ret);
        const int image_id = qi(req, "image_id", 0); const bool coco91 = qb(req, "coco91", false); const bool names = qb(req, "names", true);
        read_body(p, max_body, [&st, p, pool, ctype, ret, params, model_id, image_id, coco91, names](std::string&& body) {
            std::string img, fname;
            if (!extract_image(ctype, body, img, fname)) { send_error(*p, 400, "no image in request body (send raw bytes or multipart field 'file')"); return; }
            Job j = make_job(st, p, std::move(img), params);
            j.done = [p, model_id, ret, image_id, coco91, names](InferResult&& r) {
                auto rp = std::make_shared<InferResult>(std::move(r));
                deliver(p, [rp, model_id, ret, image_id, coco91, names](Pending& pp) { respond_result(pp, *rp, model_id, ret, image_id, coco91, names); });
            };
            if (!pool->submit(std::move(j))) send_error(*p, 503, "queue full for model " + model_id, "Retry-After", "1");
        });
    });

    // ---- multi-file: K independent batch-1 jobs spread over the pool ----
    app.post("/v1/infer/batch", [&](Res* res, Req* req) {
        auto p = begin(res, req);
        std::string model_id;
        WorkerPool* pool = resolve_pool(st, req, p, model_id);
        if (!pool) return;
        const std::string ctype(req->getHeader("content-type"));
        const InferParams params = params_from(req, "json");
        const bool names = qb(req, "names", true);
        read_body(p, max_body, [&st, p, pool, ctype, params, model_id, names](std::string&& body) {
            auto files = extract_all(ctype, body);
            if (files.empty()) { send_error(*p, 400, "multipart/form-data with one or more file parts required"); return; }
            struct Agg { std::mutex m; std::vector<nlohmann::json> out; size_t remaining; };
            auto agg = std::make_shared<Agg>(); agg->out.resize(files.size()); agg->remaining = files.size();
            for (size_t i = 0; i < files.size(); ++i) {
                Job j = make_job(st, p, std::move(files[i].second), params);
                j.request_id = p->rid + "-" + std::to_string(i);
                const std::string fname = files[i].first;
                j.done = [p, agg, i, fname, model_id, names](InferResult&& r) {
                    nlohmann::json one = r.http_status == 200 ? result_json(r, model_id, p->rid + "-" + std::to_string(i), names)
                                                              : nlohmann::json{{"error", r.error}, {"code", r.http_status}};
                    one["file"] = fname;
                    bool last = false;
                    { std::lock_guard<std::mutex> g(agg->m); agg->out[i] = std::move(one); last = (--agg->remaining == 0); }
                    if (last) deliver(p, [agg](Pending& pp) { send_json(pp, 200, {{"request_id", pp.rid}, {"count", agg->out.size()}, {"results", agg->out}}); });
                };
                if (!pool->submit(std::move(j))) {
                    // fail this part immediately; the aggregate still completes
                    InferResult r; r.http_status = 503; r.error = "queue full";
                    j.done(std::move(r));
                }
            }
        });
    });

    // ---- video upload -> NDJSON stream of per-frame results ----
    app.post("/v1/video", [&](Res* res, Req* req) {
        auto p = begin(res, req);
#ifndef HAVE_VIDEOIO
        send_error(*p, 501, "server built without OpenCV videoio (PORTABLE build)");
        return;
#else
        std::string model_id;
        WorkerPool* pool = resolve_pool(st, req, p, model_id);
        if (!pool) return;
        const InferParams params = params_from(req, "json");
        const int every = std::max(1, qi(req, "every", 1));
        const int max_frames = qi(req, "max_frames", 0);
        const bool names = qb(req, "names", true);
        read_body(p, max_body, [&st, p, pool, params, every, max_frames, model_id, names](std::string&& body) {
            namespace fs = std::filesystem;
            const fs::path tmp = fs::temp_directory_path() / ("ym-video-" + p->rid + ".bin");
            { std::ofstream f(tmp, std::ios::binary); f.write(body.data(), std::streamsize(body.size())); }
            p->res->writeStatus("200 OK"); p->res->writeHeader("Content-Type", "application/x-ndjson");
            p->res->writeHeader("X-Request-Id", p->rid); cors_headers(p->res, st.cfg);
            // decoder thread: sequential frames, one in flight, order preserved
            std::thread([&st, p, pool, params, every, max_frames, model_id, names, tmp]() {
                cv::VideoCapture cap(tmp.string());
                nlohmann::json tail = {{"done", true}};
                if (!cap.isOpened()) { tail["error"] = "cannot decode video"; }
                int idx = 0, sent = 0;
                cv::Mat frame;
                while (cap.isOpened() && !p->aborted && cap.read(frame)) {
                    const int fi = idx++;
                    if (fi % every) continue;
                    if (max_frames > 0 && sent >= max_frames) break;
                    if (!frame.isContinuous()) frame = frame.clone();
                    Job j; j.request_id = p->rid + "-f" + std::to_string(fi); j.params = params;
                    j.raw_w = frame.cols; j.raw_h = frame.rows; j.image.assign(reinterpret_cast<const char*>(frame.data), size_t(frame.total()) * 3);
                    j.enqueued = std::chrono::steady_clock::now(); j.deadline = j.enqueued + std::chrono::milliseconds(st.cfg.request_timeout_ms);
                    std::promise<InferResult> pr; auto fut = pr.get_future();
                    j.done = [&pr](InferResult&& r) { pr.set_value(std::move(r)); };
                    if (!pool->submit(std::move(j))) { tail["error"] = "queue full"; break; }
                    InferResult r = fut.get();
                    nlohmann::json line = r.http_status == 200 ? result_json(r, model_id, p->rid, names) : nlohmann::json{{"error", r.error}, {"code", r.http_status}};
                    line["frame"] = fi;
                    const std::string s = line.dump() + "\n";
                    p->loop->defer([p, s]() { if (!p->aborted) p->res->write(s); });
                    ++sent;
                }
                tail["frames"] = sent; tail["decoded"] = idx;
                const std::string s = tail.dump() + "\n";
                p->loop->defer([p, s]() { if (!p->aborted) { p->res->write(s); p->res->end(); access_log(*p->st, *p, 200, 0); } });
                std::error_code ec; fs::remove(tmp, ec);
            }).detach();
        });
#endif
    });

    // ---- WebSocket stream ----
    app.ws<WsSession*>("/v1/stream", {
        .compression = uWS::DISABLED,
        .maxPayloadLength = unsigned(cfg.ws_max_payload_mb) << 20,
        .idleTimeout = 120,
        .maxBackpressure = 4u << 20,
        .closeOnBackpressureLimit = false,
        .resetIdleTimeoutOnSend = true,
        .sendPingsAutomatically = true,
        .upgrade = [&](Res* res, Req* req, us_socket_context_t* ctx) {
            std::string model_id = qs(req, "model");
            if (model_id.empty()) { auto specs = st.reg.specs(); if (specs.size() == 1) model_id = specs[0].id; }
            std::string err;
            if (model_id.empty() || !st.reg.has_spec(model_id) || !st.reg.load(model_id, err)) {
                res->writeStatus("404 Not Found")->end("unknown model");
                return;
            }
            auto* s = new WsSession();
            s->model_id = model_id; s->params = params_from(req, qs(req, "return", "json")); s->ret = qs(req, "return", "json");
            s->params.annotated = s->ret == "annotated";
            res->template upgrade<WsSession*>(std::move(s), req->getHeader("sec-websocket-key"), req->getHeader("sec-websocket-protocol"),
                                             req->getHeader("sec-websocket-extensions"), ctx);
        },
        .open = [&](Ws* ws) {
            WsSession* s = *ws->getUserData(); s->ws = ws; s->loop = loop;
            ws->send(nlohmann::json{{"hello", "yolomaster stream"}, {"model", s->model_id}, {"protocol", "binary frame in -> json per frame out; text json {conf,iou,max_det,return} updates params"}}.dump(), uWS::OpCode::TEXT);
        },
        .message = [&](Ws* ws, std::string_view msg, uWS::OpCode op) {
            WsSession* raw = *ws->getUserData();
            if (op == uWS::OpCode::TEXT) {
                try {
                    auto j = nlohmann::json::parse(msg);
                    if (j.contains("conf")) raw->params.conf = j["conf"].get<float>();
                    if (j.contains("iou")) raw->params.iou = j["iou"].get<float>();
                    if (j.contains("max_det")) raw->params.max_det = j["max_det"].get<int>();
                    if (j.contains("return")) { raw->ret = j["return"].get<std::string>(); raw->params.annotated = raw->ret == "annotated"; }
                    ws->send(nlohmann::json{{"ok", true}, {"params", {{"conf", raw->params.conf}, {"iou", raw->params.iou}, {"max_det", raw->params.max_det}, {"return", raw->ret}}}}.dump(), uWS::OpCode::TEXT);
                } catch (const std::exception& e) { ws->send(nlohmann::json{{"error", std::string("bad json: ") + e.what()}}.dump(), uWS::OpCode::TEXT); }
                return;
            }
            // binary frame: keep-latest
            // shared_ptr alias so completions after close are harmless
            static thread_local std::map<WsSession*, std::shared_ptr<WsSession>> owners;
            auto& sp = owners[raw]; if (!sp) sp = std::shared_ptr<WsSession>(raw, [](WsSession*) {});
            const uint64_t seq = ++raw->seq;
            if (raw->busy) {
                if (raw->has_pending) { ++raw->dropped; WorkerPool* pool = st.reg.get(raw->model_id); if (pool) pool->metrics.dropped_frames.fetch_add(1); }
                raw->pending.assign(msg.data(), msg.size()); raw->has_pending = true; raw->pending_seq = seq;
                return;
            }
            ws_submit(st, sp, std::string(msg), seq);
        },
        .close = [&](Ws* ws, int, std::string_view) {
            WsSession* s = *ws->getUserData();
            s->closed = true; s->ws = nullptr;
            // deferred delete: in-flight completions hold the alias and check `closed`
            uWS::Loop::get()->defer([s]() { /* leak-free after all defers drained */ delete s; });
        }
    });

    app.any("/*", [&](Res* res, Req* req) { auto p = begin(res, req); send_error(*p, 404, "no such route: " + std::string(req->getUrl())); });

    bool ok = false;
    app.listen(cfg.host, cfg.port, [&](us_listen_socket_t* ls) {
        ok = ls != nullptr;
        if (ok) {
            ++bound_count;
            std::lock_guard<std::mutex> g(st.loops_m);
            st.closers.emplace_back(loop, [&app, ls]() { us_listen_socket_close(0, ls); app.close(); });
        }
    });
    if (!ok) { std::cerr << "[http] loop " << loop_index << ": cannot bind " << cfg.host << ":" << cfg.port << "\n"; return false; }
    if (loop_index == 0) std::cerr << now_iso() << " [http] listening on http://" << cfg.host << ":" << cfg.port << " (" << cfg.loop_threads << " loop threads)\n";
    app.run();
    return true;
}

} // namespace yolomaster::server
