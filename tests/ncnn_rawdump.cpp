// Minimal ncnn raw-tensor dumper for parity validation.
//
// The python ncnn wheels segfault at inference on this machine (even on known-good
// models the C++ runtime handles fine), so the mixture harness extracts raw tensors
// through this tool, linked against the SAME vendored ncnn SDK the edge runtime ships.
//
// usage: ncnn_rawdump model.param model.bin in.f32 C H W out.f32
//   in.f32: raw float32, channel-major [C,H,W]; out.f32: raw float32 out0, channel-major.
//   Prints "dims=? c=? h=? w=?" for the caller to reshape.
#include <net.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 8 && argc != 9) { fprintf(stderr, "usage: %s param bin in.f32 C H W out.f32 [blob]\n", argv[0]); return 2; }
    const char* blob = argc == 9 ? argv[8] : "out0";
    const int C = atoi(argv[4]), H = atoi(argv[5]), W = atoi(argv[6]);
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    if (net.load_param(argv[1]) != 0) { fprintf(stderr, "load_param failed\n"); return 3; }
    if (net.load_model(argv[2]) != 0) { fprintf(stderr, "load_model failed\n"); return 3; }

    std::vector<float> buf((size_t)C * H * W);
    FILE* f = fopen(argv[3], "rb");
    if (!f || fread(buf.data(), 4, buf.size(), f) != buf.size()) { fprintf(stderr, "input read failed\n"); return 4; }
    fclose(f);
    ncnn::Mat in(W, H, C);
    for (int c = 0; c < C; ++c)
        memcpy(in.channel(c), buf.data() + (size_t)c * H * W, (size_t)H * W * 4);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract(blob, out) != 0) { fprintf(stderr, "extract failed\n"); return 5; }
    printf("dims=%d c=%d h=%d w=%d\n", out.dims, out.c, out.h, out.w);

    FILE* g = fopen(argv[7], "wb");
    if (!g) { fprintf(stderr, "output open failed\n"); return 6; }
    const int chans = out.dims == 3 ? out.c : 1;
    for (int c = 0; c < chans; ++c)
        fwrite(out.dims == 3 ? (const float*)out.channel(c) : (const float*)out.data,
               4, (size_t)out.h * out.w, g);
    fclose(g);
    return 0;
}
