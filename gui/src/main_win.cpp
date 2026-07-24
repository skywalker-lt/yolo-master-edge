// Windows platform layer: Win32 window + D3D11 device/swapchain + Dear ImGui bootstrap.
// Injects texture-upload + file-dialog services into the portable App (app.cpp).
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include <commdlg.h>   // GetOpenFileNameA / OPENFILENAMEA (not pulled in under WIN32_LEAN_AND_MEAN)
#include <shobjidl.h>
#include <string>
#include "app.hpp"

// ---- D3D11 globals (mirrors the upstream imgui example_win32_directx11) ----
static ID3D11Device*           g_pd3dDevice        = nullptr;
static ID3D11DeviceContext*    g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*         g_pSwapChain        = nullptr;
static ID3D11RenderTargetView* g_mainRTV           = nullptr;
static HWND                    g_hwnd              = nullptr;

static void CreateRenderTarget() {
    ID3D11Texture2D* back = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&back));
    if (back) { g_pd3dDevice->CreateRenderTargetView(back, nullptr, &g_mainRTV); back->Release(); }
}
static void CleanupRenderTarget() {
    if (g_mainRTV) { g_mainRTV->Release(); g_mainRTV = nullptr; }
}
static bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    UINT flags = 0;
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            levels, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &fl,
            &g_pd3dDeviceContext) != S_OK)
        return false;
    CreateRenderTarget();
    return true;
}
static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain)        { g_pSwapChain->Release();        g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext){ g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice)       { g_pd3dDevice->Release();        g_pd3dDevice = nullptr; }
}

// ---- platform services handed to the portable App ----
static bool UploadTexture(const cv::Mat& rgba, gui::Texture& tex) {
    if (rgba.empty() || rgba.type() != CV_8UC4 || !g_pd3dDevice) return false;
    if (tex.id) { ((IUnknown*)tex.id)->Release(); tex.id = nullptr; }   // free previous SRV
    cv::Mat cont = rgba.isContinuous() ? rgba : rgba.clone();

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = cont.cols; desc.Height = cont.rows;
    desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sub = {};
    sub.pSysMem = cont.data;
    sub.SysMemPitch = (UINT)cont.step;

    ID3D11Texture2D* pTex = nullptr;
    if (FAILED(g_pd3dDevice->CreateTexture2D(&desc, &sub, &pTex)) || !pTex) return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.Format = desc.Format;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvd.Texture2D.MipLevels = 1;
    ID3D11ShaderResourceView* pSRV = nullptr;
    HRESULT hr = g_pd3dDevice->CreateShaderResourceView(pTex, &srvd, &pSRV);
    pTex->Release();                                   // SRV keeps its own ref
    if (FAILED(hr)) return false;
    tex.id = pSRV; tex.w = cont.cols; tex.h = cont.rows;
    return true;
}

static std::string OpenFileDialog(const char* title, const char* filter) {
    char fn[1024] = "";
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFile = fn;
    ofn.nMaxFile = sizeof(fn);
    ofn.lpstrFilter = filter;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn) ? std::string(fn) : std::string();
}

static std::string OpenFolderDialog(const char* /*title*/) {
    std::string result;
    IFileOpenDialog* pfd = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_PPV_ARGS(&pfd)))) {
        DWORD opts = 0;
        pfd->GetOptions(&opts);
        pfd->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        if (SUCCEEDED(pfd->Show(g_hwnd))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(pfd->GetResult(&item))) {
                PWSTR w = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &w))) {
                    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
                    if (len > 1) {
                        result.resize(len - 1);
                        WideCharToMultiByte(CP_UTF8, 0, w, -1, &result[0], len, nullptr, nullptr);
                    }
                    CoTaskMemFree(w);
                }
                item->Release();
            }
        }
        pfd->Release();
    }
    return result;
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;
    switch (msg) {
    case WM_SIZE:
        if (g_pd3dDevice && wParam != SIZE_MINIMIZED) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam),
                                        DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) return 0;   // disable ALT app menu
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

// A clean dark theme: rounded corners, breathing room, a calm blue accent.
static void ApplyStyle() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 7.f; s.ChildRounding = 7.f; s.FrameRounding = 5.f;
    s.PopupRounding = 5.f;  s.GrabRounding = 4.f;  s.TabRounding = 5.f;
    s.ScrollbarRounding = 6.f;
    s.WindowPadding = ImVec2(12, 12); s.FramePadding = ImVec2(10, 6);
    s.ItemSpacing = ImVec2(9, 8); s.ItemInnerSpacing = ImVec2(7, 6);
    s.ScrollbarSize = 13.f; s.GrabMinSize = 11.f;
    s.WindowBorderSize = 0.f; s.FrameBorderSize = 0.f; s.ChildBorderSize = 0.f;

    ImVec4* c = s.Colors;
    c[ImGuiCol_Text]                 = ImVec4(0.92f, 0.93f, 0.95f, 1.00f);
    c[ImGuiCol_TextDisabled]         = ImVec4(0.48f, 0.51f, 0.56f, 1.00f);
    c[ImGuiCol_WindowBg]             = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
    c[ImGuiCol_ChildBg]              = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
    c[ImGuiCol_PopupBg]              = ImVec4(0.12f, 0.13f, 0.16f, 0.98f);
    c[ImGuiCol_FrameBg]              = ImVec4(0.18f, 0.20f, 0.24f, 1.00f);
    c[ImGuiCol_FrameBgHovered]       = ImVec4(0.23f, 0.26f, 0.31f, 1.00f);
    c[ImGuiCol_FrameBgActive]        = ImVec4(0.27f, 0.31f, 0.37f, 1.00f);
    c[ImGuiCol_TitleBg]              = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
    c[ImGuiCol_TitleBgActive]        = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
    c[ImGuiCol_Button]               = ImVec4(0.21f, 0.45f, 0.66f, 1.00f);
    c[ImGuiCol_ButtonHovered]        = ImVec4(0.26f, 0.55f, 0.80f, 1.00f);
    c[ImGuiCol_ButtonActive]         = ImVec4(0.18f, 0.39f, 0.59f, 1.00f);
    c[ImGuiCol_Header]               = ImVec4(0.21f, 0.45f, 0.66f, 0.50f);
    c[ImGuiCol_HeaderHovered]        = ImVec4(0.26f, 0.55f, 0.80f, 0.75f);
    c[ImGuiCol_HeaderActive]         = ImVec4(0.26f, 0.55f, 0.80f, 1.00f);
    c[ImGuiCol_SliderGrab]           = ImVec4(0.37f, 0.64f, 0.88f, 1.00f);
    c[ImGuiCol_SliderGrabActive]     = ImVec4(0.47f, 0.74f, 0.98f, 1.00f);
    c[ImGuiCol_CheckMark]            = ImVec4(0.47f, 0.74f, 0.98f, 1.00f);
    c[ImGuiCol_Separator]            = ImVec4(0.24f, 0.26f, 0.30f, 1.00f);
    c[ImGuiCol_SeparatorHovered]     = ImVec4(0.26f, 0.55f, 0.80f, 0.78f);
    c[ImGuiCol_Border]               = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_ScrollbarBg]          = ImVec4(0.10f, 0.11f, 0.13f, 0.00f);
    c[ImGuiCol_ScrollbarGrab]        = ImVec4(0.26f, 0.29f, 0.34f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.33f, 0.37f, 0.43f, 1.00f);
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);   // crisp, unscaled by Windows
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);   // for IFileOpenDialog
    WNDCLASSEX wc = { sizeof(wc), CS_CLASSDC, WndProc, 0, 0, hInstance,
                      nullptr, nullptr, nullptr, nullptr, _T("YOLOMasterEdge"), nullptr };
    RegisterClassEx(&wc);
    g_hwnd = CreateWindow(wc.lpszClassName, _T("YOLO-Master Edge"), WS_OVERLAPPEDWINDOW,
                          100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(g_hwnd)) {
        CleanupDeviceD3D();
        UnregisterClass(wc.lpszClassName, wc.hInstance);
        return 1;
    }
    ShowWindow(g_hwnd, SW_SHOWMAXIMIZED);   // fill the screen for a proper first impression
    UpdateWindow(g_hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;   // don't leave an imgui.ini next to the exe

    const float dpi = GetDpiForWindow(g_hwnd) / 96.0f;   // 1.0 at 96 DPI, 1.5 at 150%, ...

    // Real anti-aliased Segoe UI instead of the pixelated default bitmap font.
    ImFontConfig fc; fc.OversampleH = 3; fc.OversampleV = 2;
    if (GetFileAttributesA("C:\\Windows\\Fonts\\segoeui.ttf") != INVALID_FILE_ATTRIBUTES)
        io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 17.0f * dpi, &fc);

    ImGui::StyleColorsDark();
    ApplyStyle();
    ImGui::GetStyle().ScaleAllSizes(dpi);   // scale paddings/rounding to match the font

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    gui::Platform plat;
    plat.upload      = UploadTexture;
    plat.open_file   = OpenFileDialog;
    plat.open_folder = OpenFolderDialog;
    gui::App app;

    bool running = true;
    while (running) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) running = false;
        }
        if (!running) break;

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        app.frame(plat);

        ImGui::Render();
        const float clear[4] = { 0.08f, 0.09f, 0.10f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRTV, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRTV, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);   // vsync
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(g_hwnd);
    UnregisterClass(wc.lpszClassName, wc.hInstance);
    CoUninitialize();
    return 0;
}
