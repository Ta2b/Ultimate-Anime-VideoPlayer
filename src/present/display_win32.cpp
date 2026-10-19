// Compilé sous Windows seulement (src/present/CMakeLists.txt).
#ifdef _WIN32

// windows.h d'abord, avec ses réglages : les en-têtes DXGI et DisplayConfig en dépendent.
// clang-format off
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_6.h>
// clang-format on

#include "present/display_win32.h"

#include <cwchar>
#include <optional>
#include <vector>

#include <SDL3/SDL_properties.h>

namespace ajp::present {

namespace {

// Blanc SDR de DisplayConfig : en millièmes de 80 nits (le blanc de scRGB).
constexpr double kSdrWhiteUnitNits = 80.0 / 1000.0;

// Libère une interface COM en fin de portée.
template <class T>
struct ComRef {
  T* ptr = nullptr;
  ComRef() = default;
  ComRef(const ComRef&) = delete;
  ComRef& operator=(const ComRef&) = delete;
  ~ComRef() {
    if (ptr != nullptr) ptr->Release();
  }
};

HMONITOR monitor_of(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  auto* hwnd = static_cast<HWND>(SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  if (hwnd == nullptr) return nullptr;
  return MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
}

std::optional<DXGI_OUTPUT_DESC1> output_desc(HMONITOR monitor) {
  ComRef<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.ptr)))) return std::nullopt;
  for (UINT a = 0;; ++a) {
    ComRef<IDXGIAdapter1> adapter;
    if (factory.ptr->EnumAdapters1(a, &adapter.ptr) == DXGI_ERROR_NOT_FOUND) break;
    for (UINT o = 0;; ++o) {
      ComRef<IDXGIOutput> output;
      if (adapter.ptr->EnumOutputs(o, &output.ptr) == DXGI_ERROR_NOT_FOUND) break;
      ComRef<IDXGIOutput6> output6;
      if (FAILED(output.ptr->QueryInterface(IID_PPV_ARGS(&output6.ptr)))) continue;
      DXGI_OUTPUT_DESC1 desc{};
      if (SUCCEEDED(output6.ptr->GetDesc1(&desc)) && desc.Monitor == monitor) return desc;
    }
  }
  return std::nullopt;
}

std::optional<double> sdr_white_nits(HMONITOR monitor) {
  MONITORINFOEXW info{};
  info.cbSize = sizeof(info);
  if (!GetMonitorInfoW(monitor, &info)) return std::nullopt;
  UINT32 path_count = 0;
  UINT32 mode_count = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS) {
    return std::nullopt;
  }
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(),
                         nullptr) != ERROR_SUCCESS) {
    return std::nullopt;
  }
  for (UINT32 i = 0; i < path_count; ++i) {
    const DISPLAYCONFIG_PATH_INFO& path = paths[i];
    DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
    source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
    source.header.size = sizeof(source);
    source.header.adapterId = path.sourceInfo.adapterId;
    source.header.id = path.sourceInfo.id;
    if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS) continue;
    if (std::wcscmp(source.viewGdiDeviceName, info.szDevice) != 0) continue;
    DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
    white.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
    white.header.size = sizeof(white);
    white.header.adapterId = path.targetInfo.adapterId;
    white.header.id = path.targetInfo.id;
    if (DisplayConfigGetDeviceInfo(&white.header) != ERROR_SUCCESS) return std::nullopt;
    return static_cast<double>(white.SDRWhiteLevel) * kSdrWhiteUnitNits;
  }
  return std::nullopt;
}

}  // namespace

void read_dxgi(SDL_Window* window, SystemDisplay& out) {
  const HMONITOR monitor = monitor_of(window);
  if (monitor == nullptr) {
    out.unknown.emplace_back("DXGI : moniteur de la fenêtre introuvable");
    return;
  }
  if (const auto desc = output_desc(monitor)) {
    out.hdr = desc->ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    if (desc->MaxLuminance > 0.0F) out.peak_nits = desc->MaxLuminance;
    if (desc->MinLuminance >= 0.0F && desc->MinLuminance < desc->MaxLuminance)
      out.min_nits = desc->MinLuminance;
  } else {
    out.unknown.emplace_back("DXGI_OUTPUT_DESC1 : sortie de la fenêtre introuvable");
  }
  if (const auto white = sdr_white_nits(monitor)) {
    out.sdr_white_nits = *white;
  } else {
    out.unknown.emplace_back("blanc SDR : DisplayConfigGetDeviceInfo(GET_SDR_WHITE_LEVEL) sans réponse");
  }
}

}  // namespace ajp::present

#endif  // _WIN32
