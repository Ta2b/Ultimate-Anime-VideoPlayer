// Le VRR armé sur l'écran de la fenêtre (host_facts.h). Sous Windows, aucune API publique de DXGI ne dit si
// G-SYNC ou FreeSync est engagé : NVAPI le dit sur une carte NVIDIA (nvapi64.dll, livrée avec le pilote,
// chargée à l'exécution par nvapi_QueryInterface, sans en-tête ni fichier de NVIDIA dans nos sources).
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

#include <SDL3/SDL_properties.h>

#include "app/play/host_facts.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace ajp::app::play {

#if defined(_WIN32)
namespace {

// Les identifiants publics de nvapi_QueryInterface (nvapi_interface.h du SDK).
constexpr std::uint32_t kNvInitialize = 0x0150E828U;
constexpr std::uint32_t kNvDisplayIdByName = 0xAE457190U;
constexpr std::uint32_t kNvGetVrrInfo = 0xDF8FDA57U;

// NV_GET_VRR_INFO_V1 : version, puis les drapeaux (bit 0 bIsVRREnabled, bit 4 bIsDisplayInVRRMode), puis
// quatre mots réservés. La version vaut sizeof | (1 << 16) (MAKE_NVAPI_VERSION).
struct NvVrrInfo {
  std::uint32_t version = sizeof(NvVrrInfo) | (1U << 16U);
  std::uint32_t flags = 0;
  std::uint32_t reserved[4] = {};
};
constexpr std::uint32_t kNvVrrEnabled = 1U << 0U;
constexpr std::uint32_t kNvInVrrMode = 1U << 4U;

using QueryInterface = void* (*)(std::uint32_t);
using Initialize = int (*)();
using DisplayIdByName = int (*)(const char*, std::uint32_t*);
using GetVrrInfo = int (*)(std::uint32_t, NvVrrInfo*);

struct Nvapi {
  DisplayIdByName display_id = nullptr;
  GetVrrInfo vrr_info = nullptr;
};

// Chargée une fois par processus ; rien sans pilote NVIDIA (carte AMD ou Intel) ou si l'initialisation
// échoue.
const Nvapi* nvapi() {
  static const Nvapi* loaded = []() -> const Nvapi* {
    // System32 seulement : une DLL du même nom ailleurs n'est jamais chargée.
    HMODULE module = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module == nullptr) return nullptr;
    const auto query = reinterpret_cast<QueryInterface>(
        reinterpret_cast<void*>(GetProcAddress(module, "nvapi_QueryInterface")));
    if (query == nullptr) return nullptr;
    const auto init = reinterpret_cast<Initialize>(query(kNvInitialize));
    static Nvapi api{.display_id = reinterpret_cast<DisplayIdByName>(query(kNvDisplayIdByName)),
                     .vrr_info = reinterpret_cast<GetVrrInfo>(query(kNvGetVrrInfo))};
    if (init == nullptr || init() != 0 || api.display_id == nullptr || api.vrr_info == nullptr)
      return nullptr;
    return &api;
  }();
  return loaded;
}

// Le nom GDI de l'écran de la fenêtre (« \\.\DISPLAY1 »), celui que NVAPI connaît.
std::string gdi_name(SDL_Window* window) {
  auto* hwnd = static_cast<HWND>(
      SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  if (hwnd == nullptr) return {};
  MONITORINFOEXA monitor{};
  monitor.cbSize = sizeof(monitor);
  if (GetMonitorInfoA(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor) == 0) return {};
  return monitor.szDevice;
}

// G-SYNC activé pour cet écran et engagé en ce moment ; rien si NVAPI ne connaît pas l'écran.
std::optional<bool> nvapi_vrr_armed(SDL_Window* window) {
  const Nvapi* api = nvapi();
  if (api == nullptr) return std::nullopt;
  const std::string name = gdi_name(window);
  std::uint32_t id = 0;
  if (name.empty() || api->display_id(name.c_str(), &id) != 0) return std::nullopt;
  NvVrrInfo info;
  if (api->vrr_info(id, &info) != 0) return std::nullopt;
  return (info.flags & kNvVrrEnabled) != 0 && (info.flags & kNvInVrrMode) != 0;
}

}  // namespace
#endif

std::optional<DisplayFacts> window_display(SDL_Window* window) {
  std::optional<DisplayFacts> display = read_display(display_name(window));
#if defined(_WIN32)
  if (display && !display->vrr_armed) display->vrr_armed = nvapi_vrr_armed(window);
#endif
  return display;
}

std::optional<bool> window_vrr_armed([[maybe_unused]] SDL_Window* window, std::string_view connector) {
#if defined(_WIN32)
  // Un appel au pilote à chaque tour de boucle serait de trop : la lecture est gardée 250 ms.
  constexpr auto kReread = std::chrono::milliseconds(250);
  static std::mutex guard;
  static std::chrono::steady_clock::time_point read_at;
  static std::optional<bool> last;
  const std::scoped_lock lock(guard);
  if (const auto now = std::chrono::steady_clock::now(); now - read_at >= kReread) {
    read_at = now;
    last = nvapi_vrr_armed(window);
  }
  (void)connector;
  return last;
#else
  return vrr_armed_now(connector);
#endif
}

}  // namespace ajp::app::play
