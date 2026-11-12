// Le VRR armé sur l'écran de la fenêtre (host_facts.h). Sous Windows, aucune API publique de DXGI ne dit si
// G-SYNC ou FreeSync est engagé : NVAPI le dit sur une carte NVIDIA (nvapi64.dll, livrée avec le pilote,
// chargée à l'exécution par nvapi_QueryInterface, sans en-tête ni fichier de NVIDIA dans nos sources).
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

#include <SDL3/SDL_properties.h>

#include "app/play/host_facts.h"
#include "app/play/vrr_watch.h"
#include "core/log.h"

#if defined(_WIN32)
#include <functional>
#include <map>
#include <utility>
#include <windows.h>
#endif

namespace ajp::app::play {

#if defined(_WIN32)
namespace {

// Les identifiants publics de nvapi_QueryInterface (nvapi_interface.h du SDK).
constexpr std::uint32_t kNvInitialize = 0x0150E828U;
constexpr std::uint32_t kNvDisplayIdByName = 0xAE457190U;
constexpr std::uint32_t kNvGetVrrInfo = 0xDF8FDA57U;

// NV_GET_VRR_INFO_V1 : version, puis les drapeaux (bit 0 bIsVRREnabled, bit 4 bIsDisplayInVRRMode,
// nv_vrr_armed), puis quatre mots réservés. La version vaut sizeof | (1 << 16) (MAKE_NVAPI_VERSION).
struct NvVrrInfo {
  std::uint32_t version = sizeof(NvVrrInfo) | (1U << 16U);
  std::uint32_t flags = 0;
  std::uint32_t reserved[4] = {};
};

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

// Ce que NVAPI dit de l'écran de la fenêtre ; rien s'il ne le connaît pas.
struct NvVrr {
  bool enabled = false;  // G-SYNC permis sur cet écran : capable (B11-03)
  bool armed = false;    // engagé en ce moment (nv_vrr_armed)
};
// Ce que NVAPI a déjà dit de chaque écran, par nom GDI (C37-06) : le bit 4 vu posé, et les derniers drapeaux
// (pour ne journaliser qu'un changement). Un écran G-SYNC n'impose plus sa règle à un écran fixe.
struct NvSeen {
  bool in_mode_seen = false;
  std::uint32_t last_flags = UINT32_MAX;
};

std::optional<NvVrr> nvapi_vrr(SDL_Window* window, const std::string& name) {
  const Nvapi* api = nvapi();
  if (api == nullptr) return std::nullopt;
  std::uint32_t id = 0;
  if (name.empty() || api->display_id(name.c_str(), &id) != 0) return std::nullopt;
  NvVrrInfo info;
  if (api->vrr_info(id, &info) != 0) return std::nullopt;
  static std::mutex guard;
  static std::map<std::string, NvSeen, std::less<>> seen;
  bool in_mode_seen = false;
  {
    const std::scoped_lock lock(guard);
    NvSeen& here = seen[name];
    if ((info.flags & kNvVrrInMode) != 0) here.in_mode_seen = true;
    in_mode_seen = here.in_mode_seen;
    // B11-02 : la structure est écrite à la main ; les drapeaux bruts vont au journal à chaque changement,
    // pour vérifier sur l'écran réel que le bit 4 suit bien l'engagement de G-SYNC.
    if (std::exchange(here.last_flags, info.flags) != info.flags)
      log_channel("app")->info("NVAPI VRR ({}) : drapeaux {:#x}, réservés {:#x} {:#x} {:#x} {:#x}", name,
                               info.flags, info.reserved[0], info.reserved[1], info.reserved[2],
                               info.reserved[3]);
  }
  // C34-03 : G-SYNC permis ne vaut armé qu'en plein écran, tant que le bit 4 n'a jamais été vu.
  const bool fullscreen = (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
  return NvVrr{.enabled = (info.flags & kNvVrrEnabled) != 0,
               .armed = nv_vrr_armed(info.flags, in_mode_seen, fullscreen)};
}

std::optional<NvVrr> nvapi_vrr(SDL_Window* window) {
  return nvapi_vrr(window, gdi_name(window));
}

// Le chemin d'interface du moniteur qui porte la fenêtre (« \\?\DISPLAY#GSM5B08#…#{e6f07b5f-…} »), que
// pick_display rapproche de l'identifiant d'instance de SetupAPI (B34-02) ; vide s'il ne se lit pas.
std::string monitor_interface(SDL_Window* window) {
  const std::string adapter = gdi_name(window);
  if (adapter.empty()) return {};
  DISPLAY_DEVICEA device{};
  device.cb = sizeof(device);
  for (DWORD i = 0; EnumDisplayDevicesA(adapter.c_str(), i, &device, EDD_GET_DEVICE_INTERFACE_NAME) != FALSE;
       ++i) {
    if ((device.StateFlags & DISPLAY_DEVICE_ACTIVE) != 0 && device.DeviceID[0] != '\0')
      return device.DeviceID;
    device = DISPLAY_DEVICEA{};
    device.cb = sizeof(device);
  }
  return {};
}

}  // namespace
#endif

std::optional<DisplayFacts> window_display(SDL_Window* window) {
  return window_display(window, display_name(window));
}

std::optional<DisplayFacts> window_display(SDL_Window* window, std::string_view sdl_name) {
#if defined(_WIN32)
  std::optional<DisplayFacts> display = read_display(sdl_name, monitor_interface(window));
  if (display) {
    if (const auto nv = nvapi_vrr(window)) {
      if (!display->vrr_armed) display->vrr_armed = nv->armed;
      // B11-03 : un écran G-SYNC à module n'annonce souvent aucune plage dans son EDID.
      if (nv->enabled) display->vrr = true;
    }
  }
#else
  static_cast<void>(window);
  std::optional<DisplayFacts> display = read_display(sdl_name);
#endif
  // Armé suppose capable, quelle que soit la source (B11-03).
  if (display && display->vrr_armed.value_or(false)) display->vrr = true;
  return display;
}

std::optional<bool> window_vrr_armed([[maybe_unused]] SDL_Window* window, std::string_view connector) {
#if defined(_WIN32)
  // Un appel au pilote à chaque tour de boucle serait de trop : la lecture est gardée 250 ms, pour cet écran
  // seulement (C37-06) : la fenêtre passée sur un autre écran le relit tout de suite.
  constexpr auto kReread = std::chrono::milliseconds(250);
  static std::mutex guard;
  static std::chrono::steady_clock::time_point read_at;
  static std::string read_for;
  static std::optional<bool> last;
  const std::string name = gdi_name(window);
  const std::scoped_lock lock(guard);
  if (const auto now = std::chrono::steady_clock::now(); now - read_at >= kReread || name != read_for) {
    read_at = now;
    read_for = name;
    const auto nv = nvapi_vrr(window, name);
    last = nv ? std::optional<bool>(nv->armed) : std::nullopt;
  }
  (void)connector;
  return last;
#else
  return vrr_armed_now(connector);
#endif
}

}  // namespace ajp::app::play
