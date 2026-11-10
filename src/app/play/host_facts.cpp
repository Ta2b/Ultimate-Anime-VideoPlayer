#include "app/play/host_facts.h"

#include <algorithm>
#include <cmath>
#include <cwchar>
#include <filesystem>
#include <format>
#include <system_error>
#include <vector>

#include <SDL3/SDL_power.h>
#include <SDL3/SDL_properties.h>

#include "app/play/display_edid.h"
#include "app/play/i18n.h"
#include "app/play/pumped_task.h"
#include "app/play/system_peak.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace ajp::app::play {

#if defined(_WIN32)
namespace {

struct WindowsMode {
  int width = 0;
  int height = 0;
  double hz = 0.0;
};

// Le mode de l'écran de la fenêtre selon la configuration d'affichage de Windows (QueryDisplayConfig), pour
// l'écran où la fenêtre est vraiment (MonitorFromWindow, pas le premier écran) : la définition du bureau en
// pixels réels, que la mise à l'échelle et la virtualisation DPI ne touchent pas, et la fréquence exacte du
// signal en fraction (59,94 Hz = 60000/1001 ; 143,86 Hz), là où le mode de SDL vient de dmDisplayFrequency,
// un entier.
std::optional<WindowsMode> windows_mode(SDL_Window* window) {
  auto* hwnd = static_cast<HWND>(
      SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  if (hwnd == nullptr) return std::nullopt;
  MONITORINFOEXW monitor{};
  monitor.cbSize = sizeof(monitor);
  if (GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor) == 0) return std::nullopt;
  UINT32 paths = 0;
  UINT32 modes = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &paths, &modes) != ERROR_SUCCESS)
    return std::nullopt;
  std::vector<DISPLAYCONFIG_PATH_INFO> path_info(paths);
  std::vector<DISPLAYCONFIG_MODE_INFO> mode_info(modes);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &paths, path_info.data(), &modes, mode_info.data(),
                         nullptr) != ERROR_SUCCESS)
    return std::nullopt;
  for (UINT32 i = 0; i < paths; ++i) {
    const DISPLAYCONFIG_PATH_INFO& path = path_info[i];
    DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
    source.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId,
                     path.sourceInfo.id};
    if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS ||
        std::wcscmp(source.viewGdiDeviceName, monitor.szDevice) != 0)
      continue;
    WindowsMode out{};
    if (const UINT32 at = path.sourceInfo.modeInfoIdx;
        at < modes && mode_info[at].infoType == DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
      out.width = static_cast<int>(mode_info[at].sourceMode.width);
      out.height = static_cast<int>(mode_info[at].sourceMode.height);
    }
    // Le signal envoyé à l'écran (vSyncFreq), sinon la fréquence du chemin ; en VRR, la fréquence maximale.
    DISPLAYCONFIG_RATIONAL rate = path.targetInfo.refreshRate;
    if (const UINT32 at = path.targetInfo.modeInfoIdx;
        at < modes && mode_info[at].infoType == DISPLAYCONFIG_MODE_INFO_TYPE_TARGET) {
      const DISPLAYCONFIG_RATIONAL signal = mode_info[at].targetMode.targetVideoSignalInfo.vSyncFreq;
      if (signal.Numerator > 0 && signal.Denominator > 0) rate = signal;
    }
    if (rate.Numerator > 0 && rate.Denominator > 0)
      out.hz = static_cast<double>(rate.Numerator) / static_cast<double>(rate.Denominator);
    return out;
  }
  return std::nullopt;
}

}  // namespace
#endif

// Période de rafraîchissement de l'écran de la fenêtre, d'après Windows (exacte) ou SDL ; 0 si inconnue (la
// session la mesure alors sur les présentations).
std::int64_t display_period_us(SDL_Window* window) {
  if (const double hz = display_refresh_hz(window); hz > 1.0) return std::llround(1e6 / hz);
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  if (mode == nullptr || mode->refresh_rate <= 0.0F) return 0;
  return static_cast<std::int64_t>(1e6 / static_cast<double>(mode->refresh_rate));
}

double display_refresh_hz(SDL_Window* window) {
#if defined(_WIN32)
  if (const auto windows = windows_mode(window); windows && windows->hz > 1.0) return windows->hz;
#endif
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  if (mode == nullptr) return 0.0;
  if (mode->refresh_rate_numerator > 0 && mode->refresh_rate_denominator > 0)
    return static_cast<double>(mode->refresh_rate_numerator) /
           static_cast<double>(mode->refresh_rate_denominator);
  return mode->refresh_rate > 0.0F ? static_cast<double>(mode->refresh_rate) : 0.0;
}

namespace {

// La définition de l'écran en pixels réels. SDL3 donne le mode en points quand le bureau est mis à
// l'échelle (un écran 4K à 200 % : 1920 × 1080, densité 2) ; Windows peut aussi le virtualiser : son
// mode courant (EnumDisplaySettings) a le dernier mot quand il est plus grand.
OutputSize display_pixels(SDL_Window* window) {
#if defined(_WIN32)
  if (const auto windows = windows_mode(window); windows && windows->width > 0 && windows->height > 0)
    return {windows->width, windows->height};
#endif
  OutputSize out{};
  if (const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
      mode != nullptr && mode->w > 0 && mode->h > 0) {
    const float density = mode->pixel_density > 0.0F ? mode->pixel_density : 1.0F;
    out = {static_cast<int>(std::lround(static_cast<float>(mode->w) * density)),
           static_cast<int>(std::lround(static_cast<float>(mode->h) * density))};
  }
#if defined(_WIN32)
  auto* hwnd = static_cast<HWND>(
      SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  if (hwnd != nullptr) {
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    if (GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &info) != 0 &&
        EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode) != 0 &&
        static_cast<int>(mode.dmPelsHeight) > out.height)
      out = {static_cast<int>(mode.dmPelsWidth), static_cast<int>(mode.dmPelsHeight)};
  }
#endif
  return out;
}

}  // namespace

int display_width(SDL_Window* window) {
  return display_pixels(window).width;
}

int display_height(SDL_Window* window) {
  return display_pixels(window).height;
}

#if defined(_WIN32)
namespace {

// L'écran de la fenêtre (P3.57) : son moniteur et son nom GDI (« \\.\DISPLAY2 »), pour ne lire que son
// chemin dans la configuration d'affichage, pas le premier ni le plus lumineux.
struct WindowMonitor {
  HMONITOR handle = nullptr;
  MONITORINFOEXW info{};
};
std::optional<WindowMonitor> window_monitor(SDL_Window* window) {
  auto* hwnd = static_cast<HWND>(
      SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  if (hwnd == nullptr) return std::nullopt;
  WindowMonitor monitor{.handle = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)};
  monitor.info.cbSize = sizeof(monitor.info);
  if (GetMonitorInfoW(monitor.handle, &monitor.info) == 0) return std::nullopt;
  return monitor;
}

// Ce chemin mène-t-il à l'écran de ce nom GDI ? Sans nom, tout chemin convient.
bool path_on(const DISPLAYCONFIG_PATH_INFO& path, const wchar_t* device) {
  if (device == nullptr) return true;
  DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
  source.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId,
                   path.sourceInfo.id};
  return DisplayConfigGetDeviceInfo(&source.header) == ERROR_SUCCESS &&
         std::wcscmp(source.viewGdiDeviceName, device) == 0;
}

// Le niveau de blanc SDR réglé dans Windows (« luminosité du contenu SDR »), en nits, pour l'écran de ce nom
// GDI ; 203 si Windows ne le dit pas. SDL exprime la marge HDR relativement à ce blanc : la crête réelle est
// la marge × ce niveau.
double windows_sdr_white_nits(const wchar_t* device) {
  constexpr double kDefault = 203.0;
  UINT32 paths = 0;
  UINT32 modes = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &paths, &modes) != ERROR_SUCCESS) return kDefault;
  std::vector<DISPLAYCONFIG_PATH_INFO> path_info(paths);
  std::vector<DISPLAYCONFIG_MODE_INFO> mode_info(modes);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &paths, path_info.data(), &modes, mode_info.data(),
                         nullptr) != ERROR_SUCCESS)
    return kDefault;
  for (UINT32 i = 0; i < paths; ++i) {
    if (!path_on(path_info[i], device)) continue;
    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color{};
    color.header = {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO, sizeof(color),
                    path_info[i].targetInfo.adapterId, path_info[i].targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&color.header) != ERROR_SUCCESS || color.advancedColorEnabled == 0)
      continue;
    DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
    white.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, sizeof(white),
                    path_info[i].targetInfo.adapterId, path_info[i].targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&white.header) != ERROR_SUCCESS || white.SDRWhiteLevel == 0) continue;
    return static_cast<double>(white.SDRWhiteLevel) / 1000.0 * 80.0;  // 1000 = 80 nits
  }
  return kDefault;
}

// Windows a-t-il le HDR actif (« Utiliser le HDR ») sur l'écran de ce nom GDI ? SDL ne le voit pas toujours
// avec Vulkan : il rend une marge de 1,00 sur un écran que Windows dit en HDR.
bool windows_hdr_active(const wchar_t* device) {
  UINT32 paths = 0;
  UINT32 modes = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &paths, &modes) != ERROR_SUCCESS) return false;
  std::vector<DISPLAYCONFIG_PATH_INFO> path_info(paths);
  std::vector<DISPLAYCONFIG_MODE_INFO> mode_info(modes);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &paths, path_info.data(), &modes, mode_info.data(),
                         nullptr) != ERROR_SUCCESS)
    return false;
  for (UINT32 i = 0; i < paths; ++i) {
    if (!path_on(path_info[i], device)) continue;
    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color{};
    color.header = {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO, sizeof(color),
                    path_info[i].targetInfo.adapterId, path_info[i].targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&color.header) == ERROR_SUCCESS && color.advancedColorEnabled != 0)
      return true;
  }
  return false;
}

}  // namespace
#endif

// Crête vue par le contenu sur l'écran de la fenêtre : la marge HDR de SDL3 (relative au blanc de
// référence du compositeur) × 203 nits ; rien si l'écran n'est pas en HDR (display.md).
std::optional<double> content_peak_nits(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  const float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F);
  const bool sdl_hdr =
      SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false) && headroom > 1.0F;
#if defined(_WIN32)
  const std::optional<WindowMonitor> monitor = window_monitor(window);
  const wchar_t* device = monitor ? monitor->info.szDevice : nullptr;
  if (sdl_hdr) {
    // Le pic que Windows a calibré (application « Étalonnage HDR ») : MaxLuminance de la sortie DXGI de
    // l'écran de la fenêtre.
    // Sous Windows le connecteur ne sert pas : demandé même sans EDID au registre (dock, écran virtuel,
    // A11-07).
    if (const auto calibrated = system_peak_nits({}, headroom, monitor ? monitor->handle : nullptr))
      return calibrated;
    // La marge de SDL dépend du curseur « luminosité du contenu SDR » de Windows (écran à 800 nits, blanc SDR
    // à 400 : marge 2) : la crête que l'EDID annonce la borne par le bas, sans quoi le gris moyen (plafonné à
    // la moitié de la crête) n'a plus d'effet.
    double peak = static_cast<double>(headroom) * windows_sdr_white_nits(device);
    if (const auto display = read_display(display_name(window)); display && display->peak_nits)
      peak = std::max(peak, *display->peak_nits);
    return peak;
  }
  // SDL ne voit pas le HDR (activé=false ou marge de 1,00 avec Vulkan) mais Windows le dit actif : la crête
  // est celle que l'EDID de l'écran annonce, sinon 600 nits (un écran HDR10 courant, prudent : trop haut
  // éclaire trop) ; « Pic auto » du panneau HDR la reprend.
  constexpr double kUnknownPeakNits = 600.0;
  if (!windows_hdr_active(device)) return std::nullopt;
  if (const auto display = read_display(display_name(window)); display && display->peak_nits)
    return *display->peak_nits;
  return kUnknownPeakNits;
#else
  if (!sdl_hdr) return std::nullopt;
  // La marge de SDL est relative au blanc de référence du compositeur (KDE : la « luminosité SDR »), pas à
  // 203 nits : le pic que le système a calibré se lit chez lui (system_peak.h).
  if (const auto display = read_display(display_name(window)))
    if (const auto calibrated = system_peak_nits(display->connector, headroom)) return calibrated;
  return static_cast<double>(headroom) * 203.0;
#endif
}

static_assert(limits::kTrueHdrPeak.lo == static_cast<int>(kMinHdrPeakNits));  // un seul seuil (P3.66)

bool system_hdr(std::optional<double> peak_nits) {
  return peak_nits && std::isfinite(*peak_nits) && *peak_nits >= kMinHdrPeakNits;
}

int auto_peak_nits(std::optional<double> detected) {
  constexpr double kUnknown = 1000.0;
  constexpr double kLow = 200.0;
  constexpr double kHigh = 2000.0;
  if (!detected || !std::isfinite(*detected) || *detected <= 0.0) return static_cast<int>(kUnknown);
  return static_cast<int>(std::lround(std::clamp(*detected, kLow, kHigh)));
}

std::string describe_hdr(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  std::string text =
      std::format("HDR de l'écran selon SDL : activé={} marge={:.2f} blanc SDR={:.2f}",  // i18n-ok : journal
                  SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false),
                  SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F),
                  SDL_GetFloatProperty(props, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 0.0F));
#if defined(_WIN32)
  UINT32 paths = 0;
  UINT32 modes = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &paths, &modes) == ERROR_SUCCESS) {
    std::vector<DISPLAYCONFIG_PATH_INFO> path_info(paths);
    std::vector<DISPLAYCONFIG_MODE_INFO> mode_info(modes);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &paths, path_info.data(), &modes, mode_info.data(),
                           nullptr) == ERROR_SUCCESS) {
      for (UINT32 i = 0; i < paths; ++i) {
        DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color{};
        color.header = {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO, sizeof(color),
                        path_info[i].targetInfo.adapterId, path_info[i].targetInfo.id};
        if (DisplayConfigGetDeviceInfo(&color.header) != ERROR_SUCCESS) continue;
        text += std::format(" ; Windows écran {} : HDR pris en charge={} activé={}", i,  // i18n-ok : journal
                            color.advancedColorSupported != 0, color.advancedColorEnabled != 0);
      }
    }
  }
#endif
  return text;
}

bool tensorrt_backend_present(const EngineCatalog& catalog, const PlayerPaths& paths) {
  // Le paquet téléchargé s'installe sous les données, pas à côté du lecteur : les deux se cherchent.
  std::vector<std::filesystem::path> candidates = installed_backends(catalog);
  candidates.insert(candidates.end(), paths.backends.begin(), paths.backends.end());
  std::error_code error;
  return std::ranges::any_of(candidates, [&](const std::filesystem::path& backend) {
    return backend.filename().string().contains("tensorrt") && std::filesystem::exists(backend, error);
  });
}

// Ce qui est sur ce poste pour l'écran de bienvenue : les backends TensorRT et Core ML.
EnginesHere engines_here(const PlayOptions& /*options*/, const PlayerPaths& paths,
                         const EngineCatalog& catalog, std::string_view gpu_name) {
  EnginesHere here;
  // L'architecture du GPU ne se lit (nvidia-smi) que s'il y a des fichiers tiers à choisir.
  const std::string arch = catalog.vendor_files.empty()
                               ? std::string()
                               : effective_cuda_arch(pumped_cuda_arch(gpu_name), gpu_name);
  for (const core::EnginePackage& package : catalog.packages) {
    here.downloads.push_back(EngineDownload{
        .id = package.engine,
        .installed = installed(catalog, package),
        .bytes = download_bytes(catalog, package, arch),
        .needed =
            install_space(catalog, package, arch, directory_bytes(core::engine_dir(catalog.layout, package)))
                .peak,
        .available = free_bytes(catalog.layout.engines),
        .license = package.license});
  }
  std::error_code error;
  for (const std::filesystem::path& backend : paths.backends) {
    if (!std::filesystem::exists(backend, error)) continue;
    const std::string file = backend.filename().string();
    here.tensorrt |= file.contains("tensorrt");
    here.coreml |= file.contains("coreml");
  }
  return here;
}

// Le nom de l'écran de la fenêtre selon SDL, pour choisir son EDID.
std::string display_name(SDL_Window* window) {
  const char* name = SDL_GetDisplayName(SDL_GetDisplayForWindow(window));
  return name != nullptr ? std::string(name) : std::string();
}

GpuFacts gpu_facts(const PlaceboScreen& screen) {
  const PlaceboScreen::DeviceFacts facts = screen.device_facts();
  const GpuVendor vendor = vendor_from_pci(facts.pci_vendor);
  return GpuFacts{
      .name = facts.name,
      .driver = facts.driver,
      .vendor = vendor,
      .vram_bytes = facts.vram_bytes,
      .host_visible_vram_bytes = facts.host_visible_vram_bytes,
      .tensor_cores = vendor == GpuVendor::Nvidia && facts.cooperative_matrix,
      .wmma = vendor == GpuVendor::Amd && facts.cooperative_matrix && amd_matrix_cores(facts.name)};
}

bool on_battery() {
  return SDL_GetPowerInfo(nullptr, nullptr) == SDL_POWERSTATE_ON_BATTERY;
}

}  // namespace ajp::app::play
