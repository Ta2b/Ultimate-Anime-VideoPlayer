#include "app/play/host_facts.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <system_error>
#include <vector>

#include <SDL3/SDL_properties.h>

#include "app/play/i18n.h"

#if defined(_WIN32)
#include <dxgi1_6.h>
#include <windows.h>
#endif

#include "vendor_hdr/sdk_store.h"

namespace ajp::app::play {

// Période de rafraîchissement de l'écran de la fenêtre, d'après SDL ; 0 si inconnue (la session
// la mesure alors sur les présentations).
std::int64_t display_period_us(SDL_Window* window) {
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  if (mode == nullptr || mode->refresh_rate <= 0.0F) return 0;
  return static_cast<std::int64_t>(1e6 / static_cast<double>(mode->refresh_rate));
}

double display_refresh_hz(SDL_Window* window) {
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  if (mode == nullptr) return 0.0;
  if (mode->refresh_rate_numerator > 0 && mode->refresh_rate_denominator > 0)
    return static_cast<double>(mode->refresh_rate_numerator) /
           static_cast<double>(mode->refresh_rate_denominator);
  return mode->refresh_rate > 0.0F ? static_cast<double>(mode->refresh_rate) : 0.0;
}

int display_height(SDL_Window* window) {
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  return mode != nullptr && mode->h > 0 ? mode->h : 0;
}

#if defined(_WIN32)
namespace {

// Le niveau de blanc SDR réglé dans Windows (« luminosité du contenu SDR »), en nits, pour le
// premier écran en HDR ; 203 si Windows ne le dit pas. SDL exprime la marge HDR relativement à ce
// blanc : la crête réelle est la marge × ce niveau.
double windows_sdr_white_nits() {
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

// L'écran de la fenêtre selon DXGI : en HDR quand son espace colorimétrique est PQ BT.2020, avec la luminance
// maximale qu'il annonce (0 si inconnue). SDL, lui, ne voit pas de marge HDR quand « luminosité du contenu
// SDR » de Windows est proche de la crête de l'écran, alors que Windows est bien en HDR.
struct DxgiHdr {
  bool found = false;
  bool hdr = false;
  double max_nits = 0.0;
};

DxgiHdr dxgi_hdr(SDL_Window* window) {
  DxgiHdr result;
  auto* hwnd = static_cast<HWND>(
      SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  if (hwnd == nullptr) return result;
  const HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return result;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
    IDXGIOutput* output = nullptr;
    for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
      IDXGIOutput6* output6 = nullptr;
      if (SUCCEEDED(output->QueryInterface(__uuidof(IDXGIOutput6), reinterpret_cast<void**>(&output6)))) {
        DXGI_OUTPUT_DESC1 desc{};
        if (SUCCEEDED(output6->GetDesc1(&desc)) && desc.Monitor == monitor) {
          result.found = true;
          result.hdr = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
          result.max_nits = static_cast<double>(desc.MaxLuminance);
        }
        output6->Release();
      }
      output->Release();
    }
    adapter->Release();
  }
  factory->Release();
  return result;
}

}  // namespace
#endif

// Crête vue par le contenu sur l'écran de la fenêtre : la marge HDR de SDL3 (relative au blanc de
// référence du compositeur) × 203 nits ; rien si l'écran n'est pas en HDR (display.md).
std::optional<double> content_peak_nits(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  const bool sdl_hdr = SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
  const float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F);
#if defined(_WIN32)
  if (sdl_hdr && headroom > 1.0F) return static_cast<double>(headroom) * windows_sdr_white_nits();
  // SDL dit SDR : Windows peut pourtant être en HDR (blanc SDR réglé près de la crête, donc marge de 1). DXGI
  // tranche.
  const DxgiHdr dxgi = dxgi_hdr(window);
  if (!dxgi.hdr) return std::nullopt;
  return std::max(dxgi.max_nits > 0.0 ? dxgi.max_nits : 1000.0, windows_sdr_white_nits());
#else
  if (!sdl_hdr || !(headroom > 1.0F)) return std::nullopt;
  return static_cast<double>(headroom) * 203.0;
#endif
}

std::string describe_hdr(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  std::string text =
      std::format("HDR de l'écran selon SDL : activé={} marge={:.2f} blanc SDR={:.2f}",  // i18n-ok : journal
                  SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false),
                  SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F),
                  SDL_GetFloatProperty(props, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 0.0F));
#if defined(_WIN32)
  const DxgiHdr dxgi = dxgi_hdr(window);
  constexpr std::string_view kDxgi =
      " ; DXGI : écran trouvé={} PQ={} luminance max={:.0f}";  // i18n-ok : journal
  text += std::format(kDxgi, dxgi.found, dxgi.hdr, dxgi.max_nits);
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

bool maxine_available(const PlayOptions& options, const PlayerPaths& paths) {
  const vendor_hdr::SdkStore store(paths.engines.parent_path() / "sdk", vendor_hdr::SdkKind::MaxineVfx);
  if (store.status().installed) return true;
  std::error_code error;
  return std::filesystem::exists(options.true_hdr_sdk, error);
}

bool tensorrt_backend_present(const PlayerPaths& paths) {
  std::error_code error;
  return std::ranges::any_of(paths.backends, [&](const std::filesystem::path& backend) {
    return backend.filename().string().contains("tensorrt") && std::filesystem::exists(backend, error);
  });
}

// Ce qui est sur ce poste pour l'écran de bienvenue : les backends TensorRT, MIGraphX et OpenVINO,
// le SDK Maxine importé ou à importer.
EnginesHere engines_here(const PlayOptions& options, const PlayerPaths& paths, const EngineCatalog& catalog,
                         std::string_view gpu_name) {
  EnginesHere here;
  // L'architecture du GPU ne se lit (nvidia-smi) que s'il y a des fichiers tiers à choisir.
  const std::string arch =
      catalog.vendor_files.empty() ? std::string() : effective_cuda_arch(detect_cuda_arch(), gpu_name);
  for (const core::EnginePackage& package : catalog.packages) {
    here.downloads.push_back(EngineDownload{
        .id = package.engine,
        .installed = installed(catalog, package),
        .bytes = download_bytes(catalog, package, arch),
        .needed =
            install_space(catalog, package, arch, directory_bytes(core::engine_dir(catalog.layout, package)))
                .peak,
        .available = free_bytes(catalog.layout.engines),
        .license = package.license,
        .missing = missing_prerequisite(package.engine).value_or("")});
  }
  std::error_code error;
  for (const std::filesystem::path& backend : paths.backends) {
    if (!std::filesystem::exists(backend, error)) continue;
    const std::string file = backend.filename().string();
    here.tensorrt |= file.contains("tensorrt");
    here.migraphx |= file.contains("migraphx");
    here.openvino |= file.contains("openvino");
    here.coreml |= file.contains("coreml");
  }
  vendor_hdr::SdkStore store(paths.engines.parent_path() / "sdk", vendor_hdr::SdkKind::MaxineVfx);
  if (store.status().installed) {
    here.maxine = true;
    here.maxine_status = i18n::tr("imported");
  } else if (std::filesystem::exists(options.true_hdr_sdk, error)) {
    here.maxine = true;
    here.maxine_status = i18n::tr_format("to import from {} (once)", options.true_hdr_sdk.string());
  } else {
    here.maxine_status = i18n::tr_format("SDK missing ({})", options.true_hdr_sdk.string());
  }
  if (!here.maxine && catalog.maxine) here.maxine_license = catalog.maxine->license;
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

}  // namespace ajp::app::play
