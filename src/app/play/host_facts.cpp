#include "app/play/host_facts.h"

#include <filesystem>
#include <format>
#include <system_error>

#include <SDL3/SDL_properties.h>

#include "vendor_hdr/sdk_store.h"

namespace ajp::app::play {

// Période de rafraîchissement de l'écran de la fenêtre, d'après SDL ; 0 si inconnue (la session
// la mesure alors sur les présentations).
std::int64_t display_period_us(SDL_Window* window) {
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  if (mode == nullptr || mode->refresh_rate <= 0.0F) return 0;
  return static_cast<std::int64_t>(1e6 / static_cast<double>(mode->refresh_rate));
}

// Crête vue par le contenu sur l'écran de la fenêtre : la marge HDR de SDL3 (relative au blanc de
// référence du compositeur) × 203 nits ; rien si l'écran n'est pas en HDR (display.md).
std::optional<double> content_peak_nits(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  if (!SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false)) return std::nullopt;
  const float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F);
  if (!(headroom > 1.0F)) return std::nullopt;
  return static_cast<double>(headroom) * 203.0;
}

// Ce qui est sur ce poste pour l'écran de bienvenue : les backends TensorRT, MIGraphX et OpenVINO,
// le SDK Maxine importé ou à importer.
EnginesHere engines_here(const PlayOptions& options, const PlayerPaths& paths, const EngineCatalog& catalog) {
  EnginesHere here;
  for (const core::EnginePackage& package : catalog.packages) {
    here.downloads.push_back(EngineDownload{.id = package.engine,
                                            .installed = installed(catalog, package),
                                            .bytes = package.size,
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
    here.maxine_status = "importé";
  } else if (std::filesystem::exists(options.true_hdr_sdk, error)) {
    here.maxine = true;
    here.maxine_status = std::format("à importer depuis {} (une fois)", options.true_hdr_sdk.string());
  } else {
    here.maxine_status = std::format("SDK absent ({})", options.true_hdr_sdk.string());
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
      .vendor = vendor,
      .vram_bytes = facts.vram_bytes,
      .tensor_cores = vendor == GpuVendor::Nvidia && facts.cooperative_matrix,
      .wmma = vendor == GpuVendor::Amd && facts.cooperative_matrix && amd_matrix_cores(facts.name)};
}

}  // namespace ajp::app::play
