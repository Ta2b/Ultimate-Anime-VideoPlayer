#include "app/play/display_gpu.h"

#include <cstring>
#include <format>

#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

#include "app/play/gpu_telemetry.h"
#include "app/play/host_facts.h"
#include "core/log.h"

#if defined(_WIN32)
#include <dxgi1_6.h>
#include <windows.h>
#elif defined(__linux__)
#include <charconv>
#include <filesystem>
#include <fstream>
#endif

namespace ajp::app::play {

std::string drm_card_of(std::string_view connector) {
  if (!connector.starts_with("card")) return {};
  const std::size_t dash = connector.find('-');
  if (dash == std::string_view::npos || dash == 4) return {};
  for (std::size_t i = 4; i < dash; ++i)
    if (connector[i] < '0' || connector[i] > '9') return {};
  return std::string(connector.substr(0, dash));
}

std::string screen_card_line(const ScreenCard& card, std::string_view render_name) {
  if (card.renders) return std::format("écran sur la carte du rendu ({})", render_name);
  return std::format(
      "écran sur la carte « {} », rendu sur « {} » : chaque image est recopiée vers l'écran (pas de G-SYNC, "
      "retard et bande passante en plus) ; le mode « dGPU seul » du BIOS ou de l'outil du constructeur "
      "l'évite",
      card.name, render_name);
}

#if defined(__linux__)
namespace {

// « 0x10de\n » : 0x10de ; 0 s'il ne se lit pas.
std::uint32_t read_hex(const std::filesystem::path& file) {
  std::ifstream in(file);
  std::string text;
  if (!(in >> text)) return 0;
  if (text.starts_with("0x")) text.erase(0, 2);
  std::uint32_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, 16);
  return error == std::errc{} ? value : 0;
}

}  // namespace
#endif

std::optional<ScreenCard> screen_card([[maybe_unused]] SDL_Window* window,
                                      [[maybe_unused]] std::string_view connector,
                                      [[maybe_unused]] const GpuDevice& render) {
#if defined(_WIN32)
  auto* hwnd = static_cast<HWND>(
      SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  if (hwnd == nullptr || !render.luid) return std::nullopt;
  const HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
    return std::nullopt;
  std::optional<ScreenCard> found;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT a = 0; !found && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
    IDXGIOutput* output = nullptr;
    for (UINT o = 0; !found && adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
      DXGI_OUTPUT_DESC desc{};
      if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == monitor) {
        DXGI_ADAPTER_DESC1 card{};
        if (SUCCEEDED(adapter->GetDesc1(&card))) {
          char name[256] = {};
          WideCharToMultiByte(CP_UTF8, 0, card.Description, -1, name, sizeof(name), nullptr, nullptr);
          DeviceLuid luid{};
          std::memcpy(luid.data(), &card.AdapterLuid, luid.size());
          found = ScreenCard{.name = name, .renders = luid == *render.luid};
        }
      }
      output->Release();
    }
    adapter->Release();
  }
  factory->Release();
  return found;
#elif defined(__linux__)
  const std::string card = drm_card_of(connector);
  if (card.empty() || render.pci_vendor == 0) return std::nullopt;
  const std::filesystem::path device = std::filesystem::path("/sys/class/drm") / card / "device";
  const std::uint32_t vendor = read_hex(device / "vendor");
  const std::uint32_t id = read_hex(device / "device");
  if (vendor == 0) return std::nullopt;
  // Deux cartes identiques (même fabricant et même modèle) ne se départagent pas ainsi : prises pour une.
  return ScreenCard{
      .name = std::format("PCI {:04x}:{:04x}", vendor, id),
      .renders = vendor == render.pci_vendor && (render.pci_device == 0 || id == render.pci_device)};
#else
  return std::nullopt;
#endif
}

std::optional<i18n::Translated> note_render_device(SDL_Window* window, std::string_view connector,
                                                   const GpuDevice& render, bool smooth_motion) {
  const auto log = log_channel("app");
  if (smooth_motion && render.vendor() == GpuVendor::Nvidia) {
    const SmoothMotionBlock block = smooth_motion_block(render.name, render.driver_version);
    if (block == SmoothMotionBlock::Driver)
      log->warn("pilote NVIDIA {} : Smooth Motion demande 580.82.07 ou plus récent", render.driver_version);
    if (block == SmoothMotionBlock::Gpu)
      log->warn("{} : Smooth Motion ne vaut que pour les RTX 40 et 50, le pilote n'ajoute rien", render.name);
  }
  const std::optional<ScreenCard> card = screen_card(window, connector, render);
  if (!card) return std::nullopt;
  const std::string line = screen_card_line(*card, render.name);
  if (card->renders) {
    log->info("{}", line);
    return std::nullopt;
  }
  log->warn("{}", line);
  static bool told = false;  // une fois par lancement : chaque réouverture de la chaîne repasse ici
  if (told || !screen_card_warns(*card, display_height(window), display_refresh_hz(window)))
    return std::nullopt;
  told = true;
  return i18n::ui_tr(
      "The display is on another graphics card than the one rendering: switch the laptop to dGPU only mode "
      "for G-SYNC and lower latency");
}

}  // namespace ajp::app::play
