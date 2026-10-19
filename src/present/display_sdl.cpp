#include <cmath>
#include <format>
#include <string_view>

#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

#include "present/display_system.h"

#ifdef _WIN32
#include "present/display_win32.h"
#endif

namespace ajp::present {

namespace {

// Blanc de scRGB : la valeur 1,0 vaut 80 nits. Sous Windows, SDL3 donne le blanc SDR de la fenêtre
// dans cette unité (SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT) et la marge HDR en multiples de ce blanc.
constexpr double kScRgbWhiteNits = 80.0;

std::unexpected<Refusal> sdl_refusal(std::string_view call) {
  return std::unexpected(Refusal{RefusalCode::PresentFailed, std::format("{} : {}", call, SDL_GetError())});
}

void read_mode(const SDL_DisplayMode& mode, SystemDisplay& out) {
  const double density = mode.pixel_density > 0.0F ? mode.pixel_density : 1.0;
  out.width = static_cast<int>(std::lround(mode.w * density));
  out.height = static_cast<int>(std::lround(mode.h * density));
  if (mode.refresh_rate_numerator > 0 && mode.refresh_rate_denominator > 0) {
    out.refresh = core::Ratio{mode.refresh_rate_numerator, mode.refresh_rate_denominator};
  } else if (mode.refresh_rate > 0.0F) {
    out.refresh = core::Ratio{std::llround(mode.refresh_rate * 1000.0), 1000};
  } else {
    out.unknown.emplace_back("rafraîchissement : le mode courant ne le précise pas");
  }
}

void read_hdr(SDL_Window* window, SDL_DisplayID display, SystemDisplay& out) {
  const SDL_PropertiesID window_props = SDL_GetWindowProperties(window);
  const SDL_PropertiesID display_props = SDL_GetDisplayProperties(display);
  const bool hdr = SDL_GetBooleanProperty(window_props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false) ||
                   SDL_GetBooleanProperty(display_props, SDL_PROP_DISPLAY_HDR_ENABLED_BOOLEAN, false);
  out.hdr = hdr;
  if (!hdr) return;
  const double white = SDL_GetFloatProperty(window_props, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 0.0F);
  const double headroom = SDL_GetFloatProperty(window_props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F);
  const char* driver = SDL_GetCurrentVideoDriver();
  if (driver != nullptr && std::string_view(driver) == "windows") {
    if (white > 0.0) out.sdr_white_nits = white * kScRgbWhiteNits;
    if (white > 0.0 && headroom > 0.0) out.peak_nits = headroom * white * kScRgbWhiteNits;
    return;
  }
  // Wayland (color-management-v1) et macOS : la marge est relative au blanc de référence du
  // compositeur, que SDL3 ne transmet pas (son blanc vaut toujours 1). Une sortie PQ y place son
  // blanc à 203 nits (BT.2408) : la crête vue du contenu vaut marge × 203, comme dans mpv.
  if (headroom > 0.0) out.peak_nits = headroom * core::display_defaults::kSdrWhiteNits;
  out.unknown.emplace_back("blanc SDR du compositeur : SDL3 n'en donne que la marge relative");
}

}  // namespace

std::expected<SystemDisplay, Refusal> read_system_display(const Window& window) {
  const SDL_DisplayID display = SDL_GetDisplayForWindow(window.sdl());
  if (display == 0) return sdl_refusal("SDL_GetDisplayForWindow");
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(display);
  if (mode == nullptr) return sdl_refusal("SDL_GetCurrentDisplayMode");

  SystemDisplay out;
  if (const char* name = SDL_GetDisplayName(display)) out.name = name;
  read_mode(*mode, out);
  read_hdr(window.sdl(), display, out);
  out.unknown.emplace_back("VRR : SDL3 ne le dit pas, il vient des réglages");
#ifdef _WIN32
  read_dxgi(window.sdl(), out);
#else
  // Sans libdisplay-info (hors des dépendances), ni EDID ni DisplayID : ce qui manque.
  out.unknown.emplace_back("noir de l'écran : EDID et DisplayID non lus (libdisplay-info absente)");
  out.unknown.emplace_back("indice OLED (DisplayID 0x21, octet 0x1E) : non lu");
  if (!out.peak_nits) out.unknown.emplace_back("crête du panneau : seul le compositeur en HDR la donne");
#endif
  return out;
}

}  // namespace ajp::present
