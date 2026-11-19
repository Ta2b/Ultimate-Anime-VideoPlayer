#include "app/play/subtitle_fonts.h"

#include <cstddef>
#include <span>
#include <string>
#include <system_error>
#include <utility>

#include <SDL3/SDL_filesystem.h>

#include "core/utf8_path.h"

#if KIRARI_WITH_FONTCONFIG
#include <fontconfig/fontconfig.h>
#endif

namespace kirari::app::play {
namespace {

std::span<const std::byte> bytes(std::string_view data) {
  return std::as_bytes(std::span(data));
}

// family est installée : fontconfig la connaît sous ce nom exact. Sans fontconfig (Windows, macOS), le
// système a les polices courantes, Trebuchet MS comprise.
bool system_has_family(std::string_view family) {
#if KIRARI_WITH_FONTCONFIG
  const std::string name(family);
  FcPattern* pattern = FcPatternCreate();
  FcPatternAddString(pattern, FC_FAMILY, reinterpret_cast<const FcChar8*>(name.c_str()));
  FcObjectSet* objects = FcObjectSetBuild(FC_FAMILY, nullptr);
  FcFontSet* fonts = FcFontList(nullptr, pattern, objects);
  const bool found = fonts != nullptr && fonts->nfont > 0;
  if (fonts != nullptr) FcFontSetDestroy(fonts);
  FcObjectSetDestroy(objects);
  FcPatternDestroy(pattern);
  return found;
#else
  (void)family;
  return true;
#endif
}

}  // namespace

std::filesystem::path bundled_cjk_font() {
  const char* base = SDL_GetBasePath();
  if (base == nullptr) return {};
  const std::filesystem::path path =
      core::path_from_utf8(base) / "fonts" / core::path_from_utf8(kBundledCjkFile);
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) ? path : std::filesystem::path();
}

subs::RendererOptions with_subtitle_fonts(subs::RendererOptions options) {
  options.bundled.push_back({.name = "GandhiSans-Regular.otf", .data = bytes(kGandhiSansRegular)});
  options.bundled.push_back({.name = "GandhiSans-Italic.otf", .data = bytes(kGandhiSansItalic)});
  options.bundled.push_back({.name = "GandhiSans-Bold.otf", .data = bytes(kGandhiSansBold)});
  options.bundled.push_back({.name = "GandhiSans-BoldItalic.otf", .data = bytes(kGandhiSansBoldItalic)});
  // Une police absente d'un fichier ASS retombe sur Gandhi Sans plutôt que sur la police système du moment.
  options.default_family = std::string(kSubtitleFamily);
  // Dernier recours de libass quand ni la police demandée ni celles du système n'ont le glyphe (japonais sur
  // un Linux sans police CJK).
  if (options.default_font.empty()) options.default_font = bundled_cjk_font();
  // Trebuchet MS, la police des sous-titres de Crunchyroll et d'ADN, n'est livrée qu'avec Windows et macOS.
  // Ailleurs, Gandhi Sans, du même dessin, élargie pour garder la longueur des lignes.
  if (!system_has_family(kCrunchyrollFamily)) {
    options.substitutes.push_back({.family = std::string(kCrunchyrollFamily),
                                   .replacement = std::string(kSubtitleFamily),
                                   .scale_x = kSubtitleScaleX,
                                   .attachment_prefix = "trebuc"});
  }
  return options;
}

}  // namespace kirari::app::play
