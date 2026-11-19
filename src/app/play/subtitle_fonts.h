// Les polices que le lecteur livre pour les sous-titres. Un fichier ASS qui nomme sa police (jointe ou du
// système) la garde ; les pistes texte (SRT, WebVTT) et un ASS dont la police manque prennent Gandhi Sans
// (third_party/gandhi-sans), la police de dialogue des sous-titres d'anime de fansub. Le japonais, le chinois
// et le coréen viennent des polices du système ; sous Linux, le paquet joint en dernier recours Noto Sans CJK
// JP (fonts/ à côté du lecteur) pour un système qui n'en a aucune.
#pragma once

#include <filesystem>
#include <string_view>

#include "subs/subtitle_renderer.h"

namespace kirari::app::play {

// Les quatre styles de Gandhi Sans, octet pour octet (src/app/CMakeLists.txt).
extern const std::string_view kGandhiSansRegular;
extern const std::string_view kGandhiSansItalic;
extern const std::string_view kGandhiSansBold;
extern const std::string_view kGandhiSansBoldItalic;

inline constexpr std::string_view kSubtitleFamily = "Gandhi Sans";
// Gandhi Sans est un peu plus étroite que Trebuchet MS : élargie de 5 %, elle en garde la longueur de ligne.
inline constexpr double kSubtitleScaleX = 1.05;
// La police des sous-titres de Crunchyroll et d'ADN, propriétaire : remplacée par Gandhi Sans là où elle
// manque.
inline constexpr std::string_view kCrunchyrollFamily = "Trebuchet MS";
// Le nom du fichier de repli CJK dans fonts/ à côté du lecteur (tools/release/fetch-fonts.sh).
inline constexpr std::string_view kBundledCjkFile = "NotoSansCJKjp-Regular.otf";

// fonts/<kBundledCjkFile> à côté du lecteur ; vide s'il n'y est pas (Windows, macOS, construction locale).
std::filesystem::path bundled_cjk_font();

// options, avec les polices livrées : Gandhi Sans jointe et prise par défaut, à la place de Trebuchet MS
// quand le système ne l'a pas, la police CJK livrée en dernier recours.
subs::RendererOptions with_subtitle_fonts(subs::RendererOptions options);

}  // namespace kirari::app::play
