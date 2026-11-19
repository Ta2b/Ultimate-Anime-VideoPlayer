// Les polices de l'interface. La principale est embarquée dans le lecteur (Inter, third_party/inter) : le
// menu, la barre et le compteur de FPS ont le même dessin sous Linux, Windows et macOS, quelle que soit la
// police du bureau. Avant, sous Linux, c'était celle de fontconfig (Noto Sans sous Fedora), plus large que
// Segoe UI, et l'OSD prenait « sans-serif » par libass. Les polices du système ne servent plus qu'aux glyphes
// qu'Inter n'a pas : japonais, chinois, coréen, symboles.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/mapped_file.h"

struct ImFont;
struct ImFontAtlas;

namespace kirari::app::play {

// Inter 4.1, Regular (third_party/inter, SIL OFL 1.1), octet pour octet.
extern const std::string_view kUiFontTtf;
// Son nom de famille, pour libass (osd_text.cpp).
inline constexpr std::string_view kUiFontFamily = "Inter";
// Pixels à l'échelle 1, multipliés par l'échelle de l'écran. Inter a un œil plus grand que Segoe UI
// (hauteur d'x 0,546 em contre 0,500) : 16 px d'Inter ont la taille de 17 px de Segoe UI, celle d'avant.
inline constexpr float kUiFontSize = 16.0F;

// Une police du système, lue en mémoire : son fichier et son numéro dans une collection (.ttc).
struct FontFile {
  std::string path;
  int index = 0;
  std::shared_ptr<const MappedFile> data;  // projeté (P4.19)
};

// Les polices de repli du système, dans l'ordre de fusion (fontconfig hors Windows, la table plus bas sous
// Windows). Lent à froid : appelée sur un thread (FontSearch, ui.h).
std::vector<FontFile> search_fonts();

// La police embarquée, puis fallbacks fusionnés derrière elle ; bold : la même, épaissie par FreeType (titres
// des infobulles, panel_tip.cpp), nullptr sans FreeType. fallbacks doit vivre autant que l'atlas.
void load_fonts(ImFontAtlas& atlas, std::vector<FontFile>& fallbacks, ImFont*& bold);

struct WindowsFontGroup {
  std::string_view role;                  // pour le journal
  bool required = false;                  // sans elle, aucune police de repli
  std::array<const wchar_t*, 4> files{};  // la première qui se lit sert ; nullptr : fin de liste
};

// Sous Windows, où il n'y a pas de fontconfig, les polices de repli que Windows 10 et 11 installent toujours,
// par groupe, dans l'ordre où ImGui les fusionne (un glyphe vient du premier groupe qui l'a) :
// - Segoe UI (latin accentué, grec, cyrillique), pour ce qui manquerait à Inter ;
// - Yu Gothic pour le japonais des titres de pistes (Meiryo ou MS Gothic à défaut), avant le chinois : les
//   idéogrammes communs gardent leur dessin japonais ;
// - Malgun Gothic pour le coréen et Microsoft YaHei pour le chinois, sans lesquelles une piste « 한국어 »
//   ou un titre chinois s'affichaient en carrés (A24-23) ;
// - Segoe UI Symbol pour les symboles (▶, filets).
// Hors Windows la table ne sert qu'au test.
inline constexpr std::array<WindowsFontGroup, 5> kWindowsFonts{{
    {.role = "latin", .required = true, .files = {L"segoeui.ttf", L"arial.ttf"}},
    {.role = "japonais", .files = {L"YuGothM.ttc", L"YuGothR.ttc", L"meiryo.ttc", L"msgothic.ttc"}},
    {.role = "coréen", .files = {L"malgun.ttf", L"gulim.ttc"}},  // i18n-ok : journal
    {.role = "chinois", .files = {L"msyh.ttc", L"msyh.ttf", L"simsun.ttc"}},
    {.role = "symboles", .files = {L"seguisym.ttf"}},
}};

}  // namespace kirari::app::play
