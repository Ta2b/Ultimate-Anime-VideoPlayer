// Les polices de l'interface sous Windows, où il n'y a pas de fontconfig : celles que Windows 10 et 11
// installent toujours, par groupe, dans l'ordre où ImGui les fusionne (un glyphe vient du premier groupe
// qui l'a). Le japonais passe avant le chinois : les idéogrammes communs gardent leur dessin japonais.
// Hors Windows la table ne sert qu'au test.
#pragma once

#include <array>
#include <string_view>

namespace ajp::app::play {

struct WindowsFontGroup {
  std::string_view role;                  // pour le journal
  bool required = false;                  // sans elle, la police vectorielle d'ImGui
  std::array<const wchar_t*, 4> files{};  // la première qui se lit sert ; nullptr : fin de liste
};

// Segoe UI (latin accentué, grec, cyrillique) ; Yu Gothic pour le japonais des titres de pistes (Meiryo ou
// MS Gothic à défaut) ; Malgun Gothic pour le coréen et Microsoft YaHei pour le chinois, sans lesquelles une
// piste « 한국어 » ou un titre chinois s'affichaient en carrés (A24-23) ; Segoe UI Symbol pour les symboles
// qui manquent aux autres (▶, filets).
inline constexpr std::array<WindowsFontGroup, 5> kWindowsFonts{{
    {.role = "latin", .required = true, .files = {L"segoeui.ttf", L"arial.ttf"}},
    {.role = "japonais", .files = {L"YuGothM.ttc", L"YuGothR.ttc", L"meiryo.ttc", L"msgothic.ttc"}},
    {.role = "coréen", .files = {L"malgun.ttf", L"gulim.ttc"}},  // i18n-ok : journal
    {.role = "chinois", .files = {L"msyh.ttc", L"msyh.ttf", L"simsun.ttc"}},
    {.role = "symboles", .files = {L"seguisym.ttf"}},
}};

}  // namespace ajp::app::play
