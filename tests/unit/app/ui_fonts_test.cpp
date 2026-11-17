// La table des polices de l'interface sous Windows (ui_fonts.h) et la taille de l'interface (ui_scale.h).
#include "app/play/ui_fonts.h"

#include <algorithm>
#include <cstddef>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "app/play/ui_scale.h"

namespace ajp::app::play {
namespace {

std::size_t group_of(std::wstring_view file) {
  const auto at = std::ranges::find_if(kWindowsFonts, [&](const WindowsFontGroup& group) {
    return std::ranges::any_of(group.files,
                               [&](const wchar_t* name) { return name != nullptr && file == name; });
  });
  return static_cast<std::size_t>(at - kWindowsFonts.begin());
}

}  // namespace

TEST_CASE("Polices de Windows : le coréen et le chinois ont leur police de repli (A24-23)", "[app][ui]") {
  // La police du bureau d'abord, et la seule sans laquelle on renonce.
  CHECK(group_of(L"segoeui.ttf") == 0);
  CHECK(kWindowsFonts[0].required);
  for (std::size_t i = 1; i < kWindowsFonts.size(); ++i) CHECK_FALSE(kWindowsFonts[i].required);
  // Les polices que Windows 10 et 11 installent toujours pour chaque écriture.
  const std::size_t japanese = group_of(L"YuGothM.ttc");
  const std::size_t korean = group_of(L"malgun.ttf");
  const std::size_t chinese = group_of(L"msyh.ttc");
  REQUIRE(japanese < kWindowsFonts.size());
  REQUIRE(korean < kWindowsFonts.size());
  REQUIRE(chinese < kWindowsFonts.size());
  // Fusion dans l'ordre : un idéogramme commun garde son dessin japonais.
  CHECK(japanese < chinese);
  CHECK(korean != japanese);
  CHECK(korean != chinese);
  // Chaque groupe a au moins une police.
  for (const WindowsFontGroup& group : kWindowsFonts) CHECK(group.files[0] != nullptr);
}

TEST_CASE("ui_scale : un grand écran laissé à 100 % hors Windows s'agrandit, un choix l'emporte",
          "[app][ui]") {
  // L'écran 2560×1440 de Fedora 42 sous GNOME à 100 % : 125 %, comme Windows le propose.
  CHECK(ui_scale(0, 1.0F, 1440, false) == 1.25F);
  CHECK(ui_scale(0, 1.0F, 2160, false) == 1.5F);
  CHECK(ui_scale(0, 1.0F, 1080, false) == 1.0F);
  // Une échelle du bureau déjà choisie, ou celle de Windows et macOS, reste.
  CHECK(ui_scale(0, 1.5F, 1440, false) == 1.5F);
  CHECK(ui_scale(0, 1.0F, 1440, true) == 1.0F);
  CHECK(ui_scale(0, 0.0F, 0, false) == 1.0F);  // échelle inconnue
  // Le choix de l'utilisateur, borné à 100–200 %.
  CHECK(ui_scale(100, 1.0F, 1440, false) == 1.0F);
  CHECK(ui_scale(175, 1.0F, 1080, true) == 1.75F);
  CHECK(ui_scale(400, 1.0F, 1080, true) == 2.0F);
}

}  // namespace ajp::app::play
