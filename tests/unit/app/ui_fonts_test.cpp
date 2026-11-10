// La table des polices de l'interface sous Windows (ui_fonts.h).
#include "app/play/ui_fonts.h"

#include <algorithm>
#include <cstddef>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

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

}  // namespace ajp::app::play
