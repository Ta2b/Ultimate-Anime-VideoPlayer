// Le script ASS de l'OSD (osd_script) : les lignes, même vides, séparées par \N, et ce qu'ASS
// interpréterait (balises, échappements, fins de ligne) neutralisé dans le texte de dialogue.
#include "app/play/osd_text.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

std::string dialogue_text(const std::string& script) {
  constexpr std::string_view kDialogue = "\nDialogue: ";
  std::size_t at = script.find(kDialogue);
  REQUIRE(at != std::string::npos);
  at += kDialogue.size();
  for (int comma = 0; comma < 9; ++comma) at = script.find(',', at) + 1;
  return script.substr(at, script.size() - at - 1);
}

}  // namespace

TEST_CASE("OSD : script ASS", "[app][osd]") {
  using ajp::app::play::osd_script;
  constexpr ajp::app::play::OutputSize kSize{.width = 1920, .height = 1080};

  SECTION("une coupure entre chaque ligne, même vide") {
    CHECK(dialogue_text(osd_script(std::vector<std::string>{"a", "b"}, kSize)) == "a\\Nb");
    // Trouvé par tests/fuzz : la ligne vide de tête disparaissait.
    CHECK(dialogue_text(osd_script(std::vector<std::string>{"", "titre"}, kSize)) == "\\Ntitre");
    CHECK(dialogue_text(osd_script(std::vector<std::string>{"a", "", ""}, kSize)) == "a\\N\\N");
    CHECK(dialogue_text(osd_script(std::vector<std::string>{}, kSize)).empty());
  }
  SECTION("balises, échappements et fins de ligne neutralisés") {
    CHECK(dialogue_text(osd_script(std::vector<std::string>{"{\\b1}x\\N\r\ny"}, kSize)) == "(∖b1)x∖Ny");
  }
  SECTION("un chemin Windows garde sa forme, sans saut de ligne sur « \\new » (B24-12)") {
    CHECK(dialogue_text(osd_script(std::vector<std::string>{"C:\\Videos\\new\\a.mkv"}, kSize)) ==
          "C:∖Videos∖new∖a.mkv");
  }
  SECTION("les lignes longues se coupent à la largeur de la fenêtre (WrapStyle 0)") {
    CHECK(osd_script(std::vector<std::string>{"a"}, kSize).contains("WrapStyle: 0"));
  }
}

TEST_CASE("C24-08 : un texte qui change à chaque image ne se re-rend que toutes les 250 ms", "[app][osd]") {
  CHECK(ajp::app::play::osd_refresh_due(false, true, false, 0));            // premier texte : tout de suite
  CHECK(ajp::app::play::osd_refresh_due(true, false, false, 0));            // il disparaît : tout de suite
  CHECK(ajp::app::play::osd_refresh_due(true, true, true, 0));              // autre taille : tout de suite
  CHECK_FALSE(ajp::app::play::osd_refresh_due(true, true, false, 16'000));  // statistiques à chaque image
  CHECK(
      ajp::app::play::osd_refresh_due(true, true, false, ajp::app::play::kOsdRefreshUs));  // le rendu suivant
}
