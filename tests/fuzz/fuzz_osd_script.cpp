// Texte de l'OSD (noms de pistes, titres, chemins venus du fichier lu) dans un script ASS : les
// octets, coupés en lignes sur les zéros, ne sortent jamais de leur ligne de dialogue (ni retour à
// la ligne, ni accolade de balise, ni barre oblique d'échappement), et les lignes restent
// séparées par autant de \N.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/osd_text.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"

int ajp::fuzz::osd_script(const std::uint8_t* data, std::size_t size) {
  std::vector<std::string> lines(1);
  for (const char c : std::string_view(reinterpret_cast<const char*>(data), size)) {
    if (c == '\0') {
      lines.emplace_back();
    } else {
      lines.back() += c;
    }
  }
  const std::string script = ajp::app::play::osd_script(lines, {.width = 1920, .height = 1080});
  constexpr std::string_view kDialogue = "\nDialogue: ";
  const std::size_t at = script.find(kDialogue);
  ajp::fuzz::require(at != std::string::npos, "une ligne de dialogue");
  ajp::fuzz::require(script.find(kDialogue, at + 1) == std::string::npos, "une seule ligne de dialogue");
  ajp::fuzz::require(script.back() == '\n', "le script finit sa ligne");
  // Le texte : après la neuvième virgule de la ligne de dialogue, jusqu'à la fin.
  std::size_t text = at + kDialogue.size();
  for (int comma = 0; comma < 9; ++comma) text = script.find(',', text) + 1;
  const std::string_view body(script.data() + text, script.size() - text - 1);
  std::size_t breaks = 0;
  for (std::size_t i = 0; i < body.size(); ++i) {
    const char c = body[i];
    ajp::fuzz::require(c != '\n' && c != '\r' && c != '{' && c != '}', "rien ne sort de la ligne");
    if (c == '\\') {
      ajp::fuzz::require(i + 1 < body.size() && body[i + 1] == 'N', "seul \\N est une échappée");
      ++breaks;
    }
  }
  ajp::fuzz::require(breaks + 1 == lines.size(), "autant de \\N que de lignes moins une");
  return 0;
}
