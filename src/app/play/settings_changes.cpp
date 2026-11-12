// Les réglages qui diffèrent du défaut, pour le rapport « Signaler un problème » (C39-01).
#include <set>
#include <string>
#include <string_view>

#include "app/play/auto_config.h"
#include "app/play/problem_report.h"
#include "app/play/settings.h"

namespace ajp::app::play {
namespace {

// Les lignes d'un texte TOML, coupées à chaque fin de ligne.
template <class Visit>
void for_each_line(std::string_view text, Visit&& visit) {
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    visit(text.substr(0, end));
    if (end == std::string_view::npos) break;
    text.remove_prefix(end + 1);
  }
}

}  // namespace

std::string settings_changes_text(const PlaySettings& settings) {
  // « section\0clé = valeur » du réglage par défaut : une ligne qui n'y figure pas a été changée.
  std::set<std::string> defaults;
  std::string section;
  for_each_line(format_settings(PlaySettings{}), [&](std::string_view line) {
    if (line.starts_with("[")) {
      section = std::string(line);
    } else if (!line.empty() && !line.starts_with("#")) {
      defaults.insert(section + '\0' + std::string(line));
    }
  });
  std::string out;
  std::string last_section;
  section.clear();
  for_each_line(format_settings(settings), [&](std::string_view line) {
    if (line.starts_with("[")) {
      section = std::string(line);
      return;
    }
    if (line.empty() || line.starts_with("#") || defaults.contains(section + '\0' + std::string(line)))
      return;
    if (section != last_section) {
      if (!out.empty()) out += '\n';
      out += section.empty() ? "[]" : section;
      last_section = section;
    }
    out += '\n';
    out += line;
  });
  return out.empty() ? "(aucun)" : out;
}

void note_preset_facts(const PlaySettings& settings) {
  note_report_fact("Préréglage", label(settings.preset));                        // i18n-ok : rapport
  note_report_fact("Réglages changés", "\n" + settings_changes_text(settings));  // i18n-ok : rapport
}

}  // namespace ajp::app::play
