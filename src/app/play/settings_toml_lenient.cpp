#include "app/play/settings_toml_lenient.h"

#include <format>
#include <string>

namespace ajp::app::play {

// Écarte de root la clé (sinon la section) dont le retrait fait accepter le fichier ; faux si aucune.
bool drop_offending(toml::table& root, const std::string& detail, std::vector<std::string>* dropped,
                    const TableReader& read) {
  for (auto& [section_key, section_node] : root) {
    toml::table* table = section_node.as_table();
    if (table == nullptr) continue;
    for (auto& [key, value] : *table) {
      const std::string where(section_key.str());
      const std::string name(key.str());
      toml::table trial = root;
      if (toml::table* in_trial = trial.get_as<toml::table>(where)) in_trial->erase(name);
      if (auto tried = read(trial); !tried && tried.error().detail == detail) continue;
      if (dropped != nullptr) dropped->push_back(std::format("{}.{} ({})", where, name, detail));
      table->erase(name);
      return true;
    }
  }
  for (auto& [section_key, section_node] : root) {
    if (section_node.as_table() == nullptr) continue;
    const std::string where(section_key.str());
    toml::table trial = root;
    trial.erase(where);
    if (auto tried = read(trial); !tried && tried.error().detail == detail) continue;
    if (dropped != nullptr) dropped->push_back(std::format("[{}] ({})", where, detail));
    root.erase(where);
    return true;
  }
  return false;
}

}  // namespace ajp::app::play
