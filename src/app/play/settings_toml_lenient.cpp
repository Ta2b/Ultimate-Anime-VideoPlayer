#include "app/play/settings_toml_lenient.h"

#include <format>
#include <string>

namespace ajp::app::play {

// Écarte de root la clé (sinon la section) dont le retrait fait accepter le fichier ; faux si aucune.
bool drop_offending(toml::table& root, const Refusal& refusal, std::vector<std::string>* dropped,
                    const TableReader& read) {
  // Le même refus : même code et même clé ; sans clé (une règle entre deux valeurs), même détail.
  const auto same = [&](const Refusal& other) {
    return other.code == refusal.code && other.key == refusal.key &&
           (!refusal.key.empty() || other.detail == refusal.detail);
  };
  const std::string& detail = refusal.detail;
  for (auto& [section_key, section_node] : root) {
    toml::table* table = section_node.as_table();
    if (table == nullptr) continue;
    for (auto& [key, value] : *table) {
      const std::string where(section_key.str());
      const std::string name(key.str());
      toml::table trial = root;
      if (toml::table* in_trial = trial.get_as<toml::table>(where)) in_trial->erase(name);
      if (auto tried = read(trial); !tried && same(tried.error())) continue;
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
    if (auto tried = read(trial); !tried && same(tried.error())) continue;
    if (dropped != nullptr) dropped->push_back(std::format("[{}] ({})", where, detail));
    root.erase(where);
    return true;
  }
  return false;
}

namespace {

// Les chemins des valeurs de table (hors sous-tables), « a », « b.c »…
void leaves(const toml::table& table, std::vector<std::vector<std::string>>& out,
            std::vector<std::string>& at) {
  for (const auto& [key, node] : table) {
    at.emplace_back(key.str());
    if (const toml::table* child = node.as_table())
      leaves(*child, out, at);
    else
      out.push_back(at);
    at.pop_back();
  }
}

toml::table* table_at(toml::table& root, const std::vector<std::string>& path, bool create) {
  toml::table* table = &root;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    toml::node* next = table->get(path[i]);
    if (next == nullptr && create) next = &table->insert(path[i], toml::table{}).first->second;
    table = next != nullptr ? next->as_table() : nullptr;
    if (table == nullptr) return nullptr;
  }
  return table;
}

}  // namespace

void keep_unknown_keys(toml::table& fresh, const toml::table& existing, const TableReader& read) {
  const auto baseline = read(existing);
  if (!baseline) return;
  std::vector<std::vector<std::string>> paths;
  std::vector<std::string> at;
  leaves(existing, paths, at);
  for (const std::vector<std::string>& path : paths) {
    const toml::table* fresh_parent = table_at(fresh, path, false);
    if (fresh_parent != nullptr && fresh_parent->contains(path.back())) continue;
    toml::table without = existing;
    toml::table* parent = table_at(without, path, false);
    if (parent == nullptr) continue;
    const toml::node* value = parent->get(path.back());
    if (value == nullptr) continue;
    toml::table holder;  // la valeur gardée, copiée avant le retrait
    holder.insert("v", *value);
    parent->erase(path.back());
    const auto changed = read(without);
    if (!changed || !(*changed == *baseline)) continue;  // une clé lue : l'écriture la refait, ou l'ôte
    if (toml::table* target = table_at(fresh, path, true)) target->insert(path.back(), *holder.get("v"));
  }
}

}  // namespace ajp::app::play
