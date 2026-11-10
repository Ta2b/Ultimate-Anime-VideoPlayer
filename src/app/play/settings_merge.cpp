#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "app/play/settings_toml_lenient.h"

namespace ajp::app::play {
namespace {

std::string text_of(const toml::node& node) {
  std::ostringstream out;
  node.visit([&out](const auto& value) { out << value; });
  return out.str();
}

// Le nœud au chemin path (sections puis clé), ou nul.
const toml::node* node_at(const toml::table& root, const std::vector<std::string>& path) {
  const toml::table* table = &root;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const toml::node* next = table->get(path[i]);
    table = next != nullptr ? next->as_table() : nullptr;
    if (table == nullptr) return nullptr;
  }
  return table->get(path.back());
}

bool same_value(const toml::node& a, const toml::node& b) {
  return a.type() == b.type() && text_of(a) == text_of(b);
}

// Les valeurs de existing absentes de pruned : chemin et valeur. Une section entière écartée donne
// chacune de ses valeurs.
void rejected_leaves(const toml::table& existing, const toml::table* pruned, std::vector<std::string>& at,
                     std::vector<std::pair<std::vector<std::string>, const toml::node*>>& out) {
  for (const auto& [key, node] : existing) {
    at.emplace_back(key.str());
    const toml::node* kept = pruned != nullptr ? pruned->get(key.str()) : nullptr;
    if (const toml::table* child = node.as_table()) {
      rejected_leaves(*child, kept != nullptr ? kept->as_table() : nullptr, at, out);
    } else if (kept == nullptr) {
      out.emplace_back(at, &node);
    }
    at.pop_back();
  }
}

}  // namespace

void keep_rejected_values(toml::table& fresh, const toml::table& existing, const toml::table& pruned,
                          const toml::table& read_back) {
  std::vector<std::pair<std::vector<std::string>, const toml::node*>> rejected;
  std::vector<std::string> at;
  rejected_leaves(existing, &pruned, at, rejected);
  for (const auto& [path, value] : rejected) {
    const toml::node* now = node_at(fresh, path);
    const toml::node* lenient = node_at(read_back, path);
    // Changée par ce processus depuis la lecture : la sienne l'emporte.
    if (now != nullptr && (lenient == nullptr || !same_value(*now, *lenient))) continue;
    toml::table* table = &fresh;
    for (std::size_t i = 0; i + 1 < path.size() && table != nullptr; ++i) {
      toml::node* next = table->get(path[i]);
      if (next == nullptr) next = &table->insert(path[i], toml::table{}).first->second;
      table = next->as_table();
    }
    if (table != nullptr)
      value->visit([table, &key = path.back()](const auto& copy) { table->insert_or_assign(key, copy); });
  }
}

void apply_own_changes(toml::table& disk, const toml::table& before, const toml::table& now) {
  for (const auto& [key, value] : now) {
    const toml::node* was = before.get(key.str());
    const toml::table* now_table = value.as_table();
    if (now_table != nullptr && was != nullptr && was->is_table()) {
      toml::table* on_disk = disk.get_as<toml::table>(key.str());
      if (on_disk == nullptr) {
        disk.insert_or_assign(key.str(), *now_table);
        continue;
      }
      apply_own_changes(*on_disk, *was->as_table(), *now_table);
      continue;
    }
    if (was == nullptr || was->type() != value.type() || text_of(*was) != text_of(value))
      value.visit([&disk, &key](const auto& copy) { disk.insert_or_assign(key.str(), copy); });
  }
  for (const auto& [key, value] : before)
    if (!now.contains(key.str())) disk.erase(key.str());
}

}  // namespace ajp::app::play
