#include <sstream>
#include <string>

#include "app/play/settings_toml_lenient.h"

namespace ajp::app::play {
namespace {

std::string text_of(const toml::node& node) {
  std::ostringstream out;
  node.visit([&out](const auto& value) { out << value; });
  return out.str();
}

}  // namespace

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
