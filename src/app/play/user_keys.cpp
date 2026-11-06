#include "app/play/user_keys.h"

#include <string>

#include "core/log.h"
#include "core/user_overlay.h"

namespace ajp::app::play {
namespace {

core::KeyMap& key_slot() {
  static core::KeyMap keys = core::KeyMap::defaults();
  return keys;
}

}  // namespace

void apply_user_keys(std::string_view user_toml) {
  auto parsed = core::parse_keys(user_toml);
  if (!parsed) {
    log_channel("app")->warn("user.toml [keys] : {} {} ; raccourcis par défaut", name(parsed.error().code),
                             parsed.error().detail);
    key_slot() = core::KeyMap::defaults();
    return;
  }
  key_slot() = *parsed;
}

void load_user_keys(const std::filesystem::path& user_toml) {
  auto text = core::read_text_file_if_present(user_toml);
  if (!text) {
    log_channel("app")->warn("{} : {} ; raccourcis par défaut", user_toml.string(), text.error().detail);
    key_slot() = core::KeyMap::defaults();
    return;
  }
  apply_user_keys(text->value_or(std::string()));
}

const core::KeyMap& configured_keys() {
  return key_slot();
}

}  // namespace ajp::app::play
