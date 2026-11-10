#include "app/play/user_keys.h"

#include <string>
#include <utility>
#include <vector>

#include "app/play/controls.h"
#include "app/play/i18n.h"
#include "core/log.h"
#include "core/user_overlay.h"
#include "core/utf8_path.h"

namespace ajp::app::play {
namespace {

core::KeyMap& key_slot() {
  static core::KeyMap keys = core::KeyMap::defaults();
  return keys;
}

std::vector<std::string>& ignored_slot() {
  static std::vector<std::string> ignored;
  return ignored;
}

}  // namespace

void apply_user_keys(std::string_view user_toml) {
  // Une entrée fautive ne coûte qu'elle-même (A24-07) : les autres redéfinitions tiennent.
  core::KeysRead read = core::read_keys(user_toml);
  for (const std::string& entry : read.ignored)
    log_channel("app")->warn("user.toml [keys] : entrée ignorée : {}", entry);
  key_slot() = read.map;
  ignored_slot() = std::move(read.ignored);
}

void load_user_keys(const std::filesystem::path& user_toml) {
  auto text = core::read_text_file_if_present(user_toml);
  if (!text) {
    log_channel("app")->warn("{} : {} ; raccourcis par défaut", core::to_utf8(user_toml.filename()),
                             text.error().detail);
    key_slot() = core::KeyMap::defaults();
    ignored_slot().clear();
    return;
  }
  apply_user_keys(text->value_or(std::string()));
}

std::vector<std::string> take_ignored_keys() {
  return std::exchange(ignored_slot(), {});
}

void note_ignored_keys(Controls& controls) {
  for (const std::string& entry : take_ignored_keys())
    controls.defer(i18n::ui_tr_format("Shortcut ignored: {}", entry));
}

const core::KeyMap& configured_keys() {
  return key_slot();
}

}  // namespace ajp::app::play
