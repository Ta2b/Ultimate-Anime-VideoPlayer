#include "core/keys.h"

#include <algorithm>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "core/toml_fields.h"

namespace ajp::core {
namespace {

using fields::refuse;

char lower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool same_text(std::string_view a, std::string_view b) {
  return std::ranges::equal(a, b, [](char x, char y) { return lower(x) == lower(y); });
}

std::string_view trim(std::string_view text) {
  while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
  while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
  return text;
}

Key offset(Key first, int n) {
  return static_cast<Key>(static_cast<int>(first) + n);
}
int distance(Key first, Key key) {
  return static_cast<int>(key) - static_cast<int>(first);
}

// Touches nommées ; le premier nom d'une touche est sa forme canonique.
constexpr std::array<std::pair<std::string_view, Key>, 29> kNamedKeys{{
    {"Space", Key::Space},
    {"Tab", Key::Tab},
    {"Enter", Key::Enter},
    {"Return", Key::Enter},
    {"Escape", Key::Escape},
    {"Esc", Key::Escape},
    {"Backspace", Key::Backspace},
    {"Delete", Key::Delete},
    {"Del", Key::Delete},
    {"Insert", Key::Insert},
    {"Home", Key::Home},
    {"End", Key::End},
    {"PageUp", Key::PageUp},
    {"PageDown", Key::PageDown},
    {"Left", Key::Left},
    {"Right", Key::Right},
    {"Up", Key::Up},
    {"Down", Key::Down},
    {".", Key::Period},
    {",", Key::Comma},
    {"-", Key::Minus},
    {"=", Key::Equals},
    {"/", Key::Slash},
    {"\\", Key::Backslash},
    {";", Key::Semicolon},
    {"'", Key::Apostrophe},
    {"[", Key::LeftBracket},
    {"]", Key::RightBracket},
    {"`", Key::Grave},
}};

constexpr std::array<std::string_view, kActionCount> kActionNames{
    "pause",
    "seek_backward",
    "seek_forward",
    "step_forward",
    "step_backward",
    "profile_1",
    "profile_2",
    "profile_3",
    "profile_4",
    "profile_5",
    "profile_6",
    "profile_7",
    "profile_8",
    "profile_9",
    "processing_off",
    "toggle_grade",
    "toggle_interpolation",
    "toggle_osd",
};

std::optional<Key> parse_key(std::string_view text) {
  for (const auto& [name, key] : kNamedKeys) {
    if (same_text(name, text)) return key;
  }
  if (text.size() == 1) {
    const char c = lower(text.front());
    if (c >= 'a' && c <= 'z') return offset(Key::A, c - 'a');
    if (c >= '0' && c <= '9') return offset(Key::Digit0, c - '0');
  }
  if (text.size() >= 2 && lower(text.front()) == 'f') {
    int n = 0;
    for (const char c : text.substr(1)) {
      if (c < '0' || c > '9') return std::nullopt;
      n = n * 10 + (c - '0');
      if (n > 12) return std::nullopt;
    }
    if (n >= 1 && text[1] != '0') return offset(Key::F1, n - 1);
  }
  return std::nullopt;
}

std::string key_name(Key key) {
  if (key >= Key::A && key <= Key::Z) return std::string(1, static_cast<char>('A' + distance(Key::A, key)));
  if (key >= Key::Digit0 && key <= Key::Digit9) {
    return std::string(1, static_cast<char>('0' + distance(Key::Digit0, key)));
  }
  if (key >= Key::F1 && key <= Key::F12) return std::format("F{}", distance(Key::F1, key) + 1);
  const auto named = std::ranges::find(kNamedKeys, key, &std::pair<std::string_view, Key>::second);
  return std::string(named->first);
}

Chord ctrl(Key key) {
  return Chord{.key = key, .ctrl = true};
}

std::size_t index(Action action) {
  return static_cast<std::size_t>(action);
}

}  // namespace

std::string_view action_name(Action action) {
  return kActionNames[index(action)];
}

std::optional<Action> parse_action(std::string_view name) {
  const auto found = std::ranges::find(kActionNames, name);
  if (found == kActionNames.end()) return std::nullopt;
  return static_cast<Action>(found - kActionNames.begin());
}

std::expected<Chord, Refusal> parse_chord(std::string_view text) {
  std::vector<std::string_view> tokens;
  for (std::string_view rest = text;;) {
    const std::size_t plus = rest.find('+');
    tokens.push_back(trim(rest.substr(0, plus)));
    if (plus == std::string_view::npos) break;
    rest.remove_prefix(plus + 1);
  }
  const auto unknown = [&] { return refuse(RefusalCode::KeyUnknown, std::format("« {} »", text)); };
  if (std::ranges::any_of(tokens, &std::string_view::empty)) return unknown();

  Chord chord;
  for (const std::string_view modifier : std::span(tokens).first(tokens.size() - 1)) {
    bool* flag = nullptr;
    if (same_text(modifier, "ctrl") || same_text(modifier, "control")) flag = &chord.ctrl;
    if (same_text(modifier, "alt")) flag = &chord.alt;
    if (same_text(modifier, "shift")) flag = &chord.shift;
    if (same_text(modifier, "meta") || same_text(modifier, "cmd")) flag = &chord.meta;
    if (flag == nullptr || *flag) return unknown();  // inconnu ou répété
    *flag = true;
  }
  const std::optional<Key> key = parse_key(tokens.back());
  if (!key) return unknown();
  chord.key = *key;
  return chord;
}

std::string format_chord(const Chord& chord) {
  std::string text;
  if (chord.ctrl) text += "Ctrl+";
  if (chord.alt) text += "Alt+";
  if (chord.shift) text += "Shift+";
  if (chord.meta) text += "Meta+";
  return text + key_name(chord.key);
}

Command command_for(Action action) {
  switch (action) {
    case Action::TogglePause:
      return cmd::TogglePause{};
    case Action::SeekBackward:
      return cmd::SeekRelative{.delta_us = -kSeekStepUs, .exact = true};
    case Action::SeekForward:
      return cmd::SeekRelative{.delta_us = kSeekStepUs, .exact = true};
    case Action::StepForward:
      return cmd::StepFrame{.direction = 1};
    case Action::StepBackward:
      return cmd::StepFrame{.direction = -1};
    case Action::ProcessingOff:
      return cmd::ProcessingOff{};
    case Action::ToggleGrade:
      return cmd::ToggleGrade{};
    case Action::ToggleInterpolation:
      return cmd::ToggleInterpolation{};
    case Action::ToggleOsd:
      return cmd::ToggleOsd{};
    // Chaque cas est nommé : une action ajoutée sans commande fait échouer -Wswitch.
    case Action::Profile1:
    case Action::Profile2:
    case Action::Profile3:
    case Action::Profile4:
    case Action::Profile5:
    case Action::Profile6:
    case Action::Profile7:
    case Action::Profile8:
    case Action::Profile9:
      return cmd::SelectProfile{.slot = static_cast<int>(index(action) - index(Action::Profile1)) + 1};
  }
  return cmd::TogglePause{};  // inatteignable : le switch couvre chaque action
}

KeyMap KeyMap::defaults() {
  KeyMap map;
  map.chords_[index(Action::TogglePause)] = Chord{.key = Key::Space};
  map.chords_[index(Action::SeekBackward)] = Chord{.key = Key::Left};
  map.chords_[index(Action::SeekForward)] = Chord{.key = Key::Right};
  map.chords_[index(Action::StepForward)] = Chord{.key = Key::Period};
  map.chords_[index(Action::StepBackward)] = Chord{.key = Key::Comma};
  for (int slot = 1; slot <= 9; ++slot) {
    map.chords_[index(Action::Profile1) + static_cast<std::size_t>(slot - 1)] =
        ctrl(offset(Key::Digit0, slot));
  }
  map.chords_[index(Action::ProcessingOff)] = ctrl(Key::Digit0);
  map.chords_[index(Action::ToggleGrade)] = ctrl(Key::G);
  map.chords_[index(Action::ToggleInterpolation)] = ctrl(Key::I);
  map.chords_[index(Action::ToggleOsd)] = Chord{.key = Key::Tab};
  return map;
}

const std::optional<Chord>& KeyMap::chord(Action action) const {
  return chords_[index(action)];
}

std::optional<Action> KeyMap::action_for(const Chord& chord) const {
  const auto found = std::ranges::find(chords_, std::optional<Chord>(chord));
  if (found == chords_.end()) return std::nullopt;
  return static_cast<Action>(found - chords_.begin());
}

std::optional<Command> KeyMap::command_for(const Chord& chord) const {
  const std::optional<Action> action = action_for(chord);
  if (!action) return std::nullopt;
  return core::command_for(*action);
}

std::expected<KeyMap, Refusal> KeyMap::rebind(Action action, std::optional<Chord> chord) const {
  if (chord) {
    const std::optional<Action> owner = action_for(*chord);
    if (owner && *owner != action) {
      return refuse(RefusalCode::KeyConflict,
                    std::format("{} = « {} » : déjà pris par {}", action_name(action), format_chord(*chord),
                                action_name(*owner)));
    }
  }
  KeyMap map = *this;
  map.chords_[index(action)] = chord;
  return map;
}

std::expected<KeyMap, Refusal> parse_keys(std::string_view profile_toml) {
  auto root = fields::parse(profile_toml, RefusalCode::ProfileUnreadable, "profil");
  if (!root) return std::unexpected(root.error());
  KeyMap map = KeyMap::defaults();
  const toml::node* keys_node = root->get("keys");
  if (keys_node == nullptr) return map;
  const toml::table* keys = keys_node->as_table();
  if (keys == nullptr) return refuse(RefusalCode::FieldType, "keys");

  // Toutes les redéfinitions d'abord, puis les conflits : échanger deux touches reste possible.
  std::vector<Action> redefined;
  for (const auto& [key, node] : *keys) {
    const std::string where = std::format("keys.{}", key.str());
    const std::optional<Action> action = parse_action(key.str());
    if (!action) return refuse(RefusalCode::KeyUnknown, where);
    const std::optional<std::string> text = node.value_exact<std::string>();
    if (!text) return refuse(RefusalCode::FieldType, where);
    std::optional<Chord> chord;
    if (!text->empty()) {
      auto parsed_chord = parse_chord(*text);
      if (!parsed_chord) return refuse(RefusalCode::KeyUnknown, std::format("{} = « {} »", where, *text));
      chord = *parsed_chord;
    }
    map.chords_[index(*action)] = chord;
    redefined.push_back(*action);
  }
  for (const Action action : redefined) {
    const std::optional<Chord>& chord = map.chord(action);
    if (!chord) continue;
    for (std::size_t other = 0; other < kActionCount; ++other) {
      if (other == index(action) || map.chords_[other] != chord) continue;
      return refuse(RefusalCode::KeyConflict,
                    std::format("keys.{} = « {} » : déjà pris par {}", action_name(action),
                                format_chord(*chord), action_name(static_cast<Action>(other))));
    }
  }
  return map;
}

}  // namespace ajp::core
