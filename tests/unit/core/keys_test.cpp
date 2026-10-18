#include "core/keys.h"

#include <string>
#include <variant>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;
namespace cmd = ajp::cmd;

namespace {

Chord key(Key k) {
  return Chord{.key = k};
}
Chord ctrl(Key k) {
  return Chord{.key = k, .ctrl = true};
}

ajp::Command command_of(const KeyMap& map, const Chord& chord) {
  auto command = map.command_for(chord);
  REQUIRE(command);
  return *command;
}

RefusalCode keys_refusal(const std::string& toml) {
  auto map = parse_keys(toml);
  REQUIRE_FALSE(map);
  return map.error().code;
}

}  // namespace

TEST_CASE("Raccourcis : les touches par défaut du plan donnent leurs commandes", "[keys]") {
  const KeyMap map = KeyMap::defaults();
  CHECK(std::holds_alternative<cmd::TogglePause>(command_of(map, key(Key::Space))));

  const auto back = std::get<cmd::SeekRelative>(command_of(map, key(Key::Left)));
  CHECK(back.delta_us == -5'000'000);
  CHECK(back.exact);
  CHECK(std::get<cmd::SeekRelative>(command_of(map, key(Key::Right))).delta_us == 5'000'000);
  CHECK(std::get<cmd::StepFrame>(command_of(map, key(Key::Period))).direction == 1);
  CHECK(std::get<cmd::StepFrame>(command_of(map, key(Key::Comma))).direction == -1);

  CHECK(std::get<cmd::SelectProfile>(command_of(map, ctrl(Key::Digit1))).slot == 1);
  CHECK(std::get<cmd::SelectProfile>(command_of(map, ctrl(Key::Digit9))).slot == 9);
  CHECK(std::holds_alternative<cmd::ProcessingOff>(command_of(map, ctrl(Key::Digit0))));
  CHECK(std::holds_alternative<cmd::ToggleGrade>(command_of(map, ctrl(Key::G))));
  CHECK(std::holds_alternative<cmd::ToggleInterpolation>(command_of(map, ctrl(Key::I))));
  CHECK(std::holds_alternative<cmd::ToggleOsd>(command_of(map, key(Key::Tab))));

  CHECK_FALSE(map.command_for(key(Key::G)));
  CHECK_FALSE(map.command_for(Chord{.key = Key::Space, .shift = true}));
}

TEST_CASE("Raccourcis : chaque action a un nom de [keys] qui se relit", "[keys]") {
  for (std::size_t i = 0; i < kActionCount; ++i) {
    const auto action = static_cast<Action>(i);
    CHECK(parse_action(action_name(action)) == action);
  }
  CHECK(action_name(Action::TogglePause) == "pause");
  CHECK(action_name(Action::Profile3) == "profile_3");
  CHECK_FALSE(parse_action("Pause"));
  CHECK_FALSE(parse_action("profile_10"));
}

TEST_CASE("Raccourcis : le texte d'une touche se lit sans égard à la casse et s'écrit canonique", "[keys]") {
  const auto round_trip = [](const std::string& text) {
    auto chord = parse_chord(text);
    REQUIRE(chord);
    return format_chord(*chord);
  };
  CHECK(round_trip("ctrl+shift+g") == "Ctrl+Shift+G");
  CHECK(round_trip("Shift+Control+g") == "Ctrl+Shift+G");
  CHECK(round_trip("Meta+Alt+Left") == "Alt+Meta+Left");
  CHECK(round_trip("cmd+q") == "Meta+Q");
  CHECK(round_trip(" Ctrl + 1 ") == "Ctrl+1");
  CHECK(round_trip(".") == ".");
  CHECK(round_trip(",") == ",");
  CHECK(round_trip("space") == "Space");
  CHECK(round_trip("Esc") == "Escape");
  CHECK(round_trip("Return") == "Enter");
  CHECK(round_trip("f11") == "F11");
  CHECK(round_trip("Alt+F4") == "Alt+F4");
  CHECK(round_trip("\\") == "\\");

  auto chord = parse_chord("Ctrl+G");
  REQUIRE(chord);
  CHECK(*chord == Chord{.key = Key::G, .ctrl = true});
}

TEST_CASE("Raccourcis : chaque touche, avec chaque modificateur, s'écrit puis se relit", "[keys]") {
  // format_chord nomme chaque touche : une touche ajoutée sans nom ferait échouer ce test.
  for (int k = 0; k <= static_cast<int>(Key::F12); ++k) {
    for (const Chord modifiers : {Chord{}, Chord{.ctrl = true, .alt = true, .shift = true, .meta = true}}) {
      Chord chord = modifiers;
      chord.key = static_cast<Key>(k);
      const std::string text = format_chord(chord);
      INFO(text);
      CHECK(parse_chord(text) == chord);
    }
  }
}

TEST_CASE("Raccourcis : Ctrl+1 à Ctrl+9 choisissent chacun leur profil", "[keys]") {
  const KeyMap map = KeyMap::defaults();
  for (int slot = 1; slot <= 9; ++slot) {
    const auto action = static_cast<Action>(static_cast<int>(Action::Profile1) + slot - 1);
    CHECK(std::get<cmd::SelectProfile>(command_for(action)).slot == slot);
    CHECK(map.action_for(ctrl(static_cast<Key>(static_cast<int>(Key::Digit0) + slot))) == action);
  }
}

TEST_CASE("Raccourcis : une touche inconnue est un refus nommé", "[keys]") {
  for (const char* text :
       {"", "Ctrl+", "+G", "Ctrl+Ctrl+G", "Hyper+G", "F0", "F13", "F01", "Foo", "Ctrl", "G+Ctrl"}) {
    auto chord = parse_chord(text);
    INFO(text);
    REQUIRE_FALSE(chord);
    CHECK(chord.error().code == RefusalCode::KeyUnknown);
  }
}

TEST_CASE("Raccourcis : [keys] redéfinit, retire ou échange des touches", "[keys]") {
  auto map = parse_keys(R"(
version = 5
[keys]
toggle_osd = "F1"
pause = ""
seek_backward = "Right"
seek_forward = "Left"
)");
  REQUIRE(map);
  CHECK(map->chord(Action::ToggleOsd) == Chord{.key = Key::F1});
  CHECK_FALSE(map->chord(Action::TogglePause));
  CHECK_FALSE(map->action_for(key(Key::Space)));
  CHECK(map->action_for(key(Key::Right)) == Action::SeekBackward);
  CHECK(map->action_for(key(Key::Left)) == Action::SeekForward);
  CHECK(map->chord(Action::ToggleGrade) == ctrl(Key::G));

  auto without_section = parse_keys("version = 5\n");
  REQUIRE(without_section);
  CHECK(*without_section == KeyMap::defaults());
}

TEST_CASE("Raccourcis : un conflit, une action ou une touche inconnue sont refusés", "[keys]") {
  auto conflict = parse_keys("[keys]\ntoggle_osd = \"Space\"\n");
  REQUIRE_FALSE(conflict);
  CHECK(conflict.error().code == RefusalCode::KeyConflict);
  CHECK(conflict.error().detail.find("toggle_osd") != std::string::npos);
  CHECK(conflict.error().detail.find("pause") != std::string::npos);

  CHECK(keys_refusal("[keys]\ntoggle_grade = \"Ctrl+1\"\n") == RefusalCode::KeyConflict);
  CHECK(keys_refusal("[keys]\nzoom = \"Z\"\n") == RefusalCode::KeyUnknown);
  CHECK(keys_refusal("[keys]\npause = \"Ctrl+Nope\"\n") == RefusalCode::KeyUnknown);
  CHECK(keys_refusal("[keys]\npause = 3\n") == RefusalCode::FieldType);
  CHECK(keys_refusal("keys = 1\n") == RefusalCode::FieldType);
  CHECK(keys_refusal("[keys\n") == RefusalCode::ProfileUnreadable);
}

TEST_CASE("Raccourcis : rebind refuse de prendre la touche d'une autre action", "[keys]") {
  const KeyMap map = KeyMap::defaults();
  auto conflict = map.rebind(Action::ToggleOsd, key(Key::Space));
  REQUIRE_FALSE(conflict);
  CHECK(conflict.error().code == RefusalCode::KeyConflict);

  auto moved = map.rebind(Action::ToggleOsd, key(Key::F1));
  REQUIRE(moved);
  CHECK(moved->action_for(key(Key::F1)) == Action::ToggleOsd);
  CHECK_FALSE(moved->action_for(key(Key::Tab)));

  auto same = map.rebind(Action::TogglePause, key(Key::Space));
  REQUIRE(same);
  CHECK(*same == map);

  auto unbound = map.rebind(Action::TogglePause, std::nullopt);
  REQUIRE(unbound);
  auto taken = unbound->rebind(Action::ToggleOsd, key(Key::Space));
  REQUIRE(taken);
  CHECK(taken->action_for(key(Key::Space)) == Action::ToggleOsd);
}
