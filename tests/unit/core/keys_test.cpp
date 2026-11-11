#include "core/keys.h"

#include <algorithm>
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

  CHECK_FALSE(map.command_for(ctrl(Key::Digit1)));  // les profils, retirés
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
  CHECK(action_name(Action::ToggleGrade) == "toggle_grade");
  CHECK_FALSE(parse_action("Pause"));
  CHECK_FALSE(parse_action("profile_3"));
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

TEST_CASE("Raccourcis : ceux des profils retirés d'un ancien user.toml sont ignorés", "[keys]") {
  auto old = parse_keys("[keys]\nprofile_3 = \"F3\"\ntoggle_osd = \"F1\"\n");
  REQUIRE(old);
  CHECK(old->chord(Action::ToggleOsd) == key(Key::F1));
  CHECK_FALSE(old->action_for(key(Key::F3)));
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

  CHECK(keys_refusal("[keys]\ntoggle_grade = \"Ctrl+0\"\n") == RefusalCode::KeyConflict);
  CHECK(keys_refusal("[keys]\nzoom = \"Z\"\n") == RefusalCode::KeyUnknown);
  CHECK(keys_refusal("[keys]\npause = \"Ctrl+Nope\"\n") == RefusalCode::KeyUnknown);
  CHECK(keys_refusal("[keys]\npause = 3\n") == RefusalCode::FieldType);
  CHECK(keys_refusal("keys = 1\n") == RefusalCode::FieldType);
  CHECK(keys_refusal("[keys\n") == RefusalCode::ProfileUnreadable);
}

TEST_CASE("Raccourcis : la lecture tolérante ne perd que l'entrée fautive (A24-07)", "[keys]") {
  const KeysRead read = read_keys(R"(
[keys]
toggle_osd = "F1"
zoom = "Z"
pause = "Ctrl+Nope"
seek_forward = 3
toggle_grade = "Space"
)");
  // Avant : une seule de ces fautes rendait tous les raccourcis par défaut, F1 compris.
  CHECK(read.map.chord(Action::ToggleOsd) == Chord{.key = Key::F1});
  CHECK(read.map.action_for(key(Key::Space)) == Action::TogglePause);
  CHECK(read.map.chord(Action::SeekForward) == KeyMap::defaults().chord(Action::SeekForward));
  // Le conflit : toggle_grade reprend son raccourci par défaut, resté libre.
  CHECK(read.map.chord(Action::ToggleGrade) == ctrl(Key::G));
  REQUIRE(read.ignored.size() == 4);
  const auto named = [&](std::string_view what) {
    return std::ranges::any_of(read.ignored, [&](const std::string& e) { return e.contains(what); });
  };
  CHECK(named("zoom"));
  CHECK(named("Ctrl+Nope"));
  CHECK(named("seek_forward"));
  CHECK(named("toggle_grade"));

  // Un échange reste un échange, sans rien d'ignoré ; un fichier sain se lit comme avec parse_keys.
  const KeysRead swapped = read_keys("[keys]\nseek_backward = \"Right\"\nseek_forward = \"Left\"\n");
  CHECK(swapped.ignored.empty());
  CHECK(swapped.map == *parse_keys("[keys]\nseek_backward = \"Right\"\nseek_forward = \"Left\"\n"));
  CHECK(read_keys("version = 5\n").map == KeyMap::defaults());

  // Illisible : les raccourcis par défaut, et la raison.
  const KeysRead broken = read_keys("[keys\n");
  CHECK(broken.map == KeyMap::defaults());
  CHECK(broken.ignored.size() == 1);
}

TEST_CASE("Raccourcis : une touche gardée par le lecteur n'est pas donnée à une action (A24-06)", "[keys]") {
  // Avant : « pause = "M" » était accepté, puis M coupait le son et la pause n'avait plus de touche.
  const KeysRead read =
      read_keys("[keys]\npause = \"M\"\ntoggle_osd = \"Shift+N\"\nstep_forward = \"Ctrl+M\"\n");
  CHECK(read.map.chord(Action::TogglePause) == KeyMap::defaults().chord(Action::TogglePause));
  CHECK(read.map.chord(Action::ToggleOsd) == KeyMap::defaults().chord(Action::ToggleOsd));
  CHECK(read.map.chord(Action::StepForward) == ctrl(Key::M));  // avec Ctrl, la touche est libre
  REQUIRE(read.ignored.size() == 2);
  CHECK(read.ignored[0].contains("(fixed)"));
  // Aucun raccourci par défaut n'est sur une touche gardée.
  for (std::size_t a = 0; a < kActionCount; ++a) {
    const auto chord = KeyMap::defaults().chord(static_cast<Action>(a));
    if (chord && !chord->ctrl && !chord->alt && !chord->meta) CHECK_FALSE(fixed_key(chord->key));
  }
}

TEST_CASE("Raccourcis : Ctrl+Moins et Ctrl+Plus règlent le délai audio, [keys] ne les prend pas (B38-10)",
          "[keys]") {
  // Avant : « toggle_grade = "Ctrl+=" » était accepté sans avis, puis la touche réglait le délai audio.
  const KeysRead read =
      read_keys("[keys]\ntoggle_grade = \"Ctrl+=\"\npause = \"Ctrl+-\"\ntoggle_osd = \"Ctrl+Shift+=\"\n");
  REQUIRE(read.ignored.size() == 3);
  for (const std::string& line : read.ignored) CHECK(line.contains("(fixed)"));
  CHECK(read.map.chord(Action::ToggleGrade) == KeyMap::defaults().chord(Action::ToggleGrade));
  CHECK(fixed_chord(ctrl(Key::Minus)));
  CHECK_FALSE(fixed_chord(ctrl(Key::G)));
  CHECK(fixed_chord(key(Key::Q)));
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
