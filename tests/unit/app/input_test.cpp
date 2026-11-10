// Les touches du lecteur (app/play/input.h) : chaque touche de core::Key se lit d'un code SDL (A24-04), et le
// délai audio se lit aussi par la touche physique (A24-05).
#include "app/play/input.h"

#include <array>
#include <set>
#include <variant>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("Touches : chaque touche de [keys] a un code SDL qui la rend (A24-04)", "[app][input]") {
  std::set<core::Key> seen;
  for (SDL_Keycode key = SDLK_A; key <= SDLK_Z; ++key) seen.insert(key_of(key).value());
  for (SDL_Keycode key = SDLK_0; key <= SDLK_9; ++key) seen.insert(key_of(key).value());
  for (SDL_Keycode key = SDLK_F1; key <= SDLK_F12; ++key) seen.insert(key_of(key).value());
  constexpr std::array named{
      SDLK_SPACE,        SDLK_TAB,   SDLK_RETURN,    SDLK_ESCAPE,    SDLK_BACKSPACE,  SDLK_DELETE,
      SDLK_INSERT,       SDLK_HOME,  SDLK_END,       SDLK_PAGEUP,    SDLK_PAGEDOWN,   SDLK_LEFT,
      SDLK_RIGHT,        SDLK_UP,    SDLK_DOWN,      SDLK_PERIOD,    SDLK_COMMA,      SDLK_MINUS,
      SDLK_EQUALS,       SDLK_SLASH, SDLK_BACKSLASH, SDLK_SEMICOLON, SDLK_APOSTROPHE, SDLK_LEFTBRACKET,
      SDLK_RIGHTBRACKET, SDLK_GRAVE};
  for (const SDL_Keycode key : named) {
    const auto read = key_of(key);
    REQUIRE(read);  // avant : « ; », « - », « [ »… lus dans user.toml, jamais rendus par le clavier
    CHECK(seen.insert(*read).second);  // deux codes ne rendent pas la même touche
  }
  // Toutes les touches de core::Key, de la première à la dernière.
  CHECK(seen.size() == static_cast<std::size_t>(core::Key::F12) + 1);
  CHECK_FALSE(key_of(SDLK_CAPSLOCK));
}

TEST_CASE("Touches : une touche redéfinie sur « ; » arrive au lecteur (A24-04)", "[app][input]") {
  const auto keys = core::parse_keys("[keys]\nstep_forward = \";\"\n");
  REQUIRE(keys);
  SDL_Event down{};
  down.type = SDL_EVENT_KEY_DOWN;
  down.key.key = SDLK_SEMICOLON;
  down.key.scancode = SDL_SCANCODE_SEMICOLON;
  const auto input = input_of(down, *keys);
  REQUIRE(input);
  const auto* command = std::get_if<Command>(&*input);
  REQUIRE(command != nullptr);
  CHECK(std::holds_alternative<cmd::StepFrame>(*command));
  // Maintenue : l'image suivante encore (la répétition passe par la même lecture de la touche, A24-05).
  down.key.repeat = true;
  const auto held = input_of(down, *keys);
  REQUIRE(held);
  CHECK(std::holds_alternative<cmd::StepFrame>(std::get<Command>(*held)));
}

TEST_CASE("Touches : Ctrl + la touche du « - » règle le délai audio, quel que soit le code rendu (A24-05)",
          "[app][input]") {
  const core::KeyMap keys = core::KeyMap::defaults();
  SDL_Event down{};
  down.type = SDL_EVENT_KEY_DOWN;
  down.key.mod = SDL_KMOD_LCTRL;
  // AZERTY avec les chiffres français de SDL 3.4 : la touche rend « 6 », son caractère reste « - ». Ici la
  // disposition par défaut de SDL donne « - » à la touche physique SDL_SCANCODE_MINUS.
  down.key.key = SDLK_6;
  down.key.scancode = SDL_SCANCODE_MINUS;
  const auto less = input_of(down, keys);
  REQUIRE(less);
  const auto* step = std::get_if<SubtitleDelayStep>(&*less);
  REQUIRE(step != nullptr);
  CHECK(step->audio);
  CHECK(step->delta_ms == -100);
  down.key.key = SDLK_KP_PLUS;
  down.key.scancode = SDL_SCANCODE_KP_PLUS;
  const auto more = input_of(down, keys);
  REQUIRE(more);
  CHECK(std::get<SubtitleDelayStep>(*more).delta_ms == 100);
  // Sans Ctrl, la touche n'est pas le délai audio.
  down.key.mod = SDL_KMOD_NONE;
  const auto plain = input_of(down, keys);
  CHECK((!plain || !std::holds_alternative<SubtitleDelayStep>(*plain)));
}

TEST_CASE("Touches : core::fixed_key dit exactement les touches lues avant [keys] (A24-06)", "[app][input]") {
  // Sans aucun raccourci, une touche seule ne fait quelque chose que si le lecteur la garde.
  const core::KeysRead none = core::read_keys(
      "[keys]\npause = \"\"\nseek_backward = \"\"\nseek_forward = \"\"\nstep_forward = \"\"\n"
      "step_backward = \"\"\nprocessing_off = \"\"\ntoggle_grade = \"\"\ntoggle_interpolation = \"\"\n"
      "toggle_osd = \"\"\n");
  REQUIRE(none.ignored.empty());
  int fixed = 0;
  for (SDL_Keycode code = SDLK_BACKSPACE; code <= SDLK_UP; ++code) {
    if (code == SDLK_DELETE + 1) code = SDLK_CAPSLOCK;  // des caractères aux touches sans caractère
    const auto key = key_of(code);
    if (!key) continue;
    SDL_Event down{};
    down.type = SDL_EVENT_KEY_DOWN;
    down.key.key = code;
    INFO("code SDL " << code);
    CHECK(input_of(down, none.map).has_value() == core::fixed_key(*key));
    if (core::fixed_key(*key)) ++fixed;
  }
  CHECK(fixed == 15);
}

}  // namespace ajp::app::play
