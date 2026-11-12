#include "app/play/input.h"

#include <array>
#include <utility>
#include <variant>

#include "app/play/controls.h"

namespace ajp::app::play {
namespace {

template <class E>
E offset(E first, SDL_Keycode key, SDL_Keycode base) {
  return static_cast<E>(static_cast<int>(first) + static_cast<int>(key - base));
}

}  // namespace

// Les touches nommées, hors lettres, chiffres et F1 à F12 : une table, chaque touche de core::Key y est.
struct NamedKey {
  SDL_Keycode sdl;
  core::Key key;
};
constexpr std::array kNamedKeys{
    NamedKey{SDLK_SPACE, core::Key::Space},
    NamedKey{SDLK_TAB, core::Key::Tab},
    NamedKey{SDLK_RETURN, core::Key::Enter},
    NamedKey{SDLK_ESCAPE, core::Key::Escape},
    NamedKey{SDLK_BACKSPACE, core::Key::Backspace},
    NamedKey{SDLK_DELETE, core::Key::Delete},
    NamedKey{SDLK_HOME, core::Key::Home},
    NamedKey{SDLK_END, core::Key::End},
    NamedKey{SDLK_PAGEUP, core::Key::PageUp},
    NamedKey{SDLK_PAGEDOWN, core::Key::PageDown},
    NamedKey{SDLK_LEFT, core::Key::Left},
    NamedKey{SDLK_RIGHT, core::Key::Right},
    NamedKey{SDLK_UP, core::Key::Up},
    NamedKey{SDLK_DOWN, core::Key::Down},
    NamedKey{SDLK_PERIOD, core::Key::Period},
    NamedKey{SDLK_COMMA, core::Key::Comma},
    NamedKey{SDLK_INSERT, core::Key::Insert},
    NamedKey{SDLK_MINUS, core::Key::Minus},
    NamedKey{SDLK_EQUALS, core::Key::Equals},
    NamedKey{SDLK_SLASH, core::Key::Slash},
    NamedKey{SDLK_BACKSLASH, core::Key::Backslash},
    NamedKey{SDLK_SEMICOLON, core::Key::Semicolon},
    NamedKey{SDLK_APOSTROPHE, core::Key::Apostrophe},
    NamedKey{SDLK_LEFTBRACKET, core::Key::LeftBracket},
    NamedKey{SDLK_RIGHTBRACKET, core::Key::RightBracket},
    NamedKey{SDLK_GRAVE, core::Key::Grave},
};

std::optional<core::Key> key_of(SDL_Keycode key) {
  using core::Key;
  if (key >= SDLK_A && key <= SDLK_Z) return offset(Key::A, key, SDLK_A);
  if (key >= SDLK_0 && key <= SDLK_9) return offset(Key::Digit0, key, SDLK_0);
  if (key >= SDLK_F1 && key <= SDLK_F12) return offset(Key::F1, key, SDLK_F1);
  // Une boucle, pas un itérateur : celui de std::array est un pointeur ici et une classe sous MSVC.
  for (const NamedKey& named : kNamedKeys)
    if (named.sdl == key) return named.key;
  return std::nullopt;
}

namespace {

// La touche d'un accord. « . » et « , » se lisent par le caractère tapé : en AZERTY, « . » est Maj + « ; »,
// et SDL rend la touche sans Maj. Maj sert alors à produire le caractère, il ne compte plus dans l'accord
// (shift passe à faux). Vaut aussi pour une touche maintenue (A24-05).
std::optional<core::Key> chord_key(const SDL_KeyboardEvent& event, bool& shift) {
  auto key = key_of(event.key);
  if (shift && (!key || (*key != core::Key::Period && *key != core::Key::Comma))) {
    const SDL_Keycode typed = SDL_GetKeyFromScancode(event.scancode, SDL_KMOD_SHIFT, false);
    if (typed == SDLK_PERIOD || typed == SDLK_COMMA) {
      key = key_of(typed);
      shift = false;
    }
  }
  return key;
}

// P3.79 : le délai audio au clavier, Ctrl + « - » et Ctrl + « + » (100 ms), comme mpv. Le caractère de la
// touche sans modificateur compte aussi (A24-05) : en AZERTY, SDL rend « 6 » pour la touche du « - ».
std::optional<Input> audio_delay_step(const SDL_KeyboardEvent& event) {
  if ((event.mod & SDL_KMOD_CTRL) == 0) return std::nullopt;
  const SDL_Keycode typed = SDL_GetKeyFromScancode(event.scancode, SDL_KMOD_NONE, false);
  for (const SDL_Keycode k : {event.key, typed}) {
    if (k == SDLK_MINUS || k == SDLK_KP_MINUS) return SubtitleDelayStep{.delta_ms = -100, .audio = true};
    if (k == SDLK_EQUALS || k == SDLK_PLUS || k == SDLK_KP_PLUS)
      return SubtitleDelayStep{.delta_ms = 100, .audio = true};
  }
  return std::nullopt;
}

// B24-03 : F11 et Alt + Entrée, les deux gestes du bureau pour le plein écran, quand aucune touche de [keys]
// ne les a pris.
std::optional<Input> fullscreen_shortcut(const SDL_KeyboardEvent& event) {
  if ((event.mod & (SDL_KMOD_CTRL | SDL_KMOD_GUI | SDL_KMOD_SHIFT)) != 0) return std::nullopt;
  const bool alt = (event.mod & SDL_KMOD_ALT) != 0;
  if (!alt && event.key == SDLK_F11) return ToggleFullscreen{};
  if (alt && (event.key == SDLK_RETURN || event.key == SDLK_KP_ENTER)) return ToggleFullscreen{};
  return std::nullopt;
}

// C24-05 : la touche qui a lancé la rafale de seeks ; le relâchement d'une autre (Maj, Ctrl) n'y met pas fin.
SDL_Scancode& held_seek_key() {
  static SDL_Scancode key = SDL_SCANCODE_UNKNOWN;
  return key;
}

// Vrai, une fois, au relâchement de la touche de la rafale.
bool seek_key_released(SDL_Scancode released) {
  SDL_Scancode& key = held_seek_key();
  if (key == SDL_SCANCODE_UNKNOWN || key != released) return false;
  key = SDL_SCANCODE_UNKNOWN;
  return true;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::optional<Input> input_of(const SDL_Event& event, const core::KeyMap& keys) {
  // P3.43 : une touche de seek maintenue saute de point clé en point clé ; au relâchement, un seek exact
  // à la position atteinte. Le thread de la fenêtre seul lit les touches.
  if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
    return Command{cmd::Quit{}};
  if (event.type == SDL_EVENT_KEY_UP && seek_key_released(event.key.scancode))
    return Command{cmd::SeekRelative{.delta_us = 0, .exact = true}};
  if (event.type != SDL_EVENT_KEY_DOWN) return std::nullopt;
  if (event.key.repeat) {
    const SDL_Keymod held = event.key.mod;
    if ((held & (SDL_KMOD_CTRL | SDL_KMOD_ALT | SDL_KMOD_GUI)) == 0 &&
        (event.key.key == SDLK_UP || event.key.key == SDLK_DOWN))
      return VolumeStep{.delta = event.key.key == SDLK_UP ? kVolumeStep : -kVolumeStep};
    bool held_shift = (held & SDL_KMOD_SHIFT) != 0;
    const auto repeated = chord_key(event.key, held_shift);
    if (!repeated) return std::nullopt;
    auto command = keys.command_for(core::Chord{.key = *repeated,
                                                .ctrl = (held & SDL_KMOD_CTRL) != 0,
                                                .alt = (held & SDL_KMOD_ALT) != 0,
                                                .shift = held_shift,
                                                .meta = (held & SDL_KMOD_GUI) != 0});
    if (!command) return std::nullopt;
    if (auto* seek = std::get_if<cmd::SeekRelative>(&*command)) {
      seek->exact = false;
      held_seek_key() = event.key.scancode;
      return *command;
    }
    if (std::holds_alternative<cmd::StepFrame>(*command)) return *command;
    return std::nullopt;
  }
  const SDL_Keymod mod = event.key.mod;
  const bool plain = (mod & (SDL_KMOD_CTRL | SDL_KMOD_ALT | SDL_KMOD_GUI)) == 0;
  if (plain && event.key.key == SDLK_Q) return Command{cmd::Quit{}};
  if (plain && event.key.key == SDLK_F) return ToggleFullscreen{};
  if (plain && event.key.key == SDLK_ESCAPE) return LeaveFullscreen{};
  if (plain && event.key.key == SDLK_M) return ToggleMute{};
  if (plain && event.key.key == SDLK_J) return CycleSubtitles{};
  if (plain && (event.key.key == SDLK_Z || event.key.key == SDLK_X))
    return SubtitleDelayStep{.delta_ms = event.key.key == SDLK_Z ? -100 : 100};
  if (auto step = audio_delay_step(event.key)) return step;
  if (plain && event.key.key == SDLK_A) return CycleAudio{};
  if (plain && event.key.key == SDLK_P) return ToggleSettings{};
  if (plain && event.key.key == SDLK_N) {
    return StepEpisode{.direction = (mod & SDL_KMOD_SHIFT) != 0 ? -1 : 1};
  }
  if (plain && event.key.key == SDLK_O) return SkipCredits{};
  if (plain && event.key.key == SDLK_PAGEUP) return StepChapter{.direction = 1};
  if (plain && event.key.key == SDLK_PAGEDOWN) return StepChapter{.direction = -1};
  if (plain && event.key.key == SDLK_UP) return VolumeStep{.delta = kVolumeStep};
  if (plain && event.key.key == SDLK_DOWN) return VolumeStep{.delta = -kVolumeStep};
  bool shift = (mod & SDL_KMOD_SHIFT) != 0;
  const auto key = chord_key(event.key, shift);
  if (!key) return fullscreen_shortcut(event.key);
  const core::Chord chord{.key = *key,
                          .ctrl = (mod & SDL_KMOD_CTRL) != 0,
                          .alt = (mod & SDL_KMOD_ALT) != 0,
                          .shift = shift,
                          .meta = (mod & SDL_KMOD_GUI) != 0};
  if (auto command = keys.command_for(chord)) return *command;
  return fullscreen_shortcut(event.key);
}

}  // namespace ajp::app::play
