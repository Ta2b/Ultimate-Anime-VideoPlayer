#include "app/play/input.h"

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

std::optional<core::Key> key_of(SDL_Keycode key) {
  using core::Key;
  if (key >= SDLK_A && key <= SDLK_Z) return offset(Key::A, key, SDLK_A);
  if (key >= SDLK_0 && key <= SDLK_9) return offset(Key::Digit0, key, SDLK_0);
  if (key >= SDLK_F1 && key <= SDLK_F12) return offset(Key::F1, key, SDLK_F1);
  switch (key) {
    case SDLK_SPACE:
      return Key::Space;
    case SDLK_TAB:
      return Key::Tab;
    case SDLK_RETURN:
      return Key::Enter;
    case SDLK_ESCAPE:
      return Key::Escape;
    case SDLK_BACKSPACE:
      return Key::Backspace;
    case SDLK_DELETE:
      return Key::Delete;
    case SDLK_HOME:
      return Key::Home;
    case SDLK_END:
      return Key::End;
    case SDLK_PAGEUP:
      return Key::PageUp;
    case SDLK_PAGEDOWN:
      return Key::PageDown;
    case SDLK_LEFT:
      return Key::Left;
    case SDLK_RIGHT:
      return Key::Right;
    case SDLK_UP:
      return Key::Up;
    case SDLK_DOWN:
      return Key::Down;
    case SDLK_PERIOD:
      return Key::Period;
    case SDLK_COMMA:
      return Key::Comma;
    default:
      return std::nullopt;
  }
}

std::optional<Input> input_of(const SDL_Event& event, const core::KeyMap& keys) {
  // P3.43 : une touche de seek maintenue saute de point clé en point clé ; au relâchement, un seek exact
  // à la position atteinte. Le thread de la fenêtre seul lit les touches.
  static bool seek_repeated = false;
  if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
    return Command{cmd::Quit{}};
  if (event.type == SDL_EVENT_KEY_UP && std::exchange(seek_repeated, false))
    return Command{cmd::SeekRelative{.delta_us = 0, .exact = true}};
  if (event.type != SDL_EVENT_KEY_DOWN) return std::nullopt;
  if (event.key.repeat) {
    const SDL_Keymod held = event.key.mod;
    if ((held & (SDL_KMOD_CTRL | SDL_KMOD_ALT | SDL_KMOD_GUI)) == 0 &&
        (event.key.key == SDLK_UP || event.key.key == SDLK_DOWN))
      return VolumeStep{.delta = event.key.key == SDLK_UP ? kVolumeStep : -kVolumeStep};
    const auto repeated = key_of(event.key.key);
    if (!repeated) return std::nullopt;
    auto command = keys.command_for(core::Chord{.key = *repeated,
                                                .ctrl = (held & SDL_KMOD_CTRL) != 0,
                                                .alt = (held & SDL_KMOD_ALT) != 0,
                                                .shift = (held & SDL_KMOD_SHIFT) != 0,
                                                .meta = (held & SDL_KMOD_GUI) != 0});
    if (!command) return std::nullopt;
    if (auto* seek = std::get_if<cmd::SeekRelative>(&*command)) {
      seek->exact = false;
      seek_repeated = true;
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
  auto key = key_of(event.key.key);
  bool shift = (mod & SDL_KMOD_SHIFT) != 0;
  // « . » et « , » se lisent par le caractère tapé : en AZERTY, « . » est Maj + « ; », et SDL rend la
  // touche sans Maj. Maj sert alors à produire le caractère, il ne compte plus dans l'accord.
  if (shift && (!key || (*key != core::Key::Period && *key != core::Key::Comma))) {
    const SDL_Keycode typed = SDL_GetKeyFromScancode(event.key.scancode, SDL_KMOD_SHIFT, false);
    if (typed == SDLK_PERIOD || typed == SDLK_COMMA) {
      key = key_of(typed);
      shift = false;
    }
  }
  if (!key) return std::nullopt;
  const core::Chord chord{.key = *key,
                          .ctrl = (mod & SDL_KMOD_CTRL) != 0,
                          .alt = (mod & SDL_KMOD_ALT) != 0,
                          .shift = shift,
                          .meta = (mod & SDL_KMOD_GUI) != 0};
  if (auto command = keys.command_for(chord)) return *command;
  return std::nullopt;
}

}  // namespace ajp::app::play
