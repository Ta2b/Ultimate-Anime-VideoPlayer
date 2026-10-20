#include "app/play/input.h"

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
  if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
    return Command{cmd::Quit{}};
  if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) return std::nullopt;
  const SDL_Keymod mod = event.key.mod;
  const bool plain = (mod & (SDL_KMOD_CTRL | SDL_KMOD_ALT | SDL_KMOD_GUI)) == 0;
  if (plain && event.key.key == SDLK_Q) return Command{cmd::Quit{}};
  if (plain && event.key.key == SDLK_F) return ToggleFullscreen{};
  if (plain && event.key.key == SDLK_ESCAPE) return LeaveFullscreen{};
  if (plain && event.key.key == SDLK_M) return ToggleMute{};
  if (plain && event.key.key == SDLK_J) return CycleSubtitles{};
  if (plain && event.key.key == SDLK_A) return CycleAudio{};
  if (plain && event.key.key == SDLK_P) return ToggleSettings{};
  if (plain && event.key.key == SDLK_PAGEUP) return StepChapter{.direction = 1};
  if (plain && event.key.key == SDLK_PAGEDOWN) return StepChapter{.direction = -1};
  if (plain && event.key.key == SDLK_UP) return VolumeStep{.delta = kVolumeStep};
  if (plain && event.key.key == SDLK_DOWN) return VolumeStep{.delta = -kVolumeStep};
  const auto key = key_of(event.key.key);
  if (!key) return std::nullopt;
  const core::Chord chord{.key = *key,
                          .ctrl = (mod & SDL_KMOD_CTRL) != 0,
                          .alt = (mod & SDL_KMOD_ALT) != 0,
                          .shift = (mod & SDL_KMOD_SHIFT) != 0,
                          .meta = (mod & SDL_KMOD_GUI) != 0};
  if (auto command = keys.command_for(chord)) return *command;
  return std::nullopt;
}

}  // namespace ajp::app::play
