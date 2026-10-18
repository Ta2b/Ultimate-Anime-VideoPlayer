// Raccourcis clavier : section [keys] du profil, lue par core comme le reste.
//
// Les touches ont ici une représentation neutre, sans dépendance à SDL : app
// traduit ses codes de touche vers Key. Chaque action produit une Command ;
// aucun script ne s'exécute dans le lecteur.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

#include "core/messages.h"
#include "core/types.h"

namespace ajp::core {

// Touche physique. Lettres, chiffres et touches de fonction se suivent, dans l'ordre.
enum class Key : std::uint8_t {
  Space,
  Tab,
  Enter,
  Escape,
  Backspace,
  Delete,
  Insert,
  Home,
  End,
  PageUp,
  PageDown,
  Left,
  Right,
  Up,
  Down,
  Period,
  Comma,
  Minus,
  Equals,
  Slash,
  Backslash,
  Semicolon,
  Apostrophe,
  LeftBracket,
  RightBracket,
  Grave,
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
  P,
  Q,
  R,
  S,
  T,
  U,
  V,
  W,
  X,
  Y,
  Z,
  Digit0,
  Digit1,
  Digit2,
  Digit3,
  Digit4,
  Digit5,
  Digit6,
  Digit7,
  Digit8,
  Digit9,
  F1,
  F2,
  F3,
  F4,
  F5,
  F6,
  F7,
  F8,
  F9,
  F10,
  F11,
  F12,
};

// Une touche et ses modificateurs, écrite « Ctrl+Shift+G ».
struct Chord {
  Key key = Key::Space;
  bool ctrl = false;
  bool alt = false;
  bool shift = false;
  bool meta = false;  // Cmd sous macOS, touche Windows ailleurs
  friend bool operator==(const Chord&, const Chord&) = default;
};

enum class Action : std::uint8_t {
  TogglePause,
  SeekBackward,
  SeekForward,
  StepForward,
  StepBackward,
  Profile1,
  Profile2,
  Profile3,
  Profile4,
  Profile5,
  Profile6,
  Profile7,
  Profile8,
  Profile9,
  ProcessingOff,
  ToggleGrade,
  ToggleInterpolation,
  ToggleOsd,
};
inline constexpr std::size_t kActionCount = static_cast<std::size_t>(Action::ToggleOsd) + 1;

// Pas d'un seek par les flèches : exact, 5 s.
inline constexpr std::int64_t kSeekStepUs = 5'000'000;

// Nom de l'action dans [keys] : « pause », « seek_forward », « profile_3 »…
std::string_view action_name(Action action);
std::optional<Action> parse_action(std::string_view name);

// « Ctrl+G », « Space », « . » ; modificateurs et noms sans égard à la casse.
std::expected<Chord, Refusal> parse_chord(std::string_view text);
// Forme canonique : modificateurs dans l'ordre Ctrl, Alt, Shift, Meta.
std::string format_chord(const Chord& chord);

Command command_for(Action action);

// Un raccourci au plus par action, une action au plus par raccourci.
class KeyMap {
 public:
  // Espace, flèches, « . » et « , », Ctrl+0 à Ctrl+9, Ctrl+G, Ctrl+I, Tab.
  static KeyMap defaults();

  const std::optional<Chord>& chord(Action action) const;
  std::optional<Action> action_for(const Chord& chord) const;
  std::optional<Command> command_for(const Chord& chord) const;

  // Carte où action prend chord (std::nullopt : sans raccourci). Refus KeyConflict
  // si une autre action le porte déjà : rien n'est déplacé en silence.
  std::expected<KeyMap, Refusal> rebind(Action action, std::optional<Chord> chord) const;

  friend bool operator==(const KeyMap&, const KeyMap&) = default;
  friend std::expected<KeyMap, Refusal> parse_keys(std::string_view profile_toml);

 private:
  std::array<std::optional<Chord>, kActionCount> chords_{};
};

// Raccourcis par défaut, redéfinis par la section [keys] du texte TOML d'un profil :
//   [keys]
//   toggle_osd = "F1"
//   pause = ""          # sans raccourci
std::expected<KeyMap, Refusal> parse_keys(std::string_view profile_toml);

}  // namespace ajp::core
