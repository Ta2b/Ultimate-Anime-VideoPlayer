// Messages échangés entre threads par leurs boîtes Inbox. Fait partie de core_types.
#pragma once

#include <cstdint>
#include <memory>
#include <variant>

namespace ajp {

// Le consommateur rend une place : le producteur peut réessayer try_push.
struct Credit {
  std::uint32_t count = 1;
};

// Un slot du ring revient au thread gpu après le sémaphore de fin du rendu.
struct SlotFree {
  std::uint32_t slot = 0;
  std::uint64_t semaphore_value = 0;  // valeur du sémaphore timeline à attendre côté GPU
};

// Vide les files au seek ou au changement de Plan ; les images d'époque < epoch sont jetées.
struct Flush {
  std::uint32_t epoch = 0;
};

// Base opaque d'un Program (moteurs d'un Plan), construit par prepare, exécuté par gpu.
class ProgramBase {
 public:
  virtual ~ProgramBase() = default;
  virtual std::uint64_t plan_hash() const = 0;
};

struct Install {
  std::shared_ptr<const ProgramBase> program;
};

// Commandes de l'utilisateur, routées par app vers les threads concernés.
namespace cmd {
struct Play {};
struct Pause {};
struct TogglePause {};
struct Seek {
  std::int64_t target_us = 0;
  bool exact = true;  // false : point clé le plus proche
};
struct SeekRelative {
  std::int64_t delta_us = 0;
  bool exact = true;
};
struct StepFrame {
  int direction = 1;  // +1 avant, -1 arrière
};
struct SelectProfile {
  int slot = 1;  // 1 à 9
};
struct ProcessingOff {};
struct ToggleGrade {};
struct ToggleInterpolation {};
struct ToggleOsd {};
struct SetAudioDelay {
  std::int64_t delay_us = 0;
};
struct SetVolume {
  float gain = 1.0F;  // 0 à 2
};
struct Quit {};
}  // namespace cmd

using Command =
    std::variant<cmd::Play, cmd::Pause, cmd::TogglePause, cmd::Seek, cmd::SeekRelative, cmd::StepFrame,
                 cmd::SelectProfile, cmd::ProcessingOff, cmd::ToggleGrade, cmd::ToggleInterpolation,
                 cmd::ToggleOsd, cmd::SetAudioDelay, cmd::SetVolume, cmd::Quit>;

}  // namespace ajp
