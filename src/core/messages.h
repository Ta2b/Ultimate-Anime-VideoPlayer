// Messages échangés entre threads par leurs boîtes Inbox. Fait partie de core_types.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <variant>

namespace ajp {

namespace audio {
struct Hrir;  // réponses impulsionnelles de l'utilisateur (audio/hrir.h)
}

// Le consommateur rend une place : le producteur peut réessayer try_push.
struct Credit {
  std::uint32_t count = 1;
};

// Un slot du ring revient au thread gpu après le rendu qui l'a lu (protocole : core/ring.h).
struct SlotFree {
  std::uint32_t slot = 0;
  std::uint64_t semaphore_value = 0;  // valeur du sémaphore du slot à attendre côté GPU
  std::uint32_t generation = 0;       // génération du ring ; celui d'un ring remplacé est ignoré
};

// Vide les files au seek ou au changement de Plan ; les images d'époque < epoch sont jetées.
struct Flush {
  std::uint32_t epoch = 0;
};

// La source a livré la dernière image de l'époque : un étage qui attend la suivante (DistilDRBA)
// produit alors la dernière sans elle.
struct StreamEnd {
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
// « Comment écoutez-vous ? » (audio/mixer.h) : 0 casque, 1 enceintes, 2 home cinéma ; faithful : aucun
// traitement.
struct SetListening {
  std::uint8_t listen = 1;
  bool faithful = false;
  bool tuned = false;  // tuning porte les réglages fins (audio::MixTuning, dans l'ordre de to_array)
  std::array<float, 16> tuning{};  // audio::to_array : les 15 réglages fins, le reste à 0
  bool hrir_set = false;  // hrir remplace les réponses en vigueur (vide : retour au binaural paramétrique)
  std::shared_ptr<const audio::Hrir> hrir;  // fichier chargé par l'appelant, partagé en lecture seule
};
struct Quit {};
}  // namespace cmd

using Command =
    std::variant<cmd::Play, cmd::Pause, cmd::TogglePause, cmd::Seek, cmd::SeekRelative, cmd::StepFrame,
                 cmd::SelectProfile, cmd::ProcessingOff, cmd::ToggleGrade, cmd::ToggleInterpolation,
                 cmd::ToggleOsd, cmd::SetAudioDelay, cmd::SetVolume, cmd::SetListening, cmd::Quit>;

}  // namespace ajp
