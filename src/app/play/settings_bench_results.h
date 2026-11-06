// Vidéo > Post-traitement > « Résultats du bench » : les chaînes que la calibration de la machine a
// essayées (machine_calibration.h), celle que le lecteur recommande, un bouton Appliquer par ligne, et de
// quoi refaire le bench. Les résultats vivent dans calibration.toml, à côté de play.toml.
#pragma once

#include <atomic>
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "app/play/input.h"
#include "app/play/machine_calibration.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

struct MachineNote {
  bool read = false;                         // calibration.toml relu à l'ouverture de la page
  std::optional<MachineCalibration> result;  // le dernier bench
  std::string problem;                       // le dernier bench refait a échoué : pourquoi
  std::shared_ptr<std::atomic<float>> fraction;
  std::shared_ptr<std::stop_source> stop;
  std::shared_ptr<std::future<std::expected<MachineCalibration, Refusal>>> pending;
  MachineNote() = default;
  MachineNote(const MachineNote&) = delete;
  MachineNote& operator=(const MachineNote&) = delete;
  ~MachineNote();
  bool running() const { return pending != nullptr; }
};

// La section ; d reçoit la chaîne d'une ligne appliquée (rend true : elle s'applique tout de suite, sans
// « Appliquer »). actions reçoit la pause quand le bench repart.
bool bench_results_section(PlaySettings& d, const SettingsChoices& choices, MachineNote& bench,
                           std::vector<Input>& actions);

}  // namespace ajp::app::play
