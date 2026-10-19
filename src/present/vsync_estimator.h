// Période de vsync mesurée depuis les instants où la swapchain FIFO rend la main.
//
// En FIFO, la boucle de présentation avance d'une image par vsync une fois la file pleine :
// l'instant hôte où pl_swapchain_start_frame revient suit donc la grille des vsyncs, à un
// retard près et avec le bruit de l'ordonnanceur du système. Deux filtres :
//   - images manquées : chaque instant reçoit un rang entier de vsync, l'écart au précédent
//     arrondi en périodes (2 périodes : une vsync sautée), et non un rang de plus ;
//   - retards : droite des moindres carrés instant = origine + période · rang, refaite sans
//     les instants dont le résidu dépasse 3 écarts absolus médians (préemption, accroc).
// Fenêtre glissante bornée, aucune allocation : l'estimation se refait à chaque image.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ajp::present {

struct VsyncEstimate {
  double period_us = 0.0;         // période de vsync estimée
  std::int64_t anchor_index = 0;  // rang d'une vsync de référence
  double anchor_host_us = 0.0;    // son instant hôte sur la droite ajustée
  double jitter_us = 0.0;         // écart quadratique des instants gardés à la droite
  std::size_t samples = 0;        // instants gardés par le filtre
  double refresh_hz() const { return 1e6 / period_us; }
};

class VsyncEstimator {
 public:
  static constexpr std::size_t kWindow = 128;     // instants gardés, un peu moins d'une seconde
  static constexpr std::size_t kMinSamples = 16;  // avant toute estimation
  // Résidu minimal toléré : le bruit d'un ordonnanceur calme reste sous 0,1 ms.
  static constexpr double kMinThresholdUs = 100.0;
  // Instants hors de la droite, nets de ceux qui y tiennent, avant de tout recommencer
  // (changement de mode, fenêtre passée sur un autre écran).
  static constexpr int kMaxMisfits = 8;

  // Instant hôte (host_now_us) où la swapchain a rendu une image. Un instant qui recule ou
  // se répète est ignoré.
  void add(std::int64_t host_us);
  void reset();

  // Absente tant que la fenêtre n'a pas kMinSamples instants ou que moins de la moitié
  // tient sur une même droite.
  std::optional<VsyncEstimate> estimate() const { return estimate_; }

  // Rang de la vsync la plus proche de host_us, et instant prévu d'un rang ; estimation requise.
  std::optional<std::int64_t> index_at(std::int64_t host_us) const;
  std::optional<std::int64_t> host_at(std::int64_t index) const;

 private:
  struct Sample {
    std::int64_t index = 0;
    std::int64_t host_us = 0;
  };

  const Sample& at(std::size_t i) const { return samples_[(head_ + i) % kWindow]; }
  Sample& at(std::size_t i) { return samples_[(head_ + i) % kWindow]; }
  void push(const Sample& sample);
  void bootstrap();  // premiers rangs, comptés en intervalles médians
  void refit();
  bool fits(const Sample& sample) const;

  std::array<Sample, kWindow> samples_{};
  std::size_t head_ = 0;
  std::size_t count_ = 0;
  std::optional<VsyncEstimate> estimate_;
  int misfits_ = 0;
  bool bootstrapped_ = false;
};

}  // namespace ajp::present
