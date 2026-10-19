// Ordonnanceur de présentation : à chaque vsync, quelle image du ring montrer. Logique pure,
// sans GPU ni horloge système : le temps n'entre que par VsyncTick et l'horloge maître.
//
// Trois façons d'avancer, choisies par make_scheduler_config() :
//   - VsyncGrid (écran fixe, Plan calé) : l'écran est maître. Le temps média avance d'un pas
//     fixe par vsync, durée d'une image / k ; chaque image tient donc k vsyncs. Une image en
//     retard fait répéter la précédente, puis la grille rattrape en sautant ce qui est passé :
//     la synchronisation ne glisse jamais.
//   - FixedTimestamps (écran fixe, rien de calé) : l'écran reste maître ; le temps avance de
//     période × vitesse par vsync et l'image dont l'horodatage est atteint passe (3:2…).
//   - Vrr : chaque image à son horodatage sur l'horloge maître (l'audio) ; sans horloge
//     valide, une horloge hôte de repli partie de la première image.
// Une image est due quand son horodatage est atteint à une demi-vsync près ; une image sans
// horodatage (kNoPts) est due tout de suite.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>

#include "core/clock.h"
#include "core/plan.h"
#include "core/resolve.h"
#include "core/ring.h"
#include "core/types.h"
#include "present/scheduler_queue.h"

namespace ajp::present {

enum class PaceMode : std::uint8_t { VsyncGrid, FixedTimestamps, Vrr };

struct SchedulerConfig {
  PaceMode mode = PaceMode::FixedTimestamps;
  int vsyncs_per_frame = 1;      // VsyncGrid : k
  double frame_period_us = 0.0;  // VsyncGrid : durée média d'une image de sortie
  double speed = 1.0;            // vitesse de lecture (CadenceChoice::speed)
  std::uint32_t generation = 0;  // génération du ring en cours
};

namespace scheduler_limits {
inline constexpr core::Range<double> kSpeed{0.25, 4.0};
inline constexpr core::Range<double> kFramePeriodUs{100.0, 10'000'000.0};
}  // namespace scheduler_limits

// Vrr si l'écran l'est ; VsyncGrid si le Plan est calé sur les vsyncs ; FixedTimestamps sinon.
std::expected<SchedulerConfig, Refusal> make_scheduler_config(
    const core::DisplayInfo& display, const std::optional<core::CadenceChoice>& cadence,
    std::uint32_t generation);

// Une vsync où une image va paraître.
struct VsyncTick {
  std::int64_t host_us = 0;  // instant hôte prévu de la vsync (Vrr : de la présentation)
  // Rang de la vsync (VsyncEstimator::index_at) ; écrans fixes. Un rang égal au précédent
  // n'avance pas le temps ; un rang qui recule (estimation recommencée) compte pour une vsync.
  std::int64_t index = 0;
  double period_us = 0.0;  // période mesurée ; 0 : inconnue
};

struct PresentStats {
  std::uint64_t shown = 0;     // images montrées
  std::uint64_t skipped = 0;   // jetées sans être montrées : une plus récente était due
  std::uint64_t repeated = 0;  // vsyncs où l'image devait changer mais la suivante manquait
  std::uint64_t late = 0;      // images montrées après leur vsync
  std::uint64_t stale = 0;     // d'une époque passée ou d'un ring remplacé
};

struct PresentDecision {
  std::optional<RingFrame> frame;  // l'image à composer pour cette vsync
  bool changed = false;            // elle remplace la précédente
  // Slots à rendre à gpu, valables jusqu'au prochain appel de on_vsync().
  std::span<const FrameRelease> releases;
  // Écrans fixes : l'horloge de l'écran à cette vsync, à publier (DisplayClock). Vrr : invalide.
  ClockSnapshot clock;
  // Vrr : instant hôte où la prochaine image en file sera due ; absent si inconnu.
  std::optional<std::int64_t> next_due_host_us;
};

class PresentScheduler {
 public:
  // Refuse k hors de [1, 16], une durée d'image ou une vitesse hors de leurs bornes.
  static std::expected<PresentScheduler, Refusal> create(const SchedulerConfig& config);

  // Une image arrive de gpu. D'un autre ring ou d'une époque passée : rendue sans être montrée
  // (core/ring.h : chaque RingFrame reçu rend son slot). D'une époque future : vaut Flush.
  // Refuse, en la rendant, une image dont l'horodatage ne croît pas dans son époque ou qui ne
  // tient plus dans la file. Refuse sans la rendre, app la rendant elle-même, une image de plus
  // quand l'ordonnanceur en tient déjà ReleaseList::kCapacity (app ne lit plus les décisions).
  std::expected<void, Refusal> push(const RingFrame& frame);

  // Seek ou nouveau Plan. L'image montrée reste jusqu'à la première de la nouvelle époque, qui
  // passe à la vsync suivante et recale l'horloge. Refuse une époque qui recule.
  std::expected<void, Refusal> flush(std::uint32_t epoch);

  void set_paused(bool paused) { paused_ = paused; }
  // En pause : la vsync suivante montre l'image suivante dès qu'elle est en file. Le pas en
  // arrière est un seek d'app.
  void step() { step_pending_ = paused_; }
  // Ring recréé : les images de l'ancien sont rendues, celles en file sans être montrées, et
  // plus rien n'est montré avant la première du nouveau. app ne détruit l'ancien ring qu'après
  // la première image du nouveau, ces attentes achevées côté GPU (core/ring.h).
  void set_generation(std::uint32_t generation);

  PresentDecision on_vsync(const VsyncTick& tick, const ClockSnapshot& master);

  const PresentStats& stats() const { return stats_; }
  bool paused() const { return paused_; }
  std::uint32_t epoch() const { return epoch_; }

 private:
  explicit PresentScheduler(const SchedulerConfig& config) : config_(config) {}

  // VsyncGrid : temps média d'une vsync, durée d'une image / k.
  double vsync_step() const { return config_.frame_period_us / config_.vsyncs_per_frame; }
  double advance_clock(const VsyncTick& tick, const ClockSnapshot& master);
  double tolerance(const VsyncTick& tick) const;
  double clock_rate(const VsyncTick& tick) const;
  bool pick_due(double target, double tolerance);
  void count_underflow(double target, double tolerance);
  void show(const RingFrame& frame);
  void anchor(const RingFrame& frame);
  std::optional<std::int64_t> next_due(const ClockSnapshot& master) const;
  // Images tenues : en file, montrée, et slots à rendre. Un Flush, un ring recréé ou une
  // décision ne font que les déplacer vers releases_ : ce total borne releases_.
  std::size_t held() const { return releases_.size() + queue_.size() + (current_ ? 1U : 0U); }

  SchedulerConfig config_;
  FrameQueue queue_;
  ReleaseList releases_;
  std::optional<RingFrame> current_;
  PresentStats stats_;
  double clock_ = 0.0;  // temps média de la vsync courante, en microsecondes
  std::int64_t last_index_ = 0;
  std::int64_t last_host_us_ = 0;
  std::int64_t last_pushed_pts_ = kNoPts;
  std::int64_t last_interval_us_ = 0;  // écart entre les deux dernières images reçues
  std::uint32_t epoch_ = 0;
  bool anchored_ = false;  // l'horloge part de la première image de l'époque
  bool master_valid_ = false;
  bool paused_ = false;
  bool step_pending_ = false;
};

}  // namespace ajp::present
