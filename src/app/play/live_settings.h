// Réglages que main change pendant la lecture et que le thread gpu lit à chaque image (facteur
// RIFE, filtre de réduction), et les compteurs de RIFE que main lit pour les
// statistiques. Tout est atomique ou sous un verrou court : ni main ni gpu n'attendent l'autre
// plus que la copie d'un réglage.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "app/play/cut_book.h"
#include "app/play/frame_latency.h"
#include "app/play/settings.h"
#include "core/auto_rate.h"
#include "core/free_rate.h"
#include "core/steady_rate.h"
#include "core/vblank_clock.h"
#include "core/vsync_phase.h"
#include "gpu/kernel_ops.h"
#include "media/live_scenes.h"

namespace ajp::app::play {

// Ce que vaut une paire du mode Rapide réduite à une image (core/free_rate.h).
enum class LonePair : std::uint8_t {
  Source,        // la source d'une paire sans interpolation (coupe, dessin tenu, copie) : ne dit rien
  Interpolated,  // une image interpolée : son coût se mesure
  Starved,       // la source d'une paire interpolée où l'écart n'a laissé qu'elle
};

struct RifeCounts {
  std::uint64_t pairs = 0;
  std::uint64_t interpolated = 0;
  std::uint64_t cuts = 0;
  std::uint64_t held = 0;
  std::uint64_t late = 0;  // non interpolées pour rattraper l'horloge
};

// Le filtre de réduction du noyau d'ingestion pour un choix du menu. Spline36 : celui
// d'AnimeJaNai (3 lobes) ; Bicubic : B = 1, C = 0, comme
// le « bicubic » de libplacebo et de mpv ; Mitchell : B = C = 1/3 ; Catmull-Rom : B = 0, C = 1/2 ; Hermite :
// B = C = 0.
inline gpu::Reduction reduction_of(Downscale filter) {
  switch (filter) {
    case Downscale::Area:
      return {};
    case Downscale::Spline36:
      return {.kind = gpu::ReduceKind::Spline36};
    case Downscale::Bilinear:
      return {.kind = gpu::ReduceKind::Triangle};
    case Downscale::Bicubic:
      return {.kind = gpu::ReduceKind::Cubic, .b = 1.0F, .c = 0.0F};
    case Downscale::Mitchell:
      return {.kind = gpu::ReduceKind::Cubic, .b = 1.0F / 3.0F, .c = 1.0F / 3.0F};
    case Downscale::CatmullRom:
      return {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.5F};
    case Downscale::Hermite:  // cubique B = C = 0 : net et sans halo, pour l'animation
      return {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.0F};
  }
  return {};
}

// Les 32 marches quasi noires de la source (8 bits limité : la marche k vaut k/219 de la plage), en niveaux
// 0 à 1.
inline std::array<float, render::kBlackRampSteps> black_ramp_levels() {
  std::array<float, render::kBlackRampSteps> levels{};
  for (std::size_t k = 0; k < levels.size(); ++k) levels[k] = static_cast<float>(k) / 219.0F;
  return levels;
}

class LiveSettings {
 public:
  // Les coupes des plans de la pré-analyse (cut_book.h, --shot-cuts) : main écrit, le fil du GPU lit.
  CutBook& cuts() { return cut_book_; }
  const CutBook& cuts() const { return cut_book_; }
  // ×num/den ; num ≤ den : aucune interpolation (le moteur reste chargé). Les deux termes dans un
  // seul atomique : jamais l'un sans l'autre.
  void set_rife_factor(Factor factor) {
    rife_factor_.store((static_cast<std::uint32_t>(std::max(factor.num, 0)) << 8U) |
                       static_cast<std::uint32_t>(std::clamp(factor.den, 1, 255)));
  }
  Factor rife_factor() const {
    const std::uint32_t packed = rife_factor_.load();
    return {.num = static_cast<int>(packed >> 8U), .den = static_cast<int>(packed & 0xffU)};
  }

  // Mode auto : le multiplicateur ×k suit ce que le GPU tient (core/auto_rate.h). Quand il est
  // actif, factor_now() remplace rife_factor(). `ceiling` : le palier le plus haut que l'écran permet.
  // `granularity` : core::kFineGranularity sur un écran VRR armé (cadence au fps près), 1 sinon ; `ceiling`
  // se compte alors en 1 / granularity (core::auto_ceiling).
  void set_auto(bool on, int ceiling, int granularity = 1) {
    const std::scoped_lock lock(mutex_);
    auto_on_ = on;
    auto_.set_ceiling(ceiling, granularity);
  }
  // P3.33, P3.36 : plafond de l'écran et du niveau, paliers calés à fréquence fixe (core::auto_limits).
  void set_auto(bool on, core::AutoLimits limits) {
    const std::scoped_lock lock(mutex_);
    auto_on_ = on;
    auto_.set_ceiling(limits.ceiling, limits.granularity);
    auto_.set_locked(std::move(limits.locked), std::move(limits.exact));
  }
  bool auto_active() const {
    const std::scoped_lock lock(mutex_);
    return auto_on_;
  }
  int auto_granularity() const {
    const std::scoped_lock lock(mutex_);
    return auto_.granularity();
  }
  int auto_ceiling() const {
    const std::scoped_lock lock(mutex_);
    return auto_.ceiling();
  }
  // gpu : une paire préparée.
  void auto_pair(const core::PairOutcome& outcome) {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.pair(outcome);
  }
  // gpu : une paire sans verdict (coupe de scène, dessin tenu) ; le moteur vit.
  void auto_alive(std::int64_t now_us) {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.alive(now_us);
  }
  void auto_cut(std::int64_t now_us) {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.scene_cut(now_us);
  }
  // Seek, saut d'époque, (re)chargement : la mesure du mode auto repart.
  void auto_settle(std::int64_t now_us) {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.settle(now_us);
    if (free_on_) free_.settle(now_us);
    if (steady_on_) steady_.settle(now_us);
  }
  // gpu : la paire de l'image `id` crée `images` images ; `lead_us` : l'attente de ses mesures avant
  // soumission. main la retrouve par auto_written pour en tirer le coût de service.
  // `heavy` : une paire en mouvement fort (mode Rapide : son coût se mesure à part, core/free_rate.h).
  // `lone` : ce que vaut la paire si elle n'a qu'une image (mode Rapide, A12-12).
  void auto_expect(std::uint32_t id, int images, std::int64_t lead_us, bool heavy = false,
                   LonePair lone = LonePair::Source) {
    const std::scoped_lock lock(mutex_);
    if (!auto_on_ && !free_on_) return;
    ledger_[ledger_next_++ % ledger_.size()] =
        Expected{.id = id, .images = images, .lead_us = lead_us, .heavy = heavy, .lone = lone};
  }
  // gpu : `count` images de la paire de l'image `id` ne seront pas écrites (déjà à l'écran). Si les autres
  // sont toutes vues, la paire se clôt ici.
  void auto_unexpect(std::uint32_t id, int count) {
    const std::scoped_lock lock(mutex_);
    if (!auto_on_ && !free_on_) return;
    for (Expected& e : ledger_) {
      if (e.id != id || e.images == 0) continue;
      e.images = std::max(e.images - count, 0);
      if (e.images == 0) {
        e = Expected{};
      } else if (free_on_ && e.seen >= e.images) {
        close_expected(e, frame_done_us_);
      }
      return;
    }
  }
  // main : une image de sortie vue écrite. La dernière d'une paire (left == right) clôt sa mesure.
  // repeated : une copie d'une image déjà calculée, qui avance la paire sans ajouter de temps occupé.
  void auto_written(std::uint32_t left, std::uint32_t right, std::int64_t submitted_us, std::int64_t done_us,
                    bool repeated = false) {
    const std::scoped_lock lock(mutex_);
    if (!auto_on_ && !free_on_) return;
    // Le temps propre de cette image : de sa soumission (ou de la fin de la précédente, si le GPU était
    // pris) à sa fin vue. L'attente d'un slot du ring avant la soumission n'y compte pas.
    const std::int64_t service = std::max<std::int64_t>(done_us - std::max(submitted_us, frame_done_us_), 0);
    frame_done_us_ = done_us;
    for (Expected& e : ledger_) {
      if (e.id != right || e.images == 0) continue;
      if (e.first_us == 0 || submitted_us < e.first_us) e.first_us = submitted_us;
      if (repeated) {
        ++e.repeats;
      } else {
        e.busy_us += service;
      }
      // Auto : la source clôt la paire. mode Rapide : la dernière image de la paire, qui n'est pas forcément
      // la source.
      const bool last = free_on_ ? ++e.seen >= e.images : left == right;
      if (last) close_expected(e, done_us);
      return;
    }
  }
  // main : images manquées à l'écran depuis le dernier appel ; watchdog (playing : lecture en cours).
  void auto_missed(std::int64_t now_us, std::int64_t delta) {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.missed(now_us, delta);
  }
  void auto_tick(std::int64_t now_us, bool playing) {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.tick(now_us, playing);
  }
  // Départ au dernier palier tenu (en centièmes de multiplicateur).
  void auto_seed(int centi, std::int64_t now_us) {
    const std::scoped_lock lock(mutex_);
    auto_.seed(centi, now_us);
  }
  core::AutoInfo auto_info(std::int64_t now_us) const {
    const std::scoped_lock lock(mutex_);
    return auto_.info(now_us);
  }
  // Le facteur à appliquer maintenant : le palier auto, ou le facteur réglé.
  Factor factor_now() const {
    const std::scoped_lock lock(mutex_);
    if (!auto_on_) return rife_factor();
    // À fréquence fixe, la fraction exacte du palier calé (11011/3200 à 165 Hz, core::AutoLimits::exact).
    const core::Ratio exact = auto_.factor();
    return Factor{.num = static_cast<int>(exact.num), .den = static_cast<int>(exact.den)};
  }

  // mode Rapide (core/free_rate.h, core/free_grid.h) : à la place du mode auto, jamais avec lui. `min_gap_us`
  // : l'écart minimal entre deux images (période de l'écran ou du plafond), en temps réel.
  void set_free(bool on, std::int64_t min_gap_us, std::int64_t now_us) {
    const std::scoped_lock lock(mutex_);
    if (on && !free_on_) free_.reset(now_us);
    free_on_ = on;
    free_.set_min_gap(min_gap_us);
  }
  bool free_active() const {
    const std::scoped_lock lock(mutex_);
    return free_on_;
  }
  // Rapide à pas constant (--fast-steady, chantier 7.7, core/steady_rate.h) : dans le mode Rapide, N images
  // par source, constant dans un plan, à la place de la grille libre. `display` : l'écran (VRR ou fréquence
  // fixe, sa fréquence ou le plafond) ; sans a priori, N part de 1 et s'amorce sur les premières paires.
  // `prior` : la droite a priori t(n) = a + b·n (banc, profil) ; nulle : l'amorçage sur les premières paires.
  void set_steady(bool on, const core::FreeDisplay& display, double source_fps, core::CostLine prior = {}) {
    const std::scoped_lock lock(mutex_);
    if (on && (!steady_on_ || display.vrr != steady_display_.vrr || display.hz != steady_display_.hz ||
               display.cap_hz != steady_display_.cap_hz || source_fps != steady_fps_)) {
      steady_.configure({}, display, source_fps);
      if (prior.a > 0.0 || prior.b > 0.0) steady_.prior(prior);
      steady_display_ = display;
      steady_fps_ = source_fps;
    }
    steady_on_ = on;
  }
  bool steady_active() const {
    const std::scoped_lock lock(mutex_);
    return free_on_ && steady_on_;
  }
  // gpu : N pour la paire préparée maintenant.
  core::Ratio steady_factor() const {
    const std::scoped_lock lock(mutex_);
    return steady_on_ ? steady_.factor() : core::Ratio{1, 1};
  }
  // gpu : une coupe de scène vue sur la paire ; N repart de l'a priori corrigé.
  void free_cut(std::int64_t now_us) {
    const std::scoped_lock lock(mutex_);
    if (free_on_ && steady_on_) steady_.scene_cut(now_us);
  }
  core::SteadyInfo steady_info() const {
    const std::scoped_lock lock(mutex_);
    return steady_.info();
  }
  // gpu : les images planifiées par la paire préparée.
  void free_produced(int images, std::int64_t now_us) {
    const std::scoped_lock lock(mutex_);
    if (free_on_) free_.produced(images, now_us);
    if (free_on_ && steady_on_) steady_.produced(images, now_us);
  }
  // main : images sautées ou manquées à l'écran depuis le dernier appel.
  void free_missed(std::int64_t now_us, std::int64_t lost) {
    const std::scoped_lock lock(mutex_);
    if (free_on_) free_.missed(now_us, lost);
    if (free_on_ && steady_on_ && lost > 0)
      steady_.lost(now_us, static_cast<int>(std::min<std::int64_t>(lost, 1000)));
  }
  // gpu : l'écart de service de la prochaine paire, en temps réel (0 : pas de mesure), et l'écart minimal.
  // `heavy` : la paire est annoncée en mouvement fort.
  std::pair<std::int64_t, std::int64_t> free_step(std::int64_t now_us, bool heavy = false) const {
    const std::scoped_lock lock(mutex_);
    return {free_.step_us(now_us, heavy), free_.min_gap_us()};
  }
  // mode Rapide sans VRR : les instants tombent sur les vsyncs (core/free_grid.h). `period_us` : la période
  // de l'écran (0 : VRR, aucune grille) ; main donne chaque présentation neuve (vsync_note) et le délai entre
  // le vsync de présentation et l'affichage de l'image (Session::visible_delay_us).
  void set_vsync(std::int64_t period_us, std::int64_t visible_delay_us) {
    const std::scoped_lock lock(mutex_);
    vsync_.set_period(period_us);
    vsync_delay_us_ = visible_delay_us;
  }
  std::int64_t vsync_period_us() const {
    const std::scoped_lock lock(mutex_);
    return free_on_ ? vsync_.period_us() : 0;
  }
  // measured : la grille des vblanks du retour de présentation (Screen::vblank_grid, 6.8), qui prime sur la
  // phase estimée des présentations.
  // `display_gpu_us` (7.3) : le temps GPU de toutes les passes de l'écran pour cette image (0 : inconnu),
  // le d du Rapide à pas constant, lissé à 0,1 (core/steady_rate.h).
  // `vsync_us` : un vsync vu à cette présentation (ScreenTiming::vsync_us) ; 0 : aucun, la phase estimée ne
  // bouge pas.
  void vsync_note(std::int64_t vsync_us, std::optional<core::VblankGrid> measured = std::nullopt,
                  std::int64_t display_gpu_us = 0) {
    const std::scoped_lock lock(mutex_);
    if (free_on_ && vsync_us > 0) vsync_.note(vsync_us);
    measured_ = measured;
    if (steady_on_ && display_gpu_us > 0) steady_.display_cost(display_gpu_us);
  }
  // gpu : la période de l'écran et un instant hôte où l'image choisie au vsync le plus proche de `host_us`
  // est celle dont le pts tombe pile sur la grille (période 0 : pas de grille ou phase inconnue).
  struct VsyncGrid {
    std::int64_t period_us = 0;
    std::int64_t anchor_host_us = 0;
  };
  VsyncGrid vsync_grid(std::int64_t host_us) const {
    const std::scoped_lock lock(mutex_);
    if (!free_on_ || vsync_.period_us() <= 0) return {};
    if (measured_ && measured_->period_us > 0.0) {
      // Les dates de la grille sont celles de l'affichage : le délai de présentation y est déjà.
      const double index = std::round(measured_->index_of(static_cast<double>(host_us)));
      const auto period = std::llround(measured_->period_us);
      return {.period_us = period,
              .anchor_host_us = std::llround(measured_->at(static_cast<std::int64_t>(index))) + period / 2};
    }
    if (!vsync_.fresh(host_us)) return {};
    // Choisie une demi-période après son pts, affichée au vsync d'après : pts = vsync + délai + période / 2.
    return {.period_us = vsync_.period_us(),
            .anchor_host_us = vsync_.nearest(host_us) + vsync_delay_us_ + vsync_.period_us() / 2};
  }
  bool free_should_fall_back(std::int64_t now_us) const {
    const std::scoped_lock lock(mutex_);
    if (free_on_ && steady_on_) return steady_.should_fall_back(now_us);
    return free_on_ && free_.should_fall_back(now_us);
  }
  core::FreeInfo free_info(std::int64_t now_us) const {
    const std::scoped_lock lock(mutex_);
    return free_.info(now_us);
  }

  // Garder les poses (kernels/spec/poses.h), lu par gpu au début de chaque paire.
  void set_keep_poses(bool keep) { keep_poses_.store(keep); }
  bool keep_poses() const { return keep_poses_.load(); }

  void set_reduction(const gpu::Reduction& reduction) {
    const std::scoped_lock lock(mutex_);
    reduction_ = reduction;
  }
  gpu::Reduction reduction() const {
    const std::scoped_lock lock(mutex_);
    return reduction_;
  }

  // Compteurs de RIFE, écrits par gpu.
  void count_pair() { pairs_.fetch_add(1, std::memory_order_relaxed); }
  void count_interpolated() { interpolated_.fetch_add(1, std::memory_order_relaxed); }
  void count_cut() { cuts_.fetch_add(1, std::memory_order_relaxed); }
  void count_held() { held_.fetch_add(1, std::memory_order_relaxed); }
  void count_late() { late_.fetch_add(1, std::memory_order_relaxed); }
  // Images source sautées par gpu avant tout calcul, déjà passées à leur tour (RingStage::passed).
  void count_skipped() { skipped_.fetch_add(1, std::memory_order_relaxed); }
  std::uint64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
  // Coupes de scène vues par RIFE sur les images (tant qu'il interpole) : la pré-analyse des plans s'y
  // recale.
  std::uint64_t scene_cuts() const { return cuts_.load(std::memory_order_relaxed); }
  // Le grade tourne à l'écran (posé par main avec lui) : gpu débande alors les sources avant SPAN, une fois
  // par source à sa taille, au lieu que l'écran le fasse sur chaque image agrandie.
  // P4.2 : la fenêtre ne se voit pas ; le fil du GPU n'interpole plus.
  void set_hidden(bool hidden) { hidden_.store(hidden, std::memory_order_relaxed); }
  bool hidden() const { return hidden_.load(std::memory_order_relaxed); }
  void set_deband_source(bool on) { deband_source_.store(on, std::memory_order_relaxed); }
  bool deband_source() const { return deband_source_.load(std::memory_order_relaxed); }
  // Retard des images de sortie, mesuré par main (Session), lu par gpu.
  FrameLatency& latency() { return latency_; }
  const FrameLatency& latency() const { return latency_; }
  RifeCounts counts() const {
    return RifeCounts{.pairs = pairs_.load(std::memory_order_relaxed),
                      .interpolated = interpolated_.load(std::memory_order_relaxed),
                      .cuts = cuts_.load(std::memory_order_relaxed),
                      .held = held_.load(std::memory_order_relaxed),
                      .late = late_.load(std::memory_order_relaxed)};
  }

  // Les scènes d'après l'analyse des images à l'ingestion (analyse v2) : gpu les écrit, main les lit
  // tant que la pré-analyse des plans ne connaît pas l'image.
  media::LiveScenes& scenes() { return scenes_; }
  const media::LiveScenes& scenes() const { return scenes_; }

 private:
  CutBook cut_book_;
  media::LiveScenes scenes_;
  std::atomic<std::uint32_t> rife_factor_{1};  // ×0/1
  std::atomic<bool> keep_poses_{false};
  std::atomic<bool> deband_source_{false};
  std::atomic<bool> hidden_{false};
  mutable std::mutex mutex_;
  gpu::Reduction reduction_;
  std::atomic<std::uint64_t> pairs_{0};
  std::atomic<std::uint64_t> interpolated_{0};
  std::atomic<std::uint64_t> cuts_{0};
  std::atomic<std::uint64_t> held_{0};
  std::atomic<std::uint64_t> late_{0};
  std::atomic<std::uint64_t> skipped_{0};
  FrameLatency latency_;
  bool auto_on_ = false;
  core::AutoRate auto_;
  bool free_on_ = false;
  core::VsyncPhase vsync_;
  std::int64_t vsync_delay_us_ = 0;
  std::optional<core::VblankGrid> measured_;
  core::FreeRate free_;
  bool steady_on_ = false;
  core::SteadyRate steady_;
  core::FreeDisplay steady_display_;
  double steady_fps_ = 0.0;
  std::int64_t frame_done_us_ = 0;
  struct Expected {
    std::uint32_t id = 0;
    int images = 0;  // 0 : case libre
    std::int64_t lead_us = 0;
    std::int64_t first_us = 0;
    int seen = 0;              // images vues écrites (mode Rapide)
    int repeats = 0;           // dont des copies, sans calcul
    std::int64_t busy_us = 0;  // temps propre de ses images, sans l'attente d'un slot
    bool heavy = false;        // mouvement fort
    LonePair lone = LonePair::Source;
  };
  // Clôt la mesure d'une paire : son coût de service, sans les copies. Sous le verrou.
  void close_expected(Expected& e, std::int64_t done_us) {
    const int computed = std::max(e.images - e.repeats, 1);  // les copies ne coûtent rien
    if (auto_on_) {
      auto_.cost(computed, core::service_cost_us(e.first_us, previous_done_us_, done_us, e.lead_us), done_us);
    }
    if (free_on_) {
      // Une image seule ne se mesure que si elle est interpolée ; une paire affamée par l'écart fait
      // redescendre l'estimation, que plus rien ne mesurerait (A12-12).
      const int made = e.images - e.repeats;
      if (made >= 2 || (made == 1 && e.lone == LonePair::Interpolated)) {
        free_.cost(made, e.busy_us + e.lead_us, done_us, e.heavy);
      } else if (e.lone == LonePair::Starved) {
        free_.starved(done_us, e.heavy);
      }
    }
    if (free_on_ && steady_on_) steady_.pair_cost(computed, e.busy_us + e.lead_us, done_us);
    previous_done_us_ = done_us;
    e = Expected{};
  }
  std::array<Expected, 16> ledger_{};
  std::size_t ledger_next_ = 0;
  std::int64_t previous_done_us_ =
      0;  // la fin de la dernière paire mesurée : le début du service de la suivante
};

}  // namespace ajp::app::play
