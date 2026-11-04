// Le planificateur de fond du grade du contrat v2 (chantier 8.6) : un fil de faible priorité (nice 10) cuit
// la LUT de la courbe de chaque plan (render::plan_curve, puis render::bake avec le look SDR avant la courbe)
// pour le plan courant et les deux suivants, garde les huit dernières (LRU), et rend des ordres datés du
// début de leur plan : la bascule se fait à l'image près, et la recomposition de la pile quitte le thread
// d'affichage. Un fondu reçoit deux LUT (début et fin), mêlées par pts. En direct, le décalage E choisi par
// l'utilisateur ne bouge que de 0,25 diaphragme par seconde au plus.
//
// GradePlan est le cœur, sans fil (les tests le pilotent) ; GradePlanner le fait tourner sur son fil.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "media/shot_cutter.h"
#include "render/cube_lut.h"
#include "render/curve.h"
#include "render/display_model.h"

namespace ajp::app::play {

inline constexpr std::size_t kPlanAhead = 2;     // plans cuits après le courant
inline constexpr std::size_t kPlanCache = 8;     // LUT gardées (LRU)
inline constexpr double kLiveEPerSecond = 0.25;  // vitesse du décalage E en direct (diaphragmes/s)
inline constexpr double kEStep = 1.0 / 16.0;     // pas de E des clés du cache (4 cuissons/s en direct)
inline constexpr double kAplBudgetShare = 0.9;   // l'APL affichée tient sous 0,9 × MaxFALL (8.11)
static_assert(media::kShotLumBins == render::kCurveHistBins);

struct PlannerConfig {
  int lut_size = 65;
  double vibrance = 1.0;
};

// Un ordre du grade : la LUT (size³ × 3, indexée par √code PQ) qui vaut pour les images de pts dans
// [from_pts_us, until_pts_us). fade_to : la LUT de la fin d'un fondu, mêlée à `lut` par
// (pts − from) / (until − from) ; nulle hors fondu.
struct GradePlanOrder {
  std::int64_t from_pts_us = 0;
  std::int64_t until_pts_us = 0;
  int size = 0;
  std::shared_ptr<const std::vector<float>> lut;
  std::shared_ptr<const std::vector<float>> fade_to;
  render::ShotCurve curve;
  double e = 0.0;  // le décalage de l'utilisateur appliqué à la cuisson
};

// Une cuisson à faire : la clé du cache et l'entrée de la courbe.
struct BakeJob {
  struct Key {
    std::int64_t start_us = 0;
    int part = 0;  // 0 : le plan (ou le début d'un fondu), 1 : la fin d'un fondu
    int e_steps = 0;
    std::uint64_t generation = 0;  // contrat, look ou plans changés
    friend bool operator==(const Key&, const Key&) = default;
  };
  Key key;
  render::CurveInput input;
};

class GradePlan {
 public:
  explicit GradePlan(PlannerConfig config = {}) : config_(config) {}

  void set_shots(std::vector<media::Shot> shots);
  void set_contract(const render::DisplayContract& contract);
  void set_look(std::shared_ptr<const render::CubeLut> look);
  // Le décalage choisi (diaphragmes, ±kCurveMaxE) : appliqué peu à peu (take_orders).
  void set_user_e(double e);
  // Une nouvelle époque (saut) : tous les ordres se redonnent, les LUT cuites restent.
  void restart() { given_.clear(); }
  bool has_shots() const { return !shots_.empty(); }

  // Les cuissons qui manquent pour le plan de pts et les kPlanAhead suivants, dans l'ordre.
  std::vector<BakeJob> wanted(std::int64_t pts_us) const;
  // La cuisson d'un job (lourde, sans état : le fil la fait hors verrou), puis son rangement.
  static std::shared_ptr<const std::vector<float>> bake(const BakeJob& job, const render::CubeLut* look,
                                                        int size);
  void store(const BakeJob::Key& key, std::shared_ptr<const std::vector<float>> lut);

  // Les ordres prêts et pas encore rendus, dans l'ordre des plans (le premier manquant arrête la suite : un
  // ordre ne passe jamais devant celui d'un plan plus tôt). now_us fait avancer E ; un E appliqué qui
  // change relance les ordres à partir de pts.
  std::vector<GradePlanOrder> take_orders(std::int64_t pts_us, std::int64_t now_us);

  double applied_e() const { return applied_e_; }
  std::size_t cached() const { return cache_.size(); }
  const render::CubeLut* look() const { return look_.get(); }
  std::shared_ptr<const render::CubeLut> look_shared() const { return look_; }
  int lut_size() const { return config_.lut_size; }

  // L'entrée de la courbe d'un plan : APL `apl_nits` (le plan, ou une extrémité d'un fondu), E de
  // l'utilisateur.
  render::CurveInput input_of(const media::Shot& shot, double apl_nits, double user_e) const;

 private:
  std::optional<std::size_t> shot_at(std::int64_t pts_us) const;
  BakeJob job_of(std::size_t shot, int part) const;
  std::shared_ptr<const std::vector<float>> find(const BakeJob::Key& key);

  PlannerConfig config_;
  std::vector<media::Shot> shots_;
  double series_apl_nits_ = 0.0;
  render::DisplayContract contract_;
  std::shared_ptr<const render::CubeLut> look_;
  std::uint64_t generation_ = 1;
  double target_e_ = 0.0;
  double applied_e_ = 0.0;
  std::optional<std::int64_t> e_moved_at_us_;
  struct Entry {
    BakeJob::Key key;
    std::shared_ptr<const std::vector<float>> lut;
  };
  std::list<Entry> cache_;  // le plus récent en tête
  // Les ordres rendus : la clé de LUT de chaque plan déjà donné (début du plan), pour ne rien redonner.
  std::vector<std::pair<std::int64_t, BakeJob::Key>> given_;
};

// Le fil de fond autour de GradePlan.
class GradePlanner {
 public:
  explicit GradePlanner(PlannerConfig config = {});
  ~GradePlanner();
  GradePlanner(const GradePlanner&) = delete;
  GradePlanner& operator=(const GradePlanner&) = delete;

  void set_shots(std::vector<media::Shot> shots);
  void set_contract(const render::DisplayContract& contract);
  void set_look(std::shared_ptr<const render::CubeLut> look);
  void set_user_e(double e);
  void restart();
  bool has_shots() const;
  // main, à chaque image : la position de lecture (le fil cuit autour) et les ordres prêts.
  std::vector<GradePlanOrder> take_orders(std::int64_t pts_us, std::int64_t now_us);
  std::size_t baked() const;

 private:
  void run(const std::stop_token& stop);

  mutable std::mutex mutex_;
  std::condition_variable_any wake_;
  GradePlan plan_;
  std::int64_t position_us_ = 0;
  std::size_t baked_ = 0;
  std::jthread thread_;
};

}  // namespace ajp::app::play
