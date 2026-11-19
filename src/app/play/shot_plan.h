// L'Auto plan par plan (--shot-plan, docs/auto-par-plan.md) : le rythme de chaque plan, lu en avance par la
// pré-analyse (media/shot_rhythm.h), choisit son moteur d'interpolation et sa stabilisation temporelle, au
// lieu d'un seul choix pour toute la vidéo. DistilDRBA sur l'animation limitée (dessins tenus en deux, en
// trois, plans de dialogue), RIFE sur le sakuga et l'animation en un ; TemporalFix seulement sur les plans
// au grain visible sur de grandes zones immobiles, là où SPAN scintille.
//
// Jamais de bascule au milieu d'un plan : le choix change à une vraie coupe (un plan qui commence sans coupe
// connue, au point de reprise d'une analyse, garde celui d'avant), et un plan trop court pour se juger garde
// aussi le choix d'avant. L'étage gpu (upscale_shot.cpp) lit le choix de l'image qui arrive.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "media/shot_cutter.h"

namespace kirari::app::play {

// Seuils du choix. Non mesurés : la séance du PC (docs/auto-par-plan.md) les règle.
namespace shot_plan {
inline constexpr std::uint32_t kMinPairs = 6;  // en dessous, le plan garde le choix d'avant
// Images par dessin à partir desquelles un plan est de l'animation limitée (en deux : 2).
inline constexpr float kLimitedCadence = 1.6F;
inline constexpr float kSakugaShare = 0.15F;  // part de paires fortes d'un plan de sakuga
inline constexpr float kFlickerStill = 0.3F;  // part moyenne de blocs immobiles d'un plan qui scintille
inline constexpr float kFlickerGrain = 1.0F;  // et grain moyen de ces blocs, codes 8 bits
}  // namespace shot_plan

enum class ShotMotion : std::uint8_t { Rife, Drba };

struct ShotChoice {
  ShotMotion motion = ShotMotion::Rife;
  bool temporal = true;  // TemporalFix sur ce plan (s'il est chargé)
  bool judged = true;    // faux : hérité du plan d'avant (trop court, ou sans coupe devant)
  friend bool operator==(const ShotChoice&, const ShotChoice&) = default;
};

// Le choix d'un plan jugé sur son rythme.
ShotChoice judge_shot(const media::ShotRhythm& rhythm);
// Les choix de plans consécutifs, dans l'ordre. Le premier plan trop court ou sans coupe prend first.
std::vector<ShotChoice> plan_shots(std::span<const media::Shot> shots, ShotChoice first = {});
// « en deux (2,1 images par dessin), 4 % de paires fortes, grain 0,8 sur 62 % immobile » pour le journal.
std::string rhythm_text(const media::ShotRhythm& rhythm);

// Ce que l'étage gpu joue (upscale_shot.cpp) : le moteur en cours et les comptes du journal de fin.
struct ShotPlay {
  bool drba = false;          // DistilDRBA sur le plan en cours, sinon RIFE
  std::uint64_t to_drba = 0;  // bascules, aux coupes
  std::uint64_t to_rife = 0;
  std::uint64_t held_back = 0;  // un autre moteur voulu sans coupe connue devant : gardé
  std::uint64_t drba_frames = 0;
  std::uint64_t rife_frames = 0;
  std::uint64_t unknown_frames = 0;   // images d'un plan pas encore analysé
  std::uint64_t temporal_frames = 0;  // images passées par TemporalFix, et sautées par le plan
  std::uint64_t temporal_skipped = 0;
};

// L'étage gpu sous l'Auto plan par plan (upscale_shot.cpp, upscale_stage.h) : ses moteurs d'interpolation
// comptés (DistilDRBA et RIFE pour un), et le refus d'une forme de chaîne que la bascule ne suit pas.
struct UpscaleSetup;
int estimators_of(const UpscaleSetup& setup);
std::expected<void, Refusal> check_by_shot(const UpscaleSetup& setup);

// Les plans et leurs choix, partagés entre main (qui les pose quand la chronologie change) et le fil du GPU
// (qui lit le choix de chaque image). Vide : aucun choix, l'étage garde son moteur.
class ShotPlanBook {
 public:
  void set_shots(std::shared_ptr<const std::vector<media::Shot>> shots);
  // Le choix du plan qui contient pts_us, s'il est analysé.
  std::optional<ShotChoice> at(std::int64_t pts_us) const;
  // Les plans disent-ils une coupe entre a et b (cut_book.h, known_cut) ? Rien s'ils ne savent pas.
  std::optional<bool> cut_between(std::int64_t a_us, std::int64_t b_us) const;
  bool active() const;
  // Le rythme du plan qui contient pts_us (rhythm_text), pour le journal ; vide s'il n'est pas analysé.
  std::string describe(std::int64_t pts_us) const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const std::vector<media::Shot>> shots_;
  std::vector<ShotChoice> choices_;
};

}  // namespace kirari::app::play
