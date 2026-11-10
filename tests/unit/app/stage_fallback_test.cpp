// Les étages refusés qui ne coupent plus la lecture (décisions 6 et 10 ; A29-01, A31-04, A08-01, A08-16,
// A31-05, A31-15) : ce que la préparation essaie selon les refus retenus.
#include "app/play/stage_fallback.h"

#include <catch2/catch_test_macros.hpp>

#include "app/play/player_engines.h"

namespace ajp::app::play {
namespace {

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr std::string_view kLibrary = "libajp_backend_tensorrt.so";

PlaySettings chain(PrecisionTier span, PrecisionTier motion) {
  PlaySettings s;
  s.upscale = true;
  s.rife_factor = 2;
  s.span_precision = span;
  s.motion_precision = motion;
  return s;
}

void refuse(StageMemory& memory, ChainStage stage, const PlaySettings& tried, bool cancelled = false,
            const std::filesystem::path& file = "ep01.mkv") {
  Refusal why{RefusalCode::BuilderTimeout, "construction au-delà de 20 min"};
  if (cancelled)
    why = Refusal{.code = RefusalCode::Cancelled, .detail = "annulé", .key = std::string(kWatchNowKey)};
  memory.remember({.stage = stage,
                   .signature = stage_signature(stage, tried, kWidth, kHeight, kLibrary),
                   .refusal = why,
                   .cancelled = cancelled,
                   .file = file});
}

std::optional<StagePlan> plan(const PlaySettings& wanted, const StageMemory& memory, bool allow_drop = true,
                              const std::filesystem::path& file = "ep01.mkv") {
  return plan_stages(wanted, memory, kWidth, kHeight, kLibrary, file, allow_drop);
}

TEST_CASE("Étages : rien de refusé, la chaîne demandée telle quelle", "[app][stages]") {
  const PlaySettings wanted = chain(PrecisionTier::Quality, PrecisionTier::Quality);
  const auto tried = plan(wanted, StageMemory{});
  REQUIRE(tried);
  CHECK(tried->settings == wanted);
  CHECK(tried->notes.empty());
}

TEST_CASE("Étages : RIFE refusé, la lecture continue avec SPAN seul (A29-01, A31-04)", "[app][stages]") {
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  StageMemory memory;
  refuse(memory, ChainStage::Motion, wanted);
  const auto tried = plan(wanted, memory);
  REQUIRE(tried);
  CHECK(tried->settings.upscale);
  CHECK(tried->settings.rife_factor == 0);
  REQUIRE(tried->notes.size() == 1);
  CHECK(tried->notes[0].change == StageChange::Dropped);
  CHECK_FALSE(stage_notice(tried->notes).empty());
  // L'export et le banc ne laissent pas un étage de côté : rien à jouer.
  CHECK_FALSE(plan(wanted, memory, false));
}

TEST_CASE("Étages : SPAN refusé, la source gardée à sa taille, l'interpolation continue", "[app][stages]") {
  PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  wanted.work_height = 720;
  StageMemory memory;
  refuse(memory, ChainStage::Upscale, wanted);
  const auto tried = plan(wanted, memory);
  REQUIRE(tried);
  CHECK_FALSE(tried->settings.upscale);
  CHECK(tried->settings.work_height == 0);  // la source n'est plus réduite pour un SPAN absent
  CHECK(tried->settings.rife_factor == 2);
}

TEST_CASE("Étages : un cran refusé ne fait retomber que son étage (A08-16)", "[app][stages]") {
  const PlaySettings wanted = chain(PrecisionTier::Quality, PrecisionTier::Quality);
  StageMemory memory;
  refuse(memory, ChainStage::Motion, wanted);  // RIFE FP32 refusé
  const auto tried = plan(wanted, memory);
  REQUIRE(tried);
  CHECK(tried->settings.span_precision == PrecisionTier::Quality);  // SPAN FP32 gardé
  CHECK(tried->settings.motion_precision == PrecisionTier::Balanced);
  CHECK(tried->settings.rife_factor == 2);
  REQUIRE(tried->notes.size() == 1);
  CHECK(tried->notes[0].change == StageChange::Balanced);
  // Le FP16 refusé à son tour : l'interpolation part, SPAN reste au cran Qualité.
  refuse(memory, ChainStage::Motion, tried->settings);
  const auto again = plan(wanted, memory);
  REQUIRE(again);
  CHECK(again->settings.span_precision == PrecisionTier::Quality);
  CHECK(again->settings.rife_factor == 0);
}

TEST_CASE("Étages : « Annuler et regarder » ne coupe que l'étage annulé, et pour sa vidéo (A31-05, A08-01)",
          "[app][stages]") {
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Quality);
  StageMemory memory;
  refuse(memory, ChainStage::Motion, wanted, true);
  const auto tried = plan(wanted, memory);
  REQUIRE(tried);
  CHECK(tried->settings.upscale);  // SPAN déjà prêt reste
  CHECK(tried->settings.rife_factor == 0);
  CHECK(tried->settings.motion_precision == PrecisionTier::Quality);  // aucun repli de cran essayé
  REQUIRE(tried->notes.size() == 1);
  CHECK(tried->notes[0].change == StageChange::Cancelled);
  // La vidéo suivante essaie de nouveau.
  const auto next = plan(wanted, memory, true, "ep02.mkv");
  REQUIRE(next);
  CHECK(next->settings.rife_factor == 2);
  CHECK(
      watch_now(memory
                    .find(ChainStage::Motion,
                          stage_signature(ChainStage::Motion, wanted, kWidth, kHeight, kLibrary), "ep01.mkv")
                    ->refusal));
}

TEST_CASE("Étages : un refus vaut pour la séance, « Réessayer » l'oublie", "[app][stages]") {
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  StageMemory memory;
  refuse(memory, ChainStage::Upscale, wanted);
  CHECK_FALSE(plan(wanted, memory, true, "ep02.mkv")->settings.upscale);  // pas 20 minutes de plus
  // Un autre modèle est une autre construction : essayé.
  PlaySettings other = wanted;
  other.model = "2x_AnimeJaNai_HD_V3_Compact.onnx";
  CHECK(plan(other, memory)->settings.upscale);
  memory.forget();
  CHECK(plan(wanted, memory)->settings.upscale);
}

}  // namespace
}  // namespace ajp::app::play
