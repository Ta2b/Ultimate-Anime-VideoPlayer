// Les étages refusés qui ne coupent plus la lecture (décisions 6 et 10 ; A29-01, A31-04, A08-01, A08-16,
// A31-05, A31-15) : ce que la préparation essaie selon les refus retenus.
#include "app/play/stage_fallback.h"

#include <catch2/catch_test_macros.hpp>

#include "app/play/links.h"
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

TEST_CASE("Étages : « Annuler et regarder » sur SPAN vaut pour toute la préparation de la vidéo (B29-02)",
          "[app][stages]") {
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  StageMemory memory;
  refuse(memory, ChainStage::Upscale, wanted, true);
  const auto tried = plan(wanted, memory);
  REQUIRE(tried);
  CHECK_FALSE(tried->settings.upscale);
  // L'interpolation, à une autre taille de travail, demanderait une seconde construction : elle part aussi.
  CHECK(tried->settings.rife_factor == 0);
  REQUIRE(tried->notes.size() == 2);
  CHECK(tried->notes[1].stage == ChainStage::Motion);
  CHECK(tried->notes[1].change == StageChange::Cancelled);
  // La vidéo suivante essaie de nouveau les deux.
  const auto next = plan(wanted, memory, true, "ep02.mkv");
  REQUIRE(next);
  CHECK(next->settings.upscale);
  CHECK(next->settings.rife_factor == 2);
  // Un refus (pas une annulation) de SPAN laisse l'interpolation, comme avant.
  StageMemory refused;
  refuse(refused, ChainStage::Upscale, wanted);
  CHECK(plan(wanted, refused)->settings.rife_factor == 2);
}

TEST_CASE("Étages : le refus de l'interpolation suit l'ordre du warp, pas le facteur brut (B09-05)",
          "[app][stages]") {
  PlaySettings five = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  five.rife_factor = 5;
  five.rife_auto = false;  // un facteur choisi à la main : il fixe l'ordre du warp (order_factor)
  StageMemory memory;
  refuse(memory, ChainStage::Motion, five);
  // L'Auto récrit le facteur sur un autre écran (×5 → ×4) : même ordre du warp, même moteur, pas de
  // nouvelle construction.
  PlaySettings four = five;
  four.rife_factor = 4;
  CHECK(plan(four, memory)->settings.rife_factor == 0);
  // ×5/2 (×2,5) n'est pas ×5 : le warp y passe avant SPAN en sortie 1080p.
  PlaySettings half = five;
  half.rife_factor_den = 2;
  half.work_height = 540;
  CHECK(stage_signature(ChainStage::Motion, half, kWidth, kHeight, kLibrary) != stage_signature(
                                                                                    ChainStage::Motion,
                                                                                    [&] {
                                                                                      PlaySettings s = half;
                                                                                      s.rife_factor_den = 1;
                                                                                      return s;
                                                                                    }(),
                                                                                    kWidth, kHeight,
                                                                                    kLibrary));
}

TEST_CASE("Étages : l'interpolation refusée avec l'échelle par paire revient sans elle", "[app][stages]") {
  PlaySettings half = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  half.pair_scale = true;
  StageMemory memory;
  refuse(memory, ChainStage::Motion, half);
  CHECK(plan(half, memory)->settings.rife_factor == 0);
  PlaySettings full = half;
  full.pair_scale = false;
  CHECK(plan(full, memory)->settings.rife_factor == half.rife_factor);
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

TEST_CASE("Étages : une annulation ne remplace ni un refus réel ni l'annulation d'une autre vidéo (B29-03)",
          "[app][stages]") {
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  SECTION("deux vidéos annulées gardent chacune leur annulation") {
    StageMemory memory;
    refuse(memory, ChainStage::Motion, wanted, true, "ep01.mkv");
    refuse(memory, ChainStage::Motion, wanted, true, "ep02.mkv");
    CHECK(plan(wanted, memory, true, "ep01.mkv")->settings.rife_factor == 0);
    CHECK(plan(wanted, memory, true, "ep02.mkv")->settings.rife_factor == 0);
    CHECK(plan(wanted, memory, true, "ep03.mkv")->settings.rife_factor == 2);
  }
  SECTION("une annulation après un refus réel laisse le refus valoir pour toutes les vidéos") {
    StageMemory memory;
    refuse(memory, ChainStage::Motion, wanted);
    refuse(memory, ChainStage::Motion, wanted, true, "ep01.mkv");
    CHECK(plan(wanted, memory, true, "ep02.mkv")->settings.rife_factor == 0);
  }
  SECTION("un refus réel recouvre les annulations de sa signature") {
    StageMemory memory;
    refuse(memory, ChainStage::Motion, wanted, true, "ep01.mkv");
    refuse(memory, ChainStage::Motion, wanted);
    const auto next = plan(wanted, memory, true, "ep02.mkv");
    REQUIRE(next);
    CHECK(next->settings.rife_factor == 0);
    REQUIRE(next->notes.size() == 1);
    CHECK(next->notes[0].change == StageChange::Dropped);
  }
}

}  // namespace
TEST_CASE("B08-02 : un repli interne refusé est retenu pour la séance, au panneau, et oublié par Réessayer",
          "[stage_fallback]") {
  StageMemory memory;
  const std::string key = inner_key("rife-fused", kLibrary, "rife.onnx", 1920, 1088);
  const Refusal why{RefusalCode::BuilderFailed, "trtexec : mémoire"};
  {
    // La première ouverture : le fusionné se construit, échoue, le refus est retenu et neuf.
    std::vector<StageNote> notes;
    bool fresh = false;
    InnerFallbacks inner(&memory, notes, fresh);
    CHECK(inner.refused(ChainStage::Motion, key) == nullptr);
    inner.failed(ChainStage::Motion, key, why);
    CHECK(fresh);
    REQUIRE(notes.size() == 1);
    CHECK(notes[0].change == StageChange::Fallback);
    CHECK_FALSE(memory.empty());
  }
  {
    // La réouverture suivante : la construction est sautée, le panneau le dit encore, l'avis à l'écran non.
    std::vector<StageNote> notes;
    bool fresh = false;
    InnerFallbacks inner(&memory, notes, fresh);
    const Refusal* before = inner.refused(ChainStage::Motion, key);
    REQUIRE(before != nullptr);
    CHECK(before->code == RefusalCode::BuilderFailed);
    CHECK_FALSE(fresh);
    REQUIRE(notes.size() == 1);
    CHECK_FALSE(stage_panel_note(notes).empty());
    CHECK(stage_notice(notes).empty());
    // Une autre taille est un autre moteur : elle s'essaie.
    CHECK(inner.refused(ChainStage::Motion, inner_key("rife-fused", kLibrary, "rife.onnx", 1280, 736)) ==
          nullptr);
  }
  memory.forget();  // « Réessayer »
  CHECK(memory.empty());
  std::vector<StageNote> notes;
  bool fresh = false;
  InnerFallbacks inner(&memory, notes, fresh);
  CHECK(inner.refused(ChainStage::Motion, key) == nullptr);
}

TEST_CASE("B08-02 : « Annuler et regarder » n'est pas retenu comme un repli", "[stage_fallback]") {
  StageMemory memory;
  std::vector<StageNote> notes;
  bool fresh = false;
  InnerFallbacks inner(&memory, notes, fresh);
  inner.failed(ChainStage::Upscale, "k",
               Refusal{.code = RefusalCode::Cancelled, .detail = "annulé", .key = std::string(kWatchNowKey)});
  CHECK(memory.empty());
  CHECK(notes.empty());
  CHECK_FALSE(fresh);
}

TEST_CASE("B31-02 : l'anneau refusé retire l'interpolation, puis SPAN, pour la réouverture",
          "[stage_fallback]") {
  StageMemory memory;
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  const Refusal ring{.code = RefusalCode::GpuMemoryExhausted,
                     .detail = "anneau 3840x2160 x 6",
                     .key = std::string(kRingRefusalKey)};
  // Un refus après la préparation : l'interpolation part la première.
  REQUIRE(retreat_stage(&memory, ring, wanted, true, true, kWidth, kHeight, kLibrary, "ep01.mkv") ==
          ChainStage::Motion);
  auto plan = plan_stages(wanted, memory, kWidth, kHeight, kLibrary, "ep01.mkv", true);
  REQUIRE(plan);
  CHECK(plan->settings.rife_factor == 0);
  CHECK(plan->settings.upscale);
  REQUIRE(plan->notes.size() == 1);
  CHECK(plan->notes[0].change == StageChange::Dropped);
  // La chaîne rouverte sans interpolation refuse encore : SPAN part à son tour.
  REQUIRE(retreat_stage(&memory, ring, plan->settings, false, true, kWidth, kHeight, kLibrary, "ep01.mkv") ==
          ChainStage::Upscale);
  plan = plan_stages(wanted, memory, kWidth, kHeight, kLibrary, "ep01.mkv", true);
  REQUIRE(plan);
  CHECK_FALSE(plan->settings.upscale);
  // Plus rien à retirer : le refus arrête la vidéo comme avant.
  CHECK_FALSE(
      retreat_stage(&memory, ring, plan->settings, false, false, kWidth, kHeight, kLibrary, "ep01.mkv"));
}

TEST_CASE("B31-02 : un appareil perdu ou sans mémoire de séance ne retire rien", "[stage_fallback]") {
  StageMemory memory;
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  CHECK_FALSE(retreat_stage(&memory, Refusal{RefusalCode::DeviceLost, "perdu"}, wanted, true, true, kWidth,
                            kHeight, kLibrary, "ep01.mkv"));
  CHECK_FALSE(retreat_stage(
      nullptr,
      Refusal{.code = RefusalCode::GpuMemoryExhausted, .detail = "x", .key = std::string(kRingRefusalKey)},
      wanted, true, true, kWidth, kHeight, kLibrary, "ep01.mkv"));
  CHECK(memory.empty());
}

// C09-01 : un refus de l'étage GPU qui ne vient d'aucun moteur (ingestion d'un format que le backend ne lit
// pas) ne retire rien : la vidéo s'arrête comme avant, et les suivantes jouent la chaîne demandée.
TEST_CASE("C09-01 : un refus d'ingestion d'une vidéo ne retire rien aux vidéos suivantes",
          "[stage_fallback]") {
  StageMemory memory;
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  const Refusal ingest{.code = RefusalCode::FormatUnsupported,
                       .detail = "ingestion : format ou matrice non pris en charge",
                       .key = fatal_stage_key("")};
  CHECK_FALSE(retreat_stage(&memory, ingest, wanted, true, true, kWidth, kHeight, kLibrary, "bizarre.mkv"));
  CHECK(memory.empty());
  const auto next = plan_stages(wanted, memory, kWidth, kHeight, kLibrary, "ep02.mkv", true);
  REQUIRE(next);
  CHECK(next->settings.rife_factor == wanted.rife_factor);
  CHECK(next->settings.upscale);
  CHECK(next->notes.empty());
}

TEST_CASE("C09-01 : un refus d'un réseau en lecture ne retire que son étage", "[stage_fallback]") {
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  // SPAN refusé : SPAN part, l'interpolation reste.
  StageMemory memory;
  const Refusal span{
      .code = RefusalCode::VendorCallFailed, .detail = "enqueue", .key = fatal_stage_key(kSpanRefusalKey)};
  REQUIRE(retreat_stage(&memory, span, wanted, true, true, kWidth, kHeight, kLibrary, "ep01.mkv") ==
          ChainStage::Upscale);
  auto plan = plan_stages(wanted, memory, kWidth, kHeight, kLibrary, "ep01.mkv", true);
  REQUIRE(plan);
  CHECK_FALSE(plan->settings.upscale);
  CHECK(plan->settings.rife_factor == 2);
  // L'interpolation refusée : elle part seule.
  StageMemory other;
  const Refusal rife{
      .code = RefusalCode::VendorCallFailed, .detail = "enqueue", .key = fatal_stage_key(kMotionRefusalKey)};
  REQUIRE(retreat_stage(&other, rife, wanted, true, true, kWidth, kHeight, kLibrary, "ep01.mkv") ==
          ChainStage::Motion);
  plan = plan_stages(wanted, other, kWidth, kHeight, kLibrary, "ep01.mkv", true);
  REQUIRE(plan);
  CHECK(plan->settings.upscale);
  CHECK(plan->settings.rife_factor == 0);
  // L'étage nommé n'est pas chargé : rien à retirer.
  StageMemory none;
  CHECK_FALSE(retreat_stage(&none, span, wanted, true, false, kWidth, kHeight, kLibrary, "ep01.mkv"));
}

// C09-03 : la signature d'un refus d'interpolation prend l'ordre du warp que la chaîne prend (order_factor).
TEST_CASE("C09-03 : la signature de l'interpolation suit l'ordre réel du warp en Auto", "[stage_fallback]") {
  PlaySettings s = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  s.rife_auto = true;
  s.rife_factor = 3;
  s.span_order = SpanOrder::Auto;
  // Source 960 × 540, sortie de 1080 lignes, Auto Stable jusqu'à ×3 : sur 60 Hz, ×2 au plus, le warp avant
  // SPAN.
  const std::string at60 =
      stage_signature(ChainStage::Motion, s, 960, 540, kLibrary, {.display_hz = 60.0, .source_fps = 23.976});
  CHECK(at60.contains("|warp-before|"));
  // Sur 144 Hz, ×3 : après SPAN, comme la chaîne.
  const std::string at144 =
      stage_signature(ChainStage::Motion, s, 960, 540, kLibrary, {.display_hz = 144.0, .source_fps = 23.976});
  CHECK(at144.contains("|warp-after|"));
  // Sans SPAN, l'ordre ne vaut rien : avant. L'ordre n'est plus le dernier champ de la signature (l'échelle
  // par paire le suit) : il se cherche entre ses séparateurs.
  s.upscale = false;
  CHECK(
      stage_signature(ChainStage::Motion, s, 960, 540, kLibrary, {.display_hz = 144.0, .source_fps = 23.976})
          .contains("|warp-before|"));
}

TEST_CASE("C31-09 : une mémoire vidéo pleine ne retire l'étage que pour sa vidéo", "[stage_fallback]") {
  StageMemory memory;
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  memory.remember({.stage = ChainStage::Upscale,
                   .signature = stage_signature(ChainStage::Upscale, wanted, kWidth, kHeight, kLibrary),
                   .refusal = Refusal{RefusalCode::GpuMemoryExhausted, "un jeu tient la mémoire"},
                   .file = "ep01.mkv"});
  // La même vidéo rouverte reste sans SPAN : pas de boucle sur le même refus.
  auto same = plan(wanted, memory, true, "ep01.mkv");
  REQUIRE(same);
  CHECK_FALSE(same->settings.upscale);
  // L'épisode suivant réessaie SPAN (avant : retiré pour toute la séance).
  auto next = plan(wanted, memory, true, "ep02.mkv");
  REQUIRE(next);
  CHECK(next->settings.upscale);
  CHECK(next->notes.empty());
  // Un refus réel de la même signature vaut toujours pour toutes les vidéos.
  refuse(memory, ChainStage::Upscale, wanted);
  next = plan(wanted, memory, true, "ep02.mkv");
  REQUIRE(next);
  CHECK_FALSE(next->settings.upscale);
}

TEST_CASE("C31-05 : un refus de l'affichage sans clé retire un étage, pas une annulation ni un fichier",
          "[stage_fallback]") {
  CHECK(display_retreats(Refusal{RefusalCode::GpuMemoryExhausted, "pl_tex_recreate"}));
  CHECK(display_retreats(Refusal{RefusalCode::VendorCallFailed, "pl_render_image"}));
  CHECK(display_retreats(Refusal{RefusalCode::VulkanCallFailed, "vkAllocateMemory"}));
  CHECK_FALSE(display_retreats(Refusal{RefusalCode::DeviceLost, "perdu"}));
  CHECK_FALSE(display_retreats(Refusal{RefusalCode::DecodeFailed, "paquet abîmé"}));
  CHECK_FALSE(
      display_retreats(Refusal{.code = RefusalCode::VendorCallFailed, .detail = "x", .key = "autre"}));
  // Le repli lui-même : l'interpolation d'abord, comme pour un refus d'étage.
  StageMemory memory;
  const PlaySettings wanted = chain(PrecisionTier::Balanced, PrecisionTier::Balanced);
  CHECK(retreat_stage(&memory, Refusal{RefusalCode::VendorCallFailed, "pl_render_image"}, wanted, true, true,
                      kWidth, kHeight, kLibrary, "ep01.mkv") == ChainStage::Motion);
}

}  // namespace ajp::app::play
