#include "app/play/chain_setup.h"

#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"
#include "app/play/visible_error.h"
#include "stage_rig.h"

namespace ajp::app::play {
namespace {

PlaySettings fast_settings() {
  PlaySettings s;
  s.rife_factor = 2;
  s.rife_auto = true;
  s.auto_free = true;
  return s;
}

}  // namespace

TEST_CASE("Rapide : le warp et l'anneau suivent l'écran, pas le facteur demandé", "[app][chain]") {
  PlaySettings s = fast_settings();
  CHECK(top_factor(s, 165.0, 23.976) == 6);
  CHECK(ring_depth(s, factor_of(s), 165.0, 23.976) == 8U);
  // B08-08 : sur 60 Hz, ×2 au plus : l'anneau de 4 suffit, pas 8 ; écran inconnu : 8.
  CHECK(ring_depth(s, factor_of(s), 60.0, 23.976) == 4U);
  CHECK(ring_depth(s, factor_of(s), 90.0, 23.976) == 6U);
  CHECK(ring_depth(s, factor_of(s), 0.0, 23.976) == 8U);
  CHECK(ring_depth(s, Factor{}, 165.0, 23.976) == 4U);  // sans interpolation chargée

  s.auto_free = false;  // Stable : le facteur demandé borne toujours
  CHECK(top_factor(s, 165.0, 23.976) == 2);
  CHECK(ring_depth(s, factor_of(s), 165.0, 23.976) == 4U);
}

TEST_CASE("Rapide : jamais moins de 1, aucune interpolation sans facteur", "[app][chain]") {
  PlaySettings s = fast_settings();
  CHECK(top_factor(s, 24.0, 23.976) == 1);
  s.rife_factor = 0;
  CHECK(top_factor(s, 165.0, 23.976) == 1);
}

namespace {

// Une chaîne ouverte pour s : 1080p de sortie, 23,976 i/s, écran à 165 Hz, SPAN et RIFE chargés.
OpenedChain opened_for(const PlaySettings& s, int output_height = 1080) {
  OpenedChain o{.display_hz = 165.0,
                .source_fps = 23.976,
                .output_height = output_height,
                .span = true,
                .rife_loaded = true};
  o.shape = chain_shape(s, o);
  return o;
}

PlaySettings manual(int factor) {
  PlaySettings s;
  s.upscale = true;
  s.rife_factor = factor;
  s.rife_auto = false;
  s.auto_free = false;
  return s;
}

}  // namespace

TEST_CASE("A08-03 : ×2 ouvert, ×4 demande un autre anneau et un autre ordre du warp", "[app][chain]") {
  const PlaySettings x2 = manual(2);
  const OpenedChain opened = opened_for(x2);
  CHECK(opened.shape == ChainShape{.ring = 4, .warp_after_span = false});
  CHECK(fits(opened, x2));

  const PlaySettings x4 = manual(4);
  CHECK_FALSE(fits(opened, x4));
  const Change change = change_between(x2, x4, true, &opened);
  CHECK(change.cost == ChangeCost::Reopen);
  CHECK(change.reason.find("frame ring") != std::string::npos);
  CHECK(change.reason.find("after SPAN") != std::string::npos);
  // Sans chaîne à comparer (accueil), le facteur reste un réglage direct.
  CHECK(change_between(x2, x4, true).cost == ChangeCost::Live);
  // Et la part directe garde le facteur en cours jusqu'à « Appliquer ».
  CHECK(live_part(x2, x4, true, &opened).rife_factor == 2);
  CHECK(live_part(x2, x4, true).rife_factor == 4);
}

TEST_CASE("A08-03 : ×2 manuel ouvert, Auto Rapide à 165 Hz rouvre", "[app][chain]") {
  const PlaySettings x2 = manual(2);
  const OpenedChain opened = opened_for(x2);
  PlaySettings fast = x2;
  fast.rife_auto = true;
  fast.auto_free = true;
  // Rapide à 165 Hz va jusqu'à ×6 : le warp passe après SPAN, comme ×3 à la main (C32-01), et l'anneau de 8.
  CHECK(chain_shape(fast, opened) == ChainShape{.ring = 8, .warp_after_span = true});
  CHECK(change_between(x2, fast, true, &opened).cost == ChangeCost::Reopen);
  const PlaySettings live = live_part(x2, fast, true, &opened);
  CHECK_FALSE(live.rife_auto);
  CHECK_FALSE(live.auto_free);
  // Le brouillon du panneau garde le choix qui attend quand un autre réglage bouge ailleurs.
  PlaySettings louder = x2;
  louder.volume = 0.5F;
  const PlaySettings draft = follow_running(fast, x2, louder, true);
  CHECK(draft.rife_auto);
  CHECK(draft.auto_free);
  CHECK(draft.volume == 0.5F);
}

TEST_CASE("A08-03 : ce qui tient dans la chaîne ouverte reste direct", "[app][chain]") {
  // Ouverte à ×4 : anneau de 6, warp après SPAN. ×3 y tient ; ×2 demande le warp avant SPAN (qualité).
  const PlaySettings x4 = manual(4);
  const OpenedChain opened = opened_for(x4);
  CHECK(opened.shape == ChainShape{.ring = 6, .warp_after_span = true});
  CHECK(change_between(x4, manual(3), true, &opened).cost == ChangeCost::Live);
  const Change down = change_between(x4, manual(2), true, &opened);
  CHECK(down.cost == ChangeCost::Reopen);
  CHECK(down.reason.find("before SPAN") != std::string::npos);
  // Couper l'interpolation ne rouvre jamais.
  CHECK(change_between(x4, manual(0), true, &opened).cost == ChangeCost::Live);
  // Sortie au-dessus de 1080p : le warp est après SPAN à tout facteur, ×2 → ×3 ne demande que l'anneau.
  const OpenedChain uhd = opened_for(manual(4), 2160);
  CHECK(change_between(manual(4), manual(2), true, &uhd).cost == ChangeCost::Live);
  // Sans SPAN, l'ordre du warp ne compte pas.
  OpenedChain bare = opened_for(manual(4));
  bare.span = false;
  bare.shape = chain_shape(manual(4), bare);
  CHECK(change_between(manual(4), manual(2), true, &bare).cost == ChangeCost::Live);
}

TEST_CASE("B22-03 : en Auto, l'ordre du warp ne suit ni la charge ni le plafond", "[app][chain]") {
  // Auto Stable jusqu'à ×4 : le plafond (60 ou l'écran) change le palier le plus haut, pas l'ordre du warp ;
  // l'anneau est celui du facteur demandé.
  PlaySettings capped = manual(4);
  capped.rife_auto = true;
  capped.auto_cap = 60;
  const OpenedChain opened = opened_for(capped);
  CHECK(opened.shape == ChainShape{.ring = 6, .warp_after_span = true});
  PlaySettings free = capped;
  free.auto_cap = 0;
  CHECK(chain_shape(free, opened) == opened.shape);
  CHECK(change_between(capped, free, true, &opened).cost == ChangeCost::Live);
  // Un facteur choisi à la main garde la règle de ×3 (coût de SPAN par image créée).
  CHECK(order_factor(manual(4), 165.0, 23.976) == 4);
  CHECK(chain_shape(manual(4), opened_for(manual(4), 1080)).warp_after_span);
}

TEST_CASE("C32-01 : en Auto, l'ordre du warp suit le plus haut facteur de l'écran, comme à la main",
          "[app][chain]") {
  // Rapide à 165 Hz (×6 possible), sortie de 1080 lignes : après SPAN, comme ×3 choisi à la main ; avant,
  // sept passes de SPAN par source au lieu d'une.
  PlaySettings fast = fast_settings();
  CHECK(order_factor(fast, 165.0, 23.976) == 6);
  CHECK(chain_shape(fast, opened_for(fast, 1080)).warp_after_span);
  CHECK(chain_shape(fast, opened_for(fast, 2160)).warp_after_span);
  PlaySettings by_hand = manual(3);
  CHECK(chain_shape(fast, opened_for(fast, 1080)).warp_after_span ==
        chain_shape(by_hand, opened_for(by_hand, 1080)).warp_after_span);
  // Écran à 60 Hz : ×2 au plus, avant SPAN sous 1080 lignes (deux passes de SPAN au plus par source).
  OpenedChain sixty = opened_for(fast, 1080);
  sixty.display_hz = 60.0;
  CHECK(order_factor(fast, 60.0, 23.976) == 2);
  CHECK_FALSE(chain_shape(fast, sixty).warp_after_span);
  // Écran inconnu : seule la hauteur décide.
  CHECK(order_factor(fast, 0.0, 23.976) == 1);
  // Stable : le facteur demandé borne, comme l'anneau.
  PlaySettings stable = fast;
  stable.auto_free = false;
  CHECK(order_factor(stable, 165.0, 23.976) == 2);
  stable.rife_factor = 4;
  CHECK(order_factor(stable, 165.0, 23.976) == 4);
}

TEST_CASE("B22-01 : interpolation coupée puis rallumée, le moteur chargé est comparé au choix",
          "[app][settings]") {
  PlaySettings run = manual(3);
  OpenedChain opened = opened_for(run);
  opened.loaded = loaded_by(run);
  // Off, puis DistilDRBA, le cran Qualité et l'ordre Après : rien à recharger, le réglage passe et
  // s'enregistre.
  PlaySettings off = run;
  off.rife_factor = 0;
  off.motion_engine = MotionEngine::DistilDrba;
  off.motion_precision = PrecisionTier::Quality;
  off.span_order = SpanOrder::After;
  const PlaySettings live = live_part(run, off, /*rife_loaded=*/true, &opened);
  CHECK(live.motion_engine == MotionEngine::DistilDrba);
  CHECK(change_between(run, off, true, &opened).cost == ChangeCost::Live);
  // L'interpolation revient : RIFE FP16 est toujours celui qui tourne, la réouverture est annoncée.
  PlaySettings on = off;
  on.rife_factor = 3;
  const Change back = change_between(live, on, true, &opened);
  CHECK(back.cost == ChangeCost::Reopen);
  CHECK(back.reason.find("interpolation engine") != std::string::npos);
  // Le cran de précision seul (même moteur) et l'ordre du warp aussi.
  PlaySettings quality = run;
  quality.rife_factor = 0;
  quality.motion_precision = PrecisionTier::Quality;
  PlaySettings quality_on = quality;
  quality_on.rife_factor = 3;
  CHECK(change_between(quality, quality_on, true, &opened).reason.find("switches to") != std::string::npos);
  // Le même moteur qu'à l'ouverture : en direct.
  PlaySettings same = run;
  same.rife_factor = 0;
  PlaySettings same_on = run;
  CHECK(change_between(same, same_on, true, &opened).cost == ChangeCost::Live);
  // Sans chaîne à comparer, on compare au réglage en cours (comportement d'avant).
  CHECK(change_between(live, on, true).cost == ChangeCost::Live);
}

TEST_CASE("A03-21 : un fusionné refusé se dit à l'écran avec son motif, rien sinon", "[app][chain]") {
  OpenedChain opened = opened_for(manual(2));
  CHECK(fused_refusal_notice(opened).empty());
  opened.rife_fused_refused = RefusalCode::BuilderFailed;
  const std::string rife = fused_refusal_notice(opened).str();
  CHECK(rife.contains("RIFE"));
  CHECK(rife.contains(visible_error(Refusal{RefusalCode::BuilderFailed, ""})));
  // DistilDRBA chargé : c'est son repli qui est dit.
  opened.drba = true;
  opened.drba_fused_refused = RefusalCode::CacheIo;
  const std::string drba = fused_refusal_notice(opened).str();
  CHECK(drba.contains("DistilDRBA"));
  CHECK(drba.contains(visible_error(Refusal{RefusalCode::CacheIo, ""})));
  CHECK_FALSE(visible_error(Refusal{RefusalCode::CacheIo, ""}).empty());
}

TEST_CASE("C45-01 : une précision réduite gardée se dit à l'écran, chiffrée ou non mesurée", "[app][chain]") {
  CHECK(reduced_precision_notice("SPAN", gpu::PrecisionNotice{}).empty());
  const gpu::PrecisionNotice measured{
      .layers = 4, .loss_measured = true, .loss_db = 80.9F, .gain_measured = true, .gain_percent = 50.0F};
  const std::string full = reduced_precision_notice("SPAN", measured).str();
  CHECK(full.contains("SPAN"));
  CHECK(full.contains("4"));
  CHECK(full.contains("80"));
  CHECK(full.contains("+50"));
  const std::string blind = reduced_precision_notice("SPAN", gpu::PrecisionNotice{.layers = 2}).str();
  CHECK_FALSE(blind.contains("dB"));
  CHECK_FALSE(blind.empty());
  CHECK(blind != full);
}

TEST_CASE("B08-01 : Ctrl+I rallume l'interpolation au facteur que la séance avait, pas à ×3",
          "[app][chain]") {
  using Pair = std::pair<int, int>;
  // Allumée : Ctrl+I la coupe, le dénominateur reste.
  CHECK(toggled_interpolation(5, 2, std::nullopt) == Pair{0, 2});
  // Coupée par Ctrl+I ou Ctrl+0 : le facteur gardé pour la séance revient, ×5/2 compris.
  CHECK(toggled_interpolation(0, 2, Pair{5, 2}) == Pair{5, 2});
  CHECK(toggled_interpolation(0, 1, Pair{2, 1}) == Pair{2, 1});
  // Rien de gardé (play.toml sans interpolation), ou un facteur gardé qui n'interpolait pas : le défaut.
  CHECK(toggled_interpolation(0, 1, std::nullopt) == Pair{kDefaultRifeFactor, 1});
  CHECK(toggled_interpolation(0, 1, Pair{0, 1}) == Pair{kDefaultRifeFactor, 1});
}

TEST_CASE("B39-02 : les faits de la lecture vont en tête du rapport, vidéo, écran, sortie, chaîne",
          "[app][chain]") {
  const auto facts = playback_facts({.file = "ep01.mkv",
                                     .codec = "hevc",
                                     .width = 1920,
                                     .height = 1080,
                                     .frame_rate = 23.976,
                                     .transfer = "PQ",
                                     .screen = "pic 1000 nits",
                                     .hdr_output = true,
                                     .chain = "SPAN ×2 · RIFE ×2"});
  REQUIRE(facts.size() == 4);
  CHECK(facts[0].first == "Vidéo");
  CHECK(facts[0].second.contains("hevc"));
  CHECK(facts[0].second.contains("1920x1080"));
  CHECK(facts[0].second.contains("23.976"));
  CHECK(facts[0].second.contains("source PQ"));
  CHECK(facts[1].second == "pic 1000 nits");
  CHECK(facts[2].second == "HDR10");
  CHECK(facts[3].second == "SPAN ×2 · RIFE ×2");
  // Cadence inconnue : dit, pas « 0 i/s ».
  CHECK(playback_facts({.file = "a.mkv"})[0].second.contains("cadence inconnue"));
}

TEST_CASE("C08-08 : la clé de la mémoire de l'Auto distingue les modèles RIFE et SPAN chargés",
          "[app][chain]") {
  ChainEngines light;
  light.rife_model = std::filesystem::path("models/rife_v4.25_lite.onnx");
  light.model = ModelChoice{.path = "models/2x_AnimeJaNai_V3_SuperUltraCompact.onnx"};
  ChainEngines heavy;
  heavy.rife_model = std::filesystem::path("models/rife_v4.26.onnx");
  heavy.model = light.model;
  CHECK(light.auto_memory_engine("FP16") != heavy.auto_memory_engine("FP16"));
  CHECK(light.auto_memory_engine("FP16") != light.auto_memory_engine("INT8"));
  ChainEngines other_span;
  other_span.rife_model = light.rife_model;
  other_span.model = ModelChoice{.path = "models/2x_AnimeJaNai_V3_UltraCompact.onnx"};
  CHECK(light.auto_memory_engine("FP16") != other_span.auto_memory_engine("FP16"));
  CHECK(light.auto_memory_engine("FP16").contains("rife_v4.25_lite"));
}

// C40-07 : le câblage des options à l'essai vers l'étage. « Éteint ne change rien », « allumé fait ce qu'il
// dit ».
TEST_CASE("ChainEngines::upscale câble le warp de Catmull-Rom et les options à l'essai", "[app][chain]") {
  const auto motion = [] {
    return std::make_unique<test::FakeEngine>("rife", std::vector<std::string>{"a"},
                                              std::vector<std::string>{"b"});
  };
  PlaySettings s;
  ChainEngines with_motion;
  with_motion.rife_motion = motion();
  with_motion.smooth_drawings = true;
  with_motion.rife_encode_all = true;
  ChainEngines without_motion;

  s.cubic_warp = false;
  CHECK(with_motion.upscale(s, nullptr, nullptr, 1920, 1080, {}).warp == kernels::RifeWarp::Bilinear);
  s.cubic_warp = true;
  CHECK(with_motion.upscale(s, nullptr, nullptr, 1920, 1080, {}).warp == kernels::RifeWarp::CatmullRom);
  // Sans RIFE mouvement, l'option n'a rien à régler : le warp reste bilinéaire.
  CHECK(without_motion.upscale(s, nullptr, nullptr, 1920, 1080, {}).warp == kernels::RifeWarp::Bilinear);

  const UpscaleSetup setup = with_motion.upscale(s, nullptr, nullptr, 1920, 1080, {});
  CHECK(setup.smooth_drawings);
  CHECK(setup.rife_encode_all);
  CHECK_FALSE(setup.late_fate);  // sans horloge de lecture
  const UpscaleSetup plain = without_motion.upscale(s, nullptr, nullptr, 1920, 1080, {});
  CHECK_FALSE(plain.smooth_drawings);
  CHECK_FALSE(plain.rife_encode_all);
  CHECK_FALSE(plain.skip_held);
}

}  // namespace ajp::app::play
