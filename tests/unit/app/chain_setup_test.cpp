#include "app/play/chain_setup.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/visible_error.h"

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

TEST_CASE("Repli du Rapide : la séance passe en Stable, le choix des réglages reste", "[app][settings]") {
  PlaySettings s;
  s.auto_free = true;
  CHECK(free_requested(s, false));
  CHECK_FALSE(free_requested(s, true));
  CHECK(s.auto_free);  // play.toml garde Rapide
  s.auto_free = false;
  CHECK_FALSE(free_requested(s, false));
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

TEST_CASE("A22-04 : le plafond de l'Auto qui change le plus haut facteur rouvre", "[app][chain]") {
  // Auto Stable jusqu'à ×4, plafonné à 60 i/s : ×2 au plus sur 23,976, warp avant SPAN, anneau de 4.
  PlaySettings capped = manual(4);
  capped.rife_auto = true;
  capped.auto_cap = 60;
  const OpenedChain opened = opened_for(capped);
  CHECK(opened.shape == ChainShape{.ring = 6, .warp_after_span = false});
  PlaySettings free = capped;
  free.auto_cap = 0;  // l'écran : ×4, warp après SPAN
  CHECK(change_between(capped, free, true, &opened).cost == ChangeCost::Reopen);
  CHECK(live_part(capped, free, true, &opened).auto_cap == 60);
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

}  // namespace ajp::app::play
