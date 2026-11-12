#include "app/play/auto_config.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/options.h"

using namespace ajp;
using namespace ajp::app::play;

namespace {

const core::SourceInfo k1080p24{.width = 1920, .height = 1080, .fps = 23.976};

PlaySettings with(Preset preset, Priority priority = Priority::Balance) {
  PlaySettings s;
  s.preset = preset;
  s.priority = priority;
  return s;
}

PlaySettings applied(Preset preset, Tier tier, int display_height, double hz,
                     const core::SourceInfo& source = k1080p24, Priority priority = Priority::Balance) {
  return apply_preset(with(preset, priority), Machine{tier, display_height, hz}, source);
}

}  // namespace

TEST_CASE("Personnalisé ne touche à rien", "[auto_config]") {
  PlaySettings s = with(Preset::Custom);
  s.work_height = 900;
  s.rife_factor = 5;
  s.motion_engine = MotionEngine::Apple;
  CHECK(apply_preset(s, Machine{Tier::Ultra, 2160, 165.0}, k1080p24) == s);
  // Auto par défaut : un premier double-clic sur une vidéo, sans assistant, ne prend pas la chaîne lourde.
  CHECK(PlaySettings{}.preset == Preset::Auto);
}

TEST_CASE("Auto suit la tranche de la machine", "[auto_config]") {
  const PlaySettings ultra = applied(Preset::Auto, Tier::Ultra, 2160, 165.0);
  CHECK(ultra.motion_engine == MotionEngine::Rife);  // DistilDRBA attend sa porte ; GMFSS Union est manuel
  CHECK(ultra.upscale);
  CHECK(ultra.rife_auto);
  const PlaySettings fast = applied(Preset::Auto, Tier::Quality, 1440, 144.0);
  CHECK(fast.motion_engine == MotionEngine::Rife);
  CHECK(fast.rife_model == "rife_v4.26.onnx");  // Qualité et Ultra : RIFE 4.26
  CHECK(fast.work_height == 1080);              // et la source 1080p gardée (A49-02)
  const PlaySettings normal = applied(Preset::Auto, Tier::Normal, 1440, 144.0);
  CHECK(normal.motion_engine == MotionEngine::Rife);
  CHECK(normal.rife_model == "rife_v4.22_lite.onnx");
  const PlaySettings light = applied(Preset::Auto, Tier::Fast, 1080, 60.0);
  CHECK(light.rife_model == "rife_v4.22_lite.onnx");
}

TEST_CASE("la résolution de travail suit l'écran et la tranche", "[auto_config]") {
  CHECK(applied(Preset::Auto, Tier::Normal, 1080, 60.0).work_height == 540);  // 540p -> 1080p
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 60.0).work_height == 720);  // 720p -> 1440p
  CHECK(applied(Preset::Auto, Tier::Fast, 2160, 60.0).work_height == 540);    // la carte légère reste à 540p
  CHECK(applied(Preset::Auto, Tier::Normal, 0, 0.0).work_height == 540);      // écran inconnu : 1080p
  // Jamais plus que la source.
  const core::SourceInfo sd{.width = 640, .height = 480, .fps = 23.976};
  CHECK(applied(Preset::Auto, Tier::Ultra, 2160, 60.0, sd).work_height == 480);
}

TEST_CASE("la résolution de travail suit l'image dans l'écran, arrondie vers le bas (A37-06)",
          "[auto_config]") {
  // Écran 16:10 de portable : une vidéo 16:9 n'en occupe que 1080 lignes, 540p suffit.
  Machine laptop{Tier::Normal, 1200, 60.0};
  laptop.display_width = 1920;
  CHECK(apply_preset(with(Preset::Auto), laptop, k1080p24).work_height == 540);
  // Sans la largeur, la hauteur de l'écran : 600 s'arrondit vers le bas, comme la calibration.
  CHECK(applied(Preset::Auto, Tier::Normal, 1200, 60.0).work_height == 540);
  // Ultra-large 3440 × 1440 : l'image 16:9 prend toute la hauteur.
  Machine wide{Tier::Normal, 1440, 60.0};
  wide.display_width = 3440;
  CHECK(apply_preset(with(Preset::Auto), wide, k1080p24).work_height == 720);
  // Une source 4:3 sur un écran 16:9 garde toute la hauteur.
  Machine screen{Tier::Normal, 1440, 60.0};
  screen.display_width = 2560;
  const core::SourceInfo dvd{.width = 1440, .height = 1080, .fps = 23.976};
  CHECK(apply_preset(with(Preset::Auto), screen, dvd).work_height == 720);
}

TEST_CASE("Qualité et Ultra gardent la source 1080p sur tout écran, avec le SPAN b8f64 (A49-01, A49-02)",
          "[auto_config]") {
  for (const Tier tier : {Tier::Quality, Tier::Ultra}) {
    for (const int display : {1080, 1440, 2160, 0}) {
      const PlaySettings s = applied(Preset::Auto, tier, display, 144.0);
      CHECK(s.work_height == 1080);
      CHECK(chain_work_size(s, 1920, 1080) == WorkSize{1920, 1080});  // rien n'est réduit avant SPAN
      CHECK(s.model == kQualitySpanModel);
      CHECK(s.model.find("b8f64") != std::string::npos);
      CHECK(s.span_precision == PrecisionTier::Performance);  // le FP8 si son reçu passe
    }
  }
  // Les préréglages nommés suivent la même chaîne que le niveau.
  CHECK(applied(Preset::Quality, Tier::Fast, 1080, 60.0).work_height == 1080);
  CHECK(applied(Preset::Ultra, Tier::Fast, 1080, 60.0).model == kQualitySpanModel);
  // Une source 4K est ramenée à 1080p (SPAN rend la 4K) ; une 720p reste en 720p.
  const core::SourceInfo uhd{.width = 3840, .height = 2160, .fps = 23.976};
  CHECK(applied(Preset::Auto, Tier::Ultra, 2160, 60.0, uhd).work_height == 1080);
  const core::SourceInfo hd{.width = 1280, .height = 720, .fps = 23.976};
  CHECK(applied(Preset::Auto, Tier::Quality, 2160, 60.0, hd).work_height == 720);
  // La priorité Fluidité est le seul cas où Qualité réduit la source, et l'utilisateur l'a choisie.
  CHECK(applied(Preset::Auto, Tier::Quality, 1440, 144.0, k1080p24, Priority::Fluidity).work_height == 720);
  CHECK(applied(Preset::Auto, Tier::Quality, 1440, 144.0, k1080p24, Priority::Image).work_height == 1080);
  // Léger et Normal gardent le SPAN par défaut (b5f48) : la calibration, qui mesure, peut monter.
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0).model.empty());
  CHECK(applied(Preset::Auto, Tier::Fast, 1080, 60.0).model.empty());
}

TEST_CASE("l'Auto calibré joue ce que la calibration a mesuré (A49-03, A38-02, A07-11)", "[auto_config]") {
  const CalibratedChain measured{.span = true,
                                 .rife = true,
                                 .height = 1080,
                                 .span_model = std::string(kQualitySpanModel),
                                 .rife_model = "rife_v4.25.onnx"};
  const Machine machine{
      .tier = Tier::Ultra, .display_height = 1440, .display_hz = 144.0, .calibrated = measured};
  const PlaySettings s = apply_preset(with(Preset::Auto), machine, k1080p24);
  CHECK(s.upscale);
  CHECK(s.work_height == 1080);  // calibrer ne fait plus baisser la hauteur sous l'Auto estimé
  CHECK(s.model == kQualitySpanModel);
  CHECK(s.span_precision == PrecisionTier::Performance);  // ADR 0007 reprise
  CHECK(s.rife_model == "rife_v4.25.onnx");               // ni RIFE 4.22 lite imposé
  CHECK(s.auto_free);
  CHECK(s.rife_auto);
  CHECK(apply_preset(s, machine, k1080p24) == s);  // idempotente
  // La mesure borne la hauteur, la source aussi.
  Machine low = machine;
  low.calibrated->height = 540;
  CHECK(apply_preset(with(Preset::Auto), low, k1080p24).work_height == 540);
  const core::SourceInfo sd{.width = 640, .height = 480, .fps = 23.976};
  CHECK(apply_preset(with(Preset::Auto), machine, sd).work_height == 480);
  // Une calibration sans RIFE coupe l'interpolation ; sans SPAN, l'agrandissement.
  Machine no_rife = machine;
  no_rife.calibrated->rife = false;
  const PlaySettings still = apply_preset(with(Preset::Auto), no_rife, k1080p24);
  CHECK_FALSE(still.rife_auto);
  CHECK(still.rife_factor == 0);
  Machine no_span = machine;
  no_span.calibrated->span = false;
  CHECK_FALSE(apply_preset(with(Preset::Auto), no_span, k1080p24).upscale);
  // Sans SPAN, la source n'est pas réduite : la calibration l'a mesurée à sa taille (B07-01).
  const PlaySettings rife_only = apply_preset(with(Preset::Auto), no_span, k1080p24);
  CHECK(chain_work_size(rife_only, 1920, 1080) == WorkSize{1920, 1080});
  Machine weak = no_span;
  weak.tier = Tier::Economy;
  CHECK(chain_work_size(apply_preset(with(Preset::Auto), weak, k1080p24), 1920, 1080) ==
        WorkSize{1920, 1080});
  weak.calibrated->rife = false;  // rien ne tient
  CHECK(chain_work_size(apply_preset(with(Preset::Auto), weak, k1080p24), 1920, 1080) ==
        WorkSize{1920, 1080});
  // Une carte estimée Économe que la calibration a vue tenir SPAN suit la mesure.
  Machine economy = machine;
  economy.tier = Tier::Economy;
  CHECK(apply_preset(with(Preset::Auto), economy, k1080p24).upscale);
  // La calibration ne vaut que pour Auto : un préréglage nommé garde sa chaîne.
  CHECK(apply_preset(with(Preset::Normal), low, k1080p24).work_height == 720);
}

// Le README promet « jusqu'à ×5 » (C38-03) : le plafond de Fluidité part de ×4 et monte d'un cran.
TEST_CASE("Fluidité plafonne le multiplicateur à ×5, jamais ×6", "[auto_config]") {
  for (const Tier tier : {Tier::Quality, Tier::Ultra})
    CHECK(applied(Preset::Auto, tier, 2160, 144.0, k1080p24, Priority::Fluidity).rife_factor == 5);
}

TEST_CASE("la priorité décale d'un cran", "[auto_config]") {
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0, k1080p24, Priority::Fluidity).work_height == 540);
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0, k1080p24, Priority::Image).work_height == 900);
  const int balance = applied(Preset::Auto, Tier::Normal, 1440, 144.0).rife_factor;
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0, k1080p24, Priority::Fluidity).rife_factor >=
        balance);
  // SPAN au cran Performance : le FP8 quand son reçu passe la porte, sinon le FP16 (ADR 0007 reprise).
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0).span_precision == PrecisionTier::Performance);
  CHECK(applied(Preset::Fast, Tier::Normal, 1440, 144.0).span_precision == PrecisionTier::Performance);
}

TEST_CASE("le multiplicateur suit l'écran, une source rapide n'est pas interpolée", "[auto_config]") {
  const PlaySettings s = applied(Preset::Auto, Tier::Quality, 2160, 165.0);  // 165 / 23,976 = 6, plafond 4
  CHECK(s.rife_auto);
  CHECK(s.rife_factor == 4);
  const core::SourceInfo fps60{.width = 1920, .height = 1080, .fps = 60.0};
  const PlaySettings rate = applied(Preset::Auto, Tier::Quality, 2160, 60.0, fps60);
  CHECK(rate.rife_factor == 0);
  CHECK_FALSE(rate.rife_auto);
  // P3.14 : 59,94 Hz sur une source à 30 i/s fait ×2 (59,94 / 30 = 1,998, arrondi avec tolérance).
  const core::SourceInfo fps30{.width = 1920, .height = 1080, .fps = 30.0};
  CHECK(applied(Preset::Auto, Tier::Quality, 2160, 59.94, fps30).rife_factor == 2);
}

TEST_CASE("Économe ne réduit ni n'agrandit ni n'interpole", "[auto_config]") {
  const PlaySettings s = applied(Preset::Economy, Tier::Ultra, 2160, 165.0);
  CHECK_FALSE(s.upscale);
  CHECK(s.work_height == 0);  // P3.20 : une source 4K reste en 4K
  CHECK(work_size(3840, 2160, s.work_height) == WorkSize{3840, 2160});
  CHECK(s.rife_factor == 0);
  CHECK_FALSE(s.rife_auto);
}

TEST_CASE("un préréglage fixe la tranche quelle que soit la machine", "[auto_config]") {
  CHECK(*chain_level(Preset::Fast, Tier::Ultra) == Tier::Fast);
  CHECK(*chain_level(Preset::Quality, Tier::Economy) == Tier::Quality);
  CHECK(*chain_level(Preset::Auto, Tier::Normal) == Tier::Normal);
  CHECK_FALSE(chain_level(Preset::Custom, Tier::Normal));
}

TEST_CASE("l'application est idempotente et ne touche pas au reste", "[auto_config]") {
  PlaySettings base = with(Preset::Auto);
  base.volume = 0.3F;
  base.true_hdr_peak = 800;
  base.language_presets = {{"jpn", "eng"}};
  const Machine machine{Tier::Quality, 1440, 144.0};
  const PlaySettings once = apply_preset(base, machine, k1080p24);
  CHECK(apply_preset(once, machine, k1080p24) == once);
  CHECK(once.volume == 0.3F);
  CHECK(once.true_hdr_peak == 800);
  CHECK(once.language_presets == base.language_presets);
}

TEST_CASE("noms des préréglages et des priorités", "[auto_config]") {
  for (int i = 0; i < kPresetCount; ++i) {
    const auto preset = static_cast<Preset>(i);
    CHECK(preset_from_key(key(preset)) == preset);
  }
  for (int i = 0; i < kPriorityCount; ++i) {
    const auto priority = static_cast<Priority>(i);
    CHECK(priority_from_key(key(priority)) == priority);
  }
  CHECK_FALSE(preset_from_key("turbo"));
  CHECK(label(Preset::Custom) == "Custom");
}

TEST_CASE("play.toml garde le préréglage et la priorité", "[auto_config][settings]") {
  PlaySettings s = with(Preset::Quality, Priority::Image);
  const auto read = parse_settings(format_settings(s));
  REQUIRE(read);
  CHECK(read->preset == Preset::Quality);
  CHECK(read->priority == Priority::Image);
  // Un play.toml d'avant la section : Personnalisé.
  const auto old = parse_settings("version = 1\n");
  REQUIRE(old);
  CHECK(old->preset == Preset::Custom);
  CHECK_FALSE(parse_settings("version = 1\n[auto]\npreset = \"turbo\"\n"));
}

TEST_CASE("l'interpolation est en Auto par défaut et après le benchmark", "[auto_config]") {
  CHECK(PlaySettings{}.rife_auto);
  const PlaySettings measured = applied(Preset::Auto, Tier::Quality, 1440, 144.0);
  CHECK(measured.rife_auto);
  CHECK_FALSE(applied(Preset::Auto, Tier::Economy, 1080, 60.0).rife_auto);  // Économe : rien à interpoler
}

TEST_CASE("un réglage de chaîne fait à la main se voit, pas un réglage de lecture", "[auto_config]") {
  const ajp::app::play::PlaySettings base;
  ajp::app::play::PlaySettings moved = base;
  CHECK_FALSE(ajp::app::play::preset_chain_differs(base, moved));
  moved.volume = 0.5F;
  CHECK_FALSE(ajp::app::play::preset_chain_differs(base, moved));
  moved.work_height = base.work_height == 540 ? 720 : 540;
  CHECK(ajp::app::play::preset_chain_differs(base, moved));
  moved = base;
  moved.keep_poses = !base.keep_poses;
  CHECK(ajp::app::play::preset_chain_differs(base, moved));
  // Les réglages avancés de la chaîne comptent aussi : l'infobulle promet Personnalisé (C23-06).
  for (const auto change :
       {+[](ajp::app::play::PlaySettings& s) { s.rife_fused = !s.rife_fused; },
        +[](ajp::app::play::PlaySettings& s) { s.drba_fused = !s.drba_fused; },
        +[](ajp::app::play::PlaySettings& s) { s.smooth_drawings = !s.smooth_drawings; },
        +[](ajp::app::play::PlaySettings& s) { s.cubic_warp = !s.cubic_warp; },
        +[](ajp::app::play::PlaySettings& s) { s.pair_scale = !s.pair_scale; },
        +[](ajp::app::play::PlaySettings& s) { s.auto_cap = s.auto_cap == 60 ? 120 : 60; }}) {
    moved = base;
    change(moved);
    CHECK(ajp::app::play::preset_chain_differs(base, moved));
  }
  // Le mode auto Rapide n'appartient à la chaîne que sous l'Auto calibré, qui l'écrit (A07-06).
  moved = base;
  moved.auto_free = !base.auto_free;
  CHECK_FALSE(ajp::app::play::preset_chain_differs(base, moved));
  CHECK(ajp::app::play::preset_chain_differs(base, moved, true));
}

TEST_CASE("auto : ce que l'exploration a trouvé relève la chaîne du niveau, seulement en Auto",
          "[auto_config][exploration]") {
  using namespace ajp::app::play;
  const core::SourceInfo source{.width = 1920, .height = 1080, .fps = 24.0};
  Machine machine{.tier = Tier::Normal, .display_height = 1080, .display_hz = 165.0};
  PlaySettings settings;
  settings.preset = Preset::Auto;
  const PlaySettings plain = apply_preset(settings, machine, source);
  machine.explored_work_height = 900;
  machine.explored_rife_model = "rife_v4.25.onnx";
  const PlaySettings explored = apply_preset(settings, machine, source);
  CHECK(explored.work_height == 900);
  CHECK(explored.work_height > plain.work_height);
  CHECK(explored.rife_model == "rife_v4.25.onnx");
  // La source ne monte pas plus haut qu'elle-même.
  const core::SourceInfo small{.width = 1280, .height = 720, .fps = 24.0};
  CHECK(apply_preset(settings, machine, small).work_height <= 720);
  // Un préréglage choisi à la main n'est pas touché par l'exploration.
  settings.preset = Preset::Normal;
  CHECK(apply_preset(settings, machine, source).work_height == plain.work_height);
}

TEST_CASE("DistilDRBA attend sa porte de qualité", "[auto_config]") {
  static_assert(!kDistilDrbaInAuto);
  for (const Tier tier : {Tier::Fast, Tier::Normal, Tier::Quality, Tier::Ultra})
    CHECK(applied(Preset::Auto, tier, 1440, 144.0).motion_engine == MotionEngine::Rife);
}

TEST_CASE("un autre écran rouvre la chaîne d'un préréglage (P3.21)", "[auto_config]") {
  CHECK(display_moves_chain(Preset::Auto, {.opened_height = 1080, .now_height = 2160}));
  CHECK_FALSE(display_moves_chain(Preset::Auto, {.opened_height = 1440, .now_height = 1440}));
  CHECK_FALSE(display_moves_chain(Preset::Custom, {.opened_height = 1080, .now_height = 2160}));
  CHECK_FALSE(display_moves_chain(Preset::Auto, {.opened_height = 0, .now_height = 2160}));
}

TEST_CASE("un écran d'une autre fréquence rouvre la chaîne quand le multiplicateur change (A37-07)",
          "[auto_config]") {
  // Deux écrans 1440p, 60 Hz puis 165 Hz, source 24 i/s : ×2 puis ×6 voulu.
  CHECK(display_moves_chain(
      Preset::Auto,
      {.opened_height = 1440, .opened_hz = 60.0, .source_fps = 23.976, .now_height = 1440, .now_hz = 165.0}));
  // 144 et 165 Hz demandent tous deux ×6 : rien à rouvrir.
  CHECK_FALSE(display_moves_chain(Preset::Auto, {.opened_height = 1440,
                                                 .opened_hz = 144.0,
                                                 .source_fps = 23.976,
                                                 .now_height = 1440,
                                                 .now_hz = 165.0}));
  // Source 30 i/s ouverte sur une télé à 50 Hz (×1, pas d'interpolation), puis écran 144 Hz : ×4.
  CHECK(display_moves_chain(
      Preset::Normal,
      {.opened_height = 1080, .opened_hz = 50.0, .source_fps = 30.0, .now_height = 1080, .now_hz = 144.0}));
  CHECK_FALSE(display_moves_chain(
      Preset::Custom,
      {.opened_height = 1080, .opened_hz = 50.0, .source_fps = 30.0, .now_height = 1080, .now_hz = 144.0}));
}

TEST_CASE("l'Auto calibré suit la priorité Fluidité d'un palier plus bas (B07-09)", "[auto_config]") {
  const CalibratedChain measured{.span = true, .rife = true, .height = 1080};
  const Machine machine{
      .tier = Tier::Ultra, .display_height = 1440, .display_hz = 144.0, .calibrated = measured};
  CHECK(apply_preset(with(Preset::Auto, Priority::Balance), machine, k1080p24).work_height == 1080);
  CHECK(apply_preset(with(Preset::Auto, Priority::Image), machine, k1080p24).work_height == 1080);
  // Le palier mesuré sous 1080p : 720p, pas 900p, que la calibration n'essaie jamais (C07-01).
  CHECK(apply_preset(with(Preset::Auto, Priority::Fluidity), machine, k1080p24).work_height == 720);
  Machine lowest = machine;
  lowest.calibrated->height = 480;
  CHECK(apply_preset(with(Preset::Auto, Priority::Fluidity), lowest, k1080p24).work_height == 360);
}

TEST_CASE("C07-01 : la Fluidité ne monte jamais au-dessus de la mesure", "[auto_config]") {
  const Machine small{.tier = Tier::Fast,
                      .display_height = 1080,
                      .display_hz = 60.0,
                      .calibrated = CalibratedChain{.span = true, .rife = true, .height = 360}};
  CHECK(apply_preset(with(Preset::Auto, Priority::Fluidity), small, k1080p24).work_height <= 360);
  CHECK(apply_preset(with(Preset::Auto, Priority::Balance), small, k1080p24).work_height == 360);
}

TEST_CASE("C07-02 : une source à 60 i/s ne garde pas la hauteur mesurée à 24 i/s", "[auto_config]") {
  const core::SourceInfo k1080p60{.width = 1920, .height = 1080, .fps = 59.94};
  const Machine fast{.tier = Tier::Quality,
                     .display_height = 1440,
                     .display_hz = 144.0,
                     .calibrated = CalibratedChain{.span = true,
                                                   .rife = true,
                                                   .height = 1080,
                                                   .span_model = std::string(kQualitySpanModel),
                                                   .rife_model = std::string(kQualityRifeModel)}};
  const PlaySettings out = apply_preset(with(Preset::Auto), fast, k1080p60);
  CHECK(out.upscale);
  CHECK(out.work_height == 720);  // 1080² × 24 × 1,25 : 763 lignes au plus, le palier mesuré en dessous
  // Même règle sans calibration : Qualité garde la source à 24 i/s, pas à 60.
  const PlaySettings quality =
      apply_preset(with(Preset::Quality), Machine{Tier::Quality, 1440, 144.0}, k1080p60);
  CHECK(quality.work_height == 720);
  CHECK(apply_preset(with(Preset::Quality), Machine{Tier::Quality, 1440, 144.0}, k1080p24).work_height ==
        1080);
  // 30 i/s (29,97) : 25 % de pixels par seconde en plus, la hauteur mesurée reste.
  const core::SourceInfo k1080p30{.width = 1920, .height = 1080, .fps = 29.97};
  CHECK(apply_preset(with(Preset::Auto), fast, k1080p30).work_height == 1080);
  // Cadence inconnue : rien ne change.
  const core::SourceInfo unknown{.width = 1920, .height = 1080, .fps = 0.0};
  CHECK(apply_preset(with(Preset::Auto), fast, unknown).work_height == 1080);
}

TEST_CASE("C07-04 : sous l'Auto calibré, la priorité Image prend le plus haut SPAN seul", "[auto_config]") {
  // SPAN seul tient 1080p, SPAN + RIFE seulement 480p : la recommandation met la fluidité d'abord.
  const Machine machine{
      .tier = Tier::Normal,
      .display_height = 1080,
      .display_hz = 60.0,
      .calibrated = CalibratedChain{.span = true, .rife = true, .height = 480, .span_height = 1080}};
  const PlaySettings balance = apply_preset(with(Preset::Auto, Priority::Balance), machine, k1080p24);
  CHECK(balance.work_height == 480);
  const PlaySettings image = apply_preset(with(Preset::Auto, Priority::Image), machine, k1080p24);
  CHECK(image.upscale);
  CHECK(image.work_height == 1080);
  CHECK(
      image.auto_free);  // RIFE en mode Rapide, qui ne garde que les images que la carte a le temps de faire
  CHECK(image.rife_auto);
  // RIFE seul recommandé (SPAN + RIFE ne tient nulle part) : Image rend SPAN seul à son palier.
  Machine rife_only = machine;
  rife_only.calibrated = CalibratedChain{.span = false, .rife = true, .height = 0, .span_height = 720};
  const PlaySettings no_span = apply_preset(with(Preset::Auto, Priority::Balance), rife_only, k1080p24);
  CHECK_FALSE(no_span.upscale);
  const PlaySettings span_back = apply_preset(with(Preset::Auto, Priority::Image), rife_only, k1080p24);
  CHECK(span_back.upscale);
  CHECK(span_back.work_height == 720);
}

TEST_CASE("un préréglage restaure les traits, comme la calibration les mesure (B07-08)", "[auto_config]") {
  PlaySettings s = with(Preset::Normal);
  s.restore_lines = false;
  const PlaySettings out = apply_preset(s, Machine{Tier::Normal, 1080, 60.0}, k1080p24);
  CHECK(out.restore_lines);
  CHECK(span_runs(out, 1080));
  PlaySettings moved = out;
  moved.restore_lines = false;
  CHECK(preset_chain_differs(out, moved));
}
