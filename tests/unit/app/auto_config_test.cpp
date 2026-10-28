#include "app/play/auto_config.h"

#include <catch2/catch_test_macros.hpp>

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
  s.motion_engine = MotionEngine::Gmfss;
  CHECK(apply_preset(s, Machine{Tier::Ultra, 2160, 165.0}, k1080p24) == s);
  CHECK(PlaySettings{}.preset == Preset::Custom);  // les réglages d'avant ne bougent pas
}

TEST_CASE("Auto suit la tranche de la machine", "[auto_config]") {
  const PlaySettings ultra = applied(Preset::Auto, Tier::Ultra, 2160, 165.0);
  CHECK(ultra.motion_engine == MotionEngine::GmfssUnion);
  CHECK(ultra.upscale);
  CHECK(ultra.rife_auto);
  const PlaySettings fast = applied(Preset::Auto, Tier::Fast, 1440, 144.0);
  CHECK(fast.motion_engine == MotionEngine::DistilDrba);
  CHECK(fast.work_height == 720);
  const PlaySettings normal = applied(Preset::Auto, Tier::Normal, 1440, 144.0);
  CHECK(normal.motion_engine == MotionEngine::Rife);
  CHECK(normal.rife_model == "rife_v4.22_lite.onnx");
  const PlaySettings light = applied(Preset::Auto, Tier::Light, 1080, 60.0);
  CHECK(light.rife_model == "rife_v4.22_lite.onnx");
}

TEST_CASE("la résolution de travail suit l'écran et la tranche", "[auto_config]") {
  CHECK(applied(Preset::Auto, Tier::Normal, 1080, 60.0).work_height == 540);  // 540p -> 1080p
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 60.0).work_height == 720);  // 720p -> 1440p
  CHECK(applied(Preset::Auto, Tier::Fast, 2160, 60.0).work_height == 720);    // plafond de la tranche
  CHECK(applied(Preset::Auto, Tier::Light, 2160, 60.0).work_height == 540);   // la carte légère reste à 540p
  CHECK(applied(Preset::Auto, Tier::Normal, 0, 0.0).work_height == 540);      // écran inconnu : 1080p
  // Jamais plus que la source.
  const core::SourceInfo sd{.width = 640, .height = 480, .fps = 23.976};
  CHECK(applied(Preset::Auto, Tier::Ultra, 2160, 60.0, sd).work_height == 480);
}

TEST_CASE("la priorité décale d'un cran", "[auto_config]") {
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0, k1080p24, Priority::Fluidity).work_height == 540);
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0, k1080p24, Priority::Image).work_height == 900);
  const int balance = applied(Preset::Auto, Tier::Normal, 1440, 144.0).rife_factor;
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0, k1080p24, Priority::Fluidity).rife_factor >=
        balance);
  CHECK(applied(Preset::Auto, Tier::Ultra, 2160, 165.0, k1080p24, Priority::Image).gmfss_precision ==
        PrecisionTier::Quality);
  CHECK(applied(Preset::Auto, Tier::Ultra, 2160, 165.0).gmfss_precision == PrecisionTier::Balanced);
  // L'automatique ne choisit jamais le FP8 : il se demande.
  CHECK(applied(Preset::Auto, Tier::Normal, 1440, 144.0).span_precision == PrecisionTier::Balanced);
}

TEST_CASE("le multiplicateur suit l'écran, une source rapide n'est pas interpolée", "[auto_config]") {
  const PlaySettings s = applied(Preset::Auto, Tier::Fast, 2160, 165.0);  // 165 / 23,976 = 6, plafond 4
  CHECK(s.rife_auto);
  CHECK(s.rife_factor == 4);
  const core::SourceInfo fps60{.width = 1920, .height = 1080, .fps = 60.0};
  const PlaySettings rate = applied(Preset::Auto, Tier::Fast, 2160, 60.0, fps60);
  CHECK(rate.rife_factor == 0);
  CHECK_FALSE(rate.rife_auto);
}

TEST_CASE("Économe ne réduit ni n'agrandit ni n'interpole", "[auto_config]") {
  const PlaySettings s = applied(Preset::Economy, Tier::Ultra, 2160, 165.0);
  CHECK_FALSE(s.upscale);
  CHECK(s.rife_factor == 0);
  CHECK_FALSE(s.rife_auto);
}

TEST_CASE("un préréglage fixe la tranche quelle que soit la machine", "[auto_config]") {
  CHECK(*chain_level(Preset::Fast, Tier::Ultra) == Tier::Light);
  CHECK(*chain_level(Preset::Quality, Tier::Economy) == Tier::Fast);
  CHECK(*chain_level(Preset::Auto, Tier::Normal) == Tier::Normal);
  CHECK_FALSE(chain_level(Preset::Custom, Tier::Normal));
}

TEST_CASE("l'application est idempotente et ne touche pas au reste", "[auto_config]") {
  PlaySettings base = with(Preset::Auto);
  base.volume = 0.3F;
  base.true_hdr = true;
  base.language_presets = {{"jpn", "eng"}};
  const Machine machine{Tier::Fast, 1440, 144.0};
  const PlaySettings once = apply_preset(base, machine, k1080p24);
  CHECK(apply_preset(once, machine, k1080p24) == once);
  CHECK(once.volume == 0.3F);
  CHECK(once.true_hdr);
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
  CHECK(label(Preset::Custom) == "Personnalisé");
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
  const PlaySettings measured = applied(Preset::Auto, Tier::Fast, 1440, 144.0);
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
}
