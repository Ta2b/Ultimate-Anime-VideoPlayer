// Ce que le menu des paramètres sait d'une ouverture de la chaîne (player_run.cpp) : modèles, crans, moteurs,
// écran, étages refusés.
#pragma once

#include <functional>
#include <string>

#include "app/play/chain_setup.h"
#include "app/play/gmfss_model.h"
#include "app/play/player_engines.h"
#include "app/play/settings_panel.h"

namespace kirari::app::play {

std::function<void(SettingsChoices&, const PlaySettings&)> motion_models_refresh(
    const PlayOptions& options, const PlayerPaths& paths, const GpuFacts& gpu, bool fp8_engine,
    bool gmfss_supported, GmfssMenuProbe probe);

SettingsChoices chain_choices(Player& p, const ChainEngines& c, const SpanPrepare& span_prepare,
                              int auto_peak, bool source_hdr = false);

}  // namespace kirari::app::play
