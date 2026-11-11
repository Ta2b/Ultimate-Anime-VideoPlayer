#pragma once

#include <filesystem>

#include "app/play/engine_install.h"
#include "app/play/player.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

class PlaceboScreen;

// Les champs du menu qui ne dépendent que du poste (carte, tranche, sorties son, dossiers, lecteur par
// défaut) : les mêmes à l'accueil et en lecture. settings_dir : le dossier de play.toml et de bench.toml.
void fill_host_choices(SettingsChoices& choices, const PlaceboScreen& screen, const PlayerPaths& paths,
                       const EngineCatalog& catalog, const std::filesystem::path& settings_dir);

// Ce qui change pendant la séance : les sorties son, la place libre du dossier des moteurs, et si AJP est le
// lecteur par défaut (la première réponse du système est asynchrone : relue, elle finit par paraître).
void refresh_host_choices(SettingsChoices& choices, const EngineCatalog& catalog);

}  // namespace ajp::app::play
