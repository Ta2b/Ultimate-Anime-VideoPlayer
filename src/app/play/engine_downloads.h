// Les moteurs cochés à l'écran de bienvenue et à télécharger, installés l'un après l'autre pendant que la
// fenêtre montre l'avancement. Un moteur qui échoue sort des moteurs choisis (Vulkan prend le relais) et
// l'écran le dit ; fermer la fenêtre arrête tout.
#pragma once

#include <expected>
#include <filesystem>
#include <vector>

#include "app/play/engine_install.h"
#include "app/play/first_run.h"
#include "app/play/placebo_screen.h"
#include "app/play/ui.h"
#include "core/settings.h"

namespace ajp::app::play {

// false : fenêtre fermée pendant les téléchargements. engines perd les moteurs non installés, que failed
// (s'il est donné) liste : l'assistant ne calibre pas sous Vulkan un poste dont le moteur recommandé a
// échoué.
std::expected<bool, Refusal> install_chosen(PlaceboScreen& screen, PlayerUi& ui, const EngineCatalog& catalog,
                                            const FirstRunView& view, core::Settings& engines,
                                            std::vector<core::EngineId>* failed = nullptr);

// Les moteurs choisis et installés dont le module parle une autre interface que ce lecteur
// (gpu::kBackendAbi : installés par une version précédente) se réinstallent depuis le paquet de
// cette version, sous la licence déjà acceptée pour eux. Une licence changée attend l'onglet Moteurs ;
// un échec laisse Vulkan prendre le relais, comme au premier lancement. false : fenêtre fermée. updated
// passe à vrai quand un module a été remplacé : l'ancien reste chargé dans ce processus (libnvinfer ne se
// décharge pas) et masquerait le nouveau, le lecteur doit se relancer.
std::expected<bool, Refusal> update_stale_engines(PlaceboScreen& screen, PlayerUi& ui,
                                                  const EngineCatalog& catalog, const core::Settings& engines,
                                                  bool& updated);

}  // namespace ajp::app::play
