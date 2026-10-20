// Les moteurs cochés à l'écran de bienvenue et à télécharger (et RTX HDR sous Linux, depuis NGC),
// installés l'un après l'autre pendant que la fenêtre montre l'avancement. Un moteur qui échoue
// sort des moteurs choisis (Vulkan, ou le HDR adaptatif, prend le relais) et l'écran le dit ;
// fermer la fenêtre arrête tout.
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

// false : fenêtre fermée pendant les téléchargements. engines perd les moteurs non installés.
// sdk_dir : le magasin des SDK TrueHDR, où RTX HDR s'importe.
std::expected<bool, Refusal> install_chosen(PlaceboScreen& screen, PlayerUi& ui, const EngineCatalog& catalog,
                                            const FirstRunView& view, core::Settings& engines,
                                            const std::filesystem::path& sdk_dir);

}  // namespace ajp::app::play
