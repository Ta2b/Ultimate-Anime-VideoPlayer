// Ce que le poste et l'écran de la fenêtre disent d'eux-mêmes : rafraîchissement et marge HDR
// selon SDL, nom de l'écran (pour son EDID), GPU selon Vulkan, moteurs présents pour l'écran de
// bienvenue.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <SDL3/SDL_video.h>

#include "app/play/engine_install.h"
#include "app/play/first_run.h"
#include "app/play/options.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"

namespace ajp::app::play {

// Période de rafraîchissement de l'écran de la fenêtre ; 0 si inconnue (la session la mesure
// alors sur les présentations).
std::int64_t display_period_us(SDL_Window* window);
// Le même rafraîchissement, exact quand SDL le donne en fraction (59,94 Hz = 60000/1001) ; 0 si
// inconnu.
double display_refresh_hz(SDL_Window* window);
// La hauteur en pixels du mode de l'écran de la fenêtre ; 0 si inconnue.
int display_height(SDL_Window* window);
// Crête vue par le contenu : la marge HDR de SDL3 × 203 nits ; rien si l'écran n'est pas en HDR.
std::optional<double> content_peak_nits(SDL_Window* window);
// Le nom de l'écran de la fenêtre selon SDL, pour choisir son EDID.
std::string display_name(SDL_Window* window);
GpuFacts gpu_facts(const PlaceboScreen& screen);
// Le SDK Maxine est utilisable : importé dans le magasin du lecteur, ou présent à importer. La
// même définition que l'écran de bienvenue, pour le panneau de réglages.
bool maxine_available(const PlayOptions& options, const PlayerPaths& paths);
// Un backend TensorRT est installé : seul lui ouvre RTX HDR.
bool tensorrt_backend_present(const PlayerPaths& paths);
// Les backends présents ou téléchargeables (catalog), le SDK Maxine importé ou à importer.
EnginesHere engines_here(const PlayOptions& options, const PlayerPaths& paths, const EngineCatalog& catalog,
                         std::string_view gpu_name);

}  // namespace ajp::app::play
