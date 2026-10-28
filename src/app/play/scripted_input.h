// Les entrées d'essai de --keys : touches et souris poussées dans la file d'événements SDL, comme
// le clavier et la souris réels ; et « export », l'export lancé sans clic dans l'onglet Export.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <SDL3/SDL_video.h>

#include "app/play/export_job.h"
#include "media/track.h"

namespace ajp::app::play {

// « mouse=XxY » ou « click=XxY » : le point de la fenêtre, en fractions de sa taille.
std::optional<std::pair<float, float>> mouse_point(std::string_view text);
// Une touche (« space », « ctrl+2 »), ou la souris déplacée (« mouse=XxY ») ou cliquée
// (« click=XxY »), ou un fichier glissé sur la fenêtre (« drop=CHEMIN ») ; une touche inconnue est
// signalée au journal et ignorée.
void press(const std::string& name, SDL_Window* window);
// « export » ou « export=FICHIER.mkv » : l'export de la chaîne en cours (running), en HEVC très
// haute qualité à la taille de la sortie, vers FICHIER ou le nom proposé à côté de la source
// (default_export_path) ; rien pour une touche.
std::optional<ExportJob> scripted_export(std::string_view name, const PlaySettings& running,
                                         const media::TrackInfo& video, const std::filesystem::path& source);

}  // namespace ajp::app::play
