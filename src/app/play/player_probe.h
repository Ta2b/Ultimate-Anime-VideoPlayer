// La sonde d'une vidéo avant sa lecture : la piste vidéo que la source choisira, lue d'avance (sa taille fixe
// le moteur et l'anneau), hors du thread de la fenêtre et bornée.
#pragma once

#include <chrono>
#include <expected>
#include <filesystem>
#include <optional>

#include "app/play/pumped_task.h"
#include "core/types.h"
#include "media/track.h"

namespace ajp::app::play {

// Un partage réseau qui ne répond plus rend la main au bout de ce délai, avec une raison (P2.15).
inline constexpr auto kProbeLimit = std::chrono::seconds(20);

// Sur le thread appelant : le démultiplexeur ouvert, la piste vidéo choisie, et les refus d'avance (Dolby
// Vision 5, audio seul, format que la chaîne refuserait).
std::expected<media::TrackInfo, Refusal> probe_once(const std::filesystem::path& file);

// probe_once sur un thread, borné à kProbeLimit ; window_open : la fenêtre est pompée et peut se fermer.
std::expected<media::TrackInfo, Refusal> probe_video(const std::filesystem::path& file, bool window_open);

// C18-01 : la sonde du fichier donné au lancement part avant l'ouverture de la fenêtre, puis s'attend une
// fois la fenêtre ouverte (finish_given_probe), pompée : un fichier sur un NAS ou un disque en veille ne
// laisse plus jusqu'à 20 s sans rien à l'écran. Rien si le thread est refusé : la sonde se fait alors à la
// première vidéo.
using GivenProbe = PumpedTask<media::TrackInfo>;
std::optional<GivenProbe> start_given_probe(const std::filesystem::path& file);
std::expected<media::TrackInfo, Refusal> finish_given_probe(GivenProbe& probe);

}  // namespace ajp::app::play
