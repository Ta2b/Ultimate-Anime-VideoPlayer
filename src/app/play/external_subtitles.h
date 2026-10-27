// Les sous-titres à côté de la vidéo : « Épisode 01.fr.srt » ou « Épisode 01.ass » pour
// « Épisode 01.mkv », proposés dans le menu comme les pistes du conteneur.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace ajp::app::play {

struct ExternalSubtitle {
  std::filesystem::path path;
  std::string label;  // « Externe · Français · Épisode 01.fr.srt »
};

// Index de menu de la première piste externe : au-delà de tout flux de conteneur.
inline constexpr int kExternalSubtitleBase = 100'000;
inline constexpr std::size_t kMaxExternalSubtitles = 16;

// .ass, .ssa et .srt du dossier de `video` dont le nom commence par celui de la vidéo, suivi de rien
// ou d'un point (« .fr », « .eng.forced »). Triés par nom, au plus kMaxExternalSubtitles ; vide si le
// dossier ne se lit pas.
std::vector<ExternalSubtitle> find_external_subtitles(const std::filesystem::path& video);

}  // namespace ajp::app::play
