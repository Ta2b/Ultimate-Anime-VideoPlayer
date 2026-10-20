// Choix des pistes à l'ouverture d'un fichier, par priorité de langues.
#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "media/track.h"

namespace ajp::media {

// Langues voulues, de la plus voulue à la moins voulue : « jpn », « ja », « en-US »…
struct LanguagePriorities {
  std::vector<std::string> audio;
  std::vector<std::string> subtitles;
};

// Index de flux choisis ; std::nullopt : aucune piste de ce type.
struct TrackChoice {
  std::optional<int> video;
  std::optional<int> audio;
  std::optional<int> subtitle;
};

// Vidéo : la piste par défaut, sinon la première, jamais une pochette.
// La langue d'une piste : sa balise, sinon celle que nomme son titre (track_language.h).
// Audio : meilleure langue, puis drapeau défaut, puis ordre du fichier.
// Sous-titres : meilleure langue, en préférant une piste complète à une piste
// forcée ; sans langue voulue, la piste forcée dans la langue de l'audio, sinon
// la piste par défaut, sinon aucune.
TrackChoice choose_tracks(std::span<const TrackInfo> tracks, const LanguagePriorities& priorities);

}  // namespace ajp::media
