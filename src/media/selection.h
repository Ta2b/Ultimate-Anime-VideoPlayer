// Choix des pistes à l'ouverture d'un fichier, par priorité de langues.
#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "media/track.h"

namespace ajp::media {

// Un préréglage : l'audio et les sous-titres voulus ensemble. audio vide : la piste que le fichier
// préfère (sa langue ne compte pas) ; subtitles vide : aucun sous-titre.
struct LanguagePreset {
  std::string audio;
  std::string subtitles;
  friend bool operator==(const LanguagePreset&, const LanguagePreset&) = default;
};

// Langues voulues, de la plus voulue à la moins voulue : « jpn », « ja », « en-US »…
// presets : des couples audio + sous-titres, du préféré au moins préféré ; le premier dont l'audio
// ET les sous-titres existent dans le fichier est pris tel quel (sans sous-titres si son rang le
// dit). Si aucun n'est entier, on retombe sur les listes ci-dessous, complétées des langues des
// préréglages, piste par piste.
struct LanguagePriorities {
  std::vector<std::string> audio;
  std::vector<std::string> subtitles;
  std::vector<LanguagePreset> presets;
};

// Index de flux choisis ; std::nullopt : aucune piste de ce type.
struct TrackChoice {
  std::optional<int> video;
  std::optional<int> audio;
  std::optional<int> subtitle;
};

// Le module de sous-titres sait lire ce codec (ASS, SSA, SubRip, texte). Un codec inconnu (vide)
// compte comme lisible. Les pistes illisibles ne sont jamais choisies automatiquement.
bool subtitle_readable(std::string_view codec);

// Vidéo : la piste par défaut, sinon la première, jamais une pochette.
// La langue d'une piste : sa balise, sinon celle que nomme son titre (track_language.h).
// Audio : meilleure langue, puis drapeau défaut, puis ordre du fichier.
// Sous-titres : meilleure langue, en préférant une piste de dialogue (ni « Signs & Songs », ni
// commentaire, ni SDH), puis une piste complète à une piste forcée ; sans langue voulue, la piste
// forcée dans la langue de l'audio, sinon la piste par défaut, sinon aucune. Les pistes que le
// module de sous-titres ne sait pas lire (PGS, VobSub…) ne sont jamais choisies.
// Le module de sous-titres sait lire ce codec (ASS, SSA, SubRip, texte). Un codec inconnu (vide)
// compte comme lisible.
bool subtitle_readable(std::string_view codec);

TrackChoice choose_tracks(std::span<const TrackInfo> tracks, const LanguagePriorities& priorities);

}  // namespace ajp::media
