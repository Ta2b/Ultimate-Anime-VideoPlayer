// Les pistes audio et sous-titres qu'on peut choisir pendant la lecture, telles que la barre les
// montre : un libellé lisible (langue, titre) et la piste choisie.
#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "media/track.h"

namespace ajp::app::play {

struct TrackEntry {
  int index = -1;     // index du flux dans le conteneur
  std::string label;  // « Français · CR_French »
};

struct TrackMenu {
  std::vector<TrackEntry> audio;
  std::vector<TrackEntry> subtitles;
  std::optional<int> audio_selected;     // index du flux
  std::optional<int> subtitle_selected;  // rien : sous-titres désactivés
};

// Nom français d'une langue du conteneur (« jpn », « fr », « en-US ») ; la balise telle quelle si
// elle est inconnue, vide si elle est absente ou indéterminée (« und »).
std::string language_name(std::string_view tag);

// « Langue · titre », sans répéter la langue si le titre la contient déjà ; le codec pour un
// sous-titre qui n'est pas de l'ASS ; « Piste N » sans langue ni titre.
std::string track_label(const media::TrackInfo& track);

// La piste de sous-titres suivante dans le menu, puis aucune, puis la première (touche j).
std::optional<int> next_subtitle(const TrackMenu& menu);
// La piste audio suivante, en boucle (touche a) ; rien sans piste audio.
std::optional<int> next_audio(const TrackMenu& menu);

}  // namespace ajp::app::play
