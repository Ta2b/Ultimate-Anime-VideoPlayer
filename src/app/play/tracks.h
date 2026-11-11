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

// Une langue que l'interface nomme. code : ISO 639-2/B, la forme de media::canonical_language.
struct Language {
  std::string_view code;
  const char* name;   // clé anglaise de i18n::tr
  bool menu = false;  // proposée aux menus des langues préférées (onglets Son et Sous-titres)
};

// Nom français d'une langue du conteneur (« jpn », « fr », « en-US ») ; la balise telle quelle si
// elle est inconnue, vide si elle est absente ou indéterminée (« und »).
std::string language_name(std::string_view tag);
// Les langues des menus des langues préférées, dans leur ordre.
std::span<const Language> menu_languages();
// Même langue primaire, quelle que soit la forme : « fra », « fr-CA » et « FRE » se valent.
bool same_language(std::string_view a, std::string_view b);

// « Langue · titre », sans répéter la langue si le titre la contient déjà ; le codec pour un
// sous-titre qui n'est pas de l'ASS ; « Piste N » sans langue ni titre.
std::string track_label(const media::TrackInfo& track);

// La piste de sous-titres suivante dans le menu, puis aucune, puis la première (touche j).
std::optional<int> next_subtitle(const TrackMenu& menu);
// La piste audio suivante, en boucle (touche a) ; rien sans piste audio.
std::optional<int> next_audio(const TrackMenu& menu);

// Une piste choisie à la main, retenue par sa langue et son titre : l'épisode suivant n'a pas les mêmes
// numéros de flux.
struct TrackLike {
  std::string language;  // balise du conteneur ; vide : la piste n'en dit rien
  std::string title;
};

// Les pistes choisies à la main pendant la séance (B29-11) : elles valent pour les vidéos suivantes tant
// qu'une piste de la même langue existe, avant les préréglages de langue. subtitles : rien = aucun choix ;
// nullopt dedans = coupés (touche j ou menu), ce qui vaut pour toute vidéo.
struct ManualTracks {
  std::optional<TrackLike> audio;
  std::optional<std::optional<TrackLike>> subtitles;
};

// Le choix retenu d'une piste ; rien si elle ne dit pas sa langue (rien à reconnaître ailleurs).
std::optional<TrackLike> track_like(const media::TrackInfo& track);
// Dans tracks, la piste de ce type qui ressemble à wanted : même langue et même titre, sinon fallback s'il
// est de la même langue, sinon la première de la même langue ; rien si aucune n'a cette langue.
std::optional<int> matching_track(std::span<const media::TrackInfo> tracks, media::TrackKind kind,
                                  const TrackLike& wanted, std::optional<int> fallback);

}  // namespace ajp::app::play
