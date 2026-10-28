// La langue d'une piste quand le conteneur ne la donne pas : beaucoup de fichiers d'anime laissent
// la balise vide (ou « und ») et nomment la langue dans le titre de la piste (« VOSTFR »,
// « Japanese », « 日本語 »).
#pragma once

#include <string>
#include <string_view>

#include "media/track.h"

namespace ajp::media {

// La langue que nomme un titre de piste, en ISO 639-2/B (« fre », « jpn ») ; vide si aucune.
// Plusieurs langues nommées : la première du titre (« Japanese [English subs] » : « jpn »).
// Les mots de deux lettres ambigus (« en », « de », « it ») ne comptent pas.
std::string_view language_from_title(std::string_view title);

// La balise de la piste ; vide, « und », « mul » ou « zxx » : la langue de son titre.
std::string track_language(const TrackInfo& track);

}  // namespace ajp::media
