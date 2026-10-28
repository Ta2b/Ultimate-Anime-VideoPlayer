// Correspondance entre la balise de langue d'une piste et une liste de priorités.
#pragma once

#include <span>
#include <string>
#include <string_view>

namespace ajp::media {

// Score de la balise `tag` (« jpn », « ja », « en-US ») contre les priorités,
// de la plus voulue à la moins voulue. 0 : aucune correspondance. Plus le score
// est grand, meilleure est la correspondance : chaque rang dans la liste coûte
// 1, chaque sous-balise de la piste différente ou non demandée coûte 1000.
// ISO 639-1, 639-2/B et 639-2/T se valent (« fr », « fre », « fra »).
int language_score(std::span<const std::string> priorities, std::string_view tag);

// Forme canonique ISO 639-2/B d'une langue primaire (« ja » → « jpn »,
// « fra » → « fre ») ; une balise inconnue revient telle quelle.
std::string_view canonical_language(std::string_view primary);

}  // namespace ajp::media
