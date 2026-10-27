// L'épisode suivant ou précédent d'une série : les vidéos du même dossier, dans l'ordre « naturel »
// des noms (« ep2 » avant « ep10 »), comme les rangerait un explorateur de fichiers.
#pragma once

#include <filesystem>
#include <optional>
#include <string_view>

namespace ajp::app::play {

// Compare deux noms en lecture naturelle : les chiffres consécutifs forment un nombre, la casse ne
// compte pas. Négatif, nul ou positif comme strcmp.
int natural_compare(std::string_view a, std::string_view b);

// La vidéo voisine de `current` dans son dossier (direction +1 : la suivante, −1 : la précédente),
// ou rien au bout de la liste ou si le dossier ne se lit pas. N'entrent que les extensions
// vidéo connues (default_player.h).
std::optional<std::filesystem::path> sibling_episode(const std::filesystem::path& current, int direction);

}  // namespace ajp::app::play
