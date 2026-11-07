// L'épisode suivant ou précédent d'une série : les vidéos du même dossier, dans l'ordre « naturel »
// des noms (« ep2 » avant « ep10 »), comme les rangerait un explorateur de fichiers.
#pragma once

#include <chrono>
#include <cstdint>
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

// P3.27 : un fichier encore en téléchargement. Sa fin lue n'est pas la fin de la vidéo : s'il a grandi depuis
// l'ouverture, ou grandit pendant une courte attente (écrit dans les 10 dernières secondes), la lecture le
// rouvre au même endroit, durée relue, au lieu d'enchaîner l'épisode suivant.
struct GrowingFile {
  std::uintmax_t size = 0;
  static GrowingFile of(const std::filesystem::path& file);
  bool grew(const std::filesystem::path& file,
            std::chrono::milliseconds wait = std::chrono::milliseconds(3000)) const;
};

}  // namespace ajp::app::play
