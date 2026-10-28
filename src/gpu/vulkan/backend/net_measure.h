// Mesures communes aux moteurs de réseaux : médiane d'échantillons, écart entre deux sorties, et le
// choix de l'unité de calcul d'un runtime de fabricant (OpenVINO : GPU ou NPU ; Core ML : GPU,
// Neural Engine ou les deux). Chaque unité candidate exécute le réseau sur la même image de test
// (probe_image) ; le temps qui compte est la médiane de kUnitTimedRuns inférences après chauffe, et
// sa première sortie est comparée à celle de l'unité de référence. L'accélérateur préféré (NPU,
// Neural Engine), qui laisse le GPU à la chaîne, à l'interpolation et au rendu, garde le réseau
// s'il est juste et pas plus lent que kUnitTolerance fois la plus rapide ; sinon la plus rapide
// des unités justes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ajp::gpu::vk::net {

// Inférences de chauffe (compilation paresseuse, caches, montée en fréquence), puis inférences
// chronométrées : la médiane de vingt écarte un lancement retardé par l'hôte, qui sur cinq pouvait
// à lui seul faire passer un accélérateur au-dessus de la tolérance.
inline constexpr int kUnitWarmupRuns = 3;
inline constexpr std::size_t kUnitTimedRuns = 20;
// L'accélérateur préféré garde le réseau jusqu'à ce facteur du temps de l'unité la plus rapide.
inline constexpr double kUnitTolerance = 1.25;
// Écart moyen permis entre la sortie d'une unité et celle de la référence (images entre 0 et 1) :
// au-delà, le compilateur de l'unité a perdu en précision.
inline constexpr double kUnitMaxMeanError = 0.01;

// Une image de test entre 0 et 1 aux dimensions dims (NCHW : les deux dernières sont la hauteur et
// la largeur, les précédentes des plans) : un dégradé diagonal, des aplats en damier de 32 pixels
// aux bords francs et un trait sombre d'un pixel toutes les 16 lignes et colonnes, de quoi faire
// travailler un réseau d'image comme sur un dessin animé plutôt que sur du bruit. Chaque plan est
// décalé en valeur ; shift décale l'image de 2 pixels par unité vers la droite (les deux images
// d'un réseau d'interpolation : un mouvement). Moins de deux dimensions (un instant, un facteur) :
// 0,5 partout ; une dimension nulle ou négative : rien.
std::vector<float> probe_image(std::span<const std::int64_t> dims, int shift = 0);

// Médiane (moyenne des deux du milieu en nombre pair) ; 0 sans échantillon.
double median(std::span<const double> samples);

// Écart absolu moyen ; tailles différentes : l'infini ; deux sorties vides : 0 (rien à comparer).
double mean_error(std::span<const float> a, std::span<const float> b);

// Une unité mesurée.
struct UnitMeasure {
  double ms = 0.0;     // médiane d'une inférence
  double error = 0.0;  // écart moyen de sa sortie à celle de la référence (0 pour la référence)
};

// L'index de l'unité retenue dans measured (nullopt : unité refusée ou en échec) : preferred si
// elle est juste (error ≤ max_error) et à moins de tolerance fois la plus rapide des unités
// justes, sinon la plus rapide d'entre elles ; nullopt si aucune n'est juste.
std::optional<std::size_t> choose_unit(std::span<const std::optional<UnitMeasure>> measured,
                                       std::size_t preferred, double tolerance = kUnitTolerance,
                                       double max_error = kUnitMaxMeanError);

}  // namespace ajp::gpu::vk::net
