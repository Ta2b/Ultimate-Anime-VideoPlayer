// Le verdict de l'essai du RIFE natif gardé sur le disque (docs/rife-natif.md) : l'essai complet coûte
// environ 4,5 s au chargement (quatre réseaux réglés, chaque morceau chronométré et jugé, le mélange
// vérifié). Son verdict ne dépend que de la carte, du pilote, des moteurs TensorRT de la chaîne, de la
// taille, de la variante et des noyaux natifs eux-mêmes : la même identité rend le même verdict. Le fichier,
// un par identité dans le dossier des moteurs, garde l'option de chaque morceau, la fin fondue et la justesse
// de la chaîne contre TensorRT ; au chargement suivant, seuls les réseaux du mélange sont réglés, et la
// chaîne est rejugée une fois contre TensorRT avant d'être reprise. En C++ pur : testé sans CUDA
// (tests/unit/gpu/rife_native_verdict_test.cpp).
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kirari::gpu::cuda::rife_verdict {

// Ce que l'essai a retenu : l'option de chaque morceau (0 TensorRT, 1 + n le réseau natif n de l'échelle
// des précisions), la fin dans la dernière couche native, et la chaîne retenue contre TensorRT, en dB (0
// quand TensorRT garde tout).
struct Verdict {
  std::vector<int> options;
  bool fused_end = true;
  double trt_db = 0.0;
};

// L'option la plus haute : les quatre crans de l'échelle des précisions (rife_native_mix.cpp).
inline constexpr int kOptionsMax = 4;

// TensorRT garde-t-il toute la chaîne ?
bool all_trt(const Verdict& verdict);

// Le texte du fichier, et sa lecture : rien sur un texte d'un autre format, tronqué, dont le nombre de
// morceaux n'est pas pieces ou dont une option sort de 0 à options_max.
std::string text_of(const Verdict& verdict);
std::optional<Verdict> parse(std::string_view text, std::size_t pieces, int options_max);

// Le nom du fichier d'une identité (les champs joints par le backend) : « rife-natif-<16 chiffres
// hexadécimaux>.verdict », une empreinte FNV-1a 64 bits de l'identité et du format.
std::string file_name(std::string_view identity);

// La chaîne reprise du verdict tient-elle encore : sa justesse contre TensorRT, mesurée au chargement, à
// moins de kTolerance dB sous celle du verdict (les réglages d'un chargement à l'autre peuvent couper K
// autrement, et l'ordre des sommes avec).
inline constexpr double kTolerance = 1.0;
bool still_holds(const Verdict& verdict, double measured_db);

}  // namespace kirari::gpu::cuda::rife_verdict
