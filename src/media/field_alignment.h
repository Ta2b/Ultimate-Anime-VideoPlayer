// C15-01 : une piste étiquetée entrelacée dont les champs d'une image sont du même instant (un DVD PAL
// d'anime en 2:2, un H.264 MBAFF progressif marqué entrelacé) n'a rien à désentrelacer : bwdif en
// interpolerait quand même un champ et diviserait la résolution verticale par deux avant SPAN. fieldmatch
// recoud les champs de même instant, bwdif ne traite alors que ce qui reste peigné.
//
// Mais sur du vrai entrelacé (champs de deux instants), fieldmatch avant bwdif fait perdre de la qualité
// (mesuré : 31,8 dB contre 34,8 dB pour bwdif seul sur un motif de test), d'où une mesure avant de choisir.
//
// Un peigne se voit à l'écart entre lignes voisines (champs différents) rapporté à l'écart entre lignes de
// même parité : environ 0,5 à 0,9 quand les deux champs sont du même instant, 1,1 et plus quand un objet a
// bougé entre eux.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ajp::media {

class FieldAlignmentMeter {
 public:
  enum class Verdict : std::uint8_t {
    Unknown,  // trop peu d'images, trop peu de mouvement, ou un rapport entre les deux
    Aligned,  // les champs sont du même instant : fieldmatch les recoud sans perte
    Combed,   // des peignes : du vrai entrelacé, bwdif seul
  };

  // Le plan de luma d'une image : `sample_bytes` octets (1 ou 2, petit-boutiste) par échantillon, `bits` bits
  // utiles (8, 10, 12, 16).
  void push(const std::uint8_t* luma, std::ptrdiff_t stride, int width, int height, int sample_bytes,
            int bits);
  Verdict verdict() const;
  int compared() const { return compared_; }
  int moving() const { return moving_; }
  int combed() const { return combed_; }

 private:
  std::vector<std::uint16_t> previous_;  // lignes de même parité échantillonnées de l'image d'avant
  int compared_ = 0;                     // images comparées à leur précédente
  int moving_ = 0;                       // dont la scène a bougé et dont le rapport est mesurable
  int aligned_ = 0;                      // dont les lignes voisines ne se contredisent pas
  int combed_ = 0;                       // dont les lignes voisines se contredisent
};

}  // namespace ajp::media
