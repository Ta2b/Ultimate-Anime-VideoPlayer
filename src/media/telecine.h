// B15-01 : un flux entrelacé à 29,97 i/s est du télécinéma 3:2 (un anime en 23,976 i/s, deux champs d'une
// image du film mêlés à ceux des voisines) ou du vrai 30i (images de synthèse, génériques, direct). L'inverse
// du télécinéma (fieldmatch, bwdif, decimate) jette une image sur cinq par construction : juste pour le
// premier, des saccades pour le second. Le choix se fait donc sur une mesure, pas sur la cadence.
//
// Dans un 2:3 les images (T1 B1) (T2 B2) (T2 B3) (T3 D…) répètent exactement un champ d'une image à l'autre
// (deux fois par cycle de cinq) pendant que l'autre champ, lui, bouge : un champ identique et son voisin de
// même parité différent. Un vrai 30i n'a pas cet événement : ses deux champs bougent ensemble ou restent
// ensemble.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "media/demuxer.h"
#include "media/field_alignment.h"
#include "media/track.h"

namespace ajp::media {

class TelecineMeter {
 public:
  enum class Verdict : std::uint8_t {
    Unknown,   // trop peu d'images, trop peu de mouvement, ou un rapport entre les deux : le choix d'avant
    Telecine,  // des champs répétés pendant que l'autre bouge
    Native,    // aucun champ répété : de vrais 30i
  };

  // Le plan de luma d'une image entrelacée : `sample_bytes` octets (1 ou 2, petit-boutiste) par échantillon,
  // `bits` bits utiles (8, 10, 12, 16).
  void push(const std::uint8_t* luma, std::ptrdiff_t stride, int width, int height, int sample_bytes,
            int bits);
  Verdict verdict() const;
  int compared() const { return compared_; }
  int moving() const { return moving_; }
  int repeats() const { return repeats_; }

 private:
  std::vector<std::uint16_t> top_;  // échantillons des lignes paires de l'image d'avant
  std::vector<std::uint16_t> bottom_;
  int compared_ = 0;  // images comparées à leur précédente
  int moving_ = 0;    // dont un champ au moins a bougé
  int repeats_ = 0;   // dont un champ est répété et l'autre a bougé
};

// Mesure la piste vidéo de `demuxer` : quelques secondes décodées en logiciel autour du premier quart du
// fichier. Le demuxer est ensuite remis au début. std::nullopt : pas concluant (fichier trop court, décodeur
// qui échoue, image sans mouvement), la lecture garde alors son choix d'avant.
std::optional<bool> measure_telecine(Demuxer& demuxer, const TrackInfo& track);

// La même lecture, avec les deux verdicts d'une piste entrelacée : du télécinéma ou non (comme
// measure_telecine), et des champs du même instant ou non (FieldAlignmentMeter, C15-01). std::nullopt : pas
// concluant.
struct InterlaceMeasure {
  std::optional<bool> telecine;
  std::optional<bool> aligned;
};
InterlaceMeasure measure_interlace(Demuxer& demuxer, const TrackInfo& track);

}  // namespace ajp::media
