// Méga plan A4 : les visages pour la méthode « HDR anime ». La partie sans réseau, testée ici : le décodage
// d'une sortie de détecteur (YOLO : cx, cy, w, h, score par ligne, en pixels de l'entrée), la suppression
// des doublons, le suivi d'un visage d'une image à l'autre, la palette de peau d'un visage et son exclusion
// de la carte des émetteurs (une peau claire n'est pas une lumière). Le détecteur lui-même (un ONNX dans
// l'étage TensorRT) attend un modèle : aucun n'est livré ni téléchargé, rien ne tourne sans lui.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ajp::render {

// Un cadre en coordonnées normalisées de l'image (0 à 1).
struct FaceBox {
  float x0 = 0.0F;
  float y0 = 0.0F;
  float x1 = 0.0F;
  float y1 = 0.0F;
  float score = 0.0F;
  friend bool operator==(const FaceBox&, const FaceBox&) = default;
};

float face_iou(const FaceBox& a, const FaceBox& b);

// La sortie du détecteur : `rows` lignes de `stride` ≥ 5 flottants (cx, cy, w, h, score, …) en pixels d'une
// entrée input_w × input_h ; gardées au-dessus de min_score, rognées à l'image.
std::vector<FaceBox> decode_faces(std::span<const float> output, int rows, int stride, int input_w,
                                  int input_h, float min_score = 0.4F);
// Les doublons (IoU > iou) retirés, le meilleur score gardé ; triés par score décroissant.
std::vector<FaceBox> suppress_faces(std::vector<FaceBox> boxes, float iou = 0.45F);

// La peau d'un visage, en ICtCp (I : intensité PQ ; Ct, Cp : chroma).
struct SkinPalette {
  float i = 0.0F;
  float ct = 0.0F;
  float cp = 0.0F;
  friend bool operator==(const SkinPalette&, const SkinPalette&) = default;
};
// La médiane par composante des échantillons pris au centre du visage (ni cheveux ni yeux : le tiers
// central) ; vide sans échantillon.
std::optional<SkinPalette> face_palette(std::span<const SkinPalette> samples);

struct FaceTrack {
  std::uint32_t id = 0;
  FaceBox box;
  std::optional<SkinPalette> palette;
  int hits = 0;    // détections retrouvées
  int misses = 0;  // images de suite sans détection
  bool confirmed() const { return hits >= 2; }
};

// Le suivi : chaque détection reprend la piste la plus recouvrante (IoU ≥ 0,3), son cadre lissé ; une piste
// perdue trois images de suite s'efface ; une coupe efface tout.
class FaceTracker {
 public:
  static constexpr int kMaxMisses = 3;
  static constexpr float kMatchIou = 0.3F;
  static constexpr float kSmooth = 0.5F;  // part de la nouvelle détection dans le cadre suivi

  void update(std::span<const FaceBox> detections);
  void set_palette(std::uint32_t id, const SkinPalette& palette);
  void cut();
  const std::vector<FaceTrack>& tracks() const { return tracks_; }

 private:
  std::vector<FaceTrack> tracks_;
  std::uint32_t next_id_ = 1;
};

// La part d'un point (normalisé) qui appartient à un visage confirmé : 1 dans le cadre, 0 au-delà d'une marge
// de 10 % de sa taille, une rampe entre deux. La carte des émetteurs est multipliée par 1 − cette part.
float face_exclusion(std::span<const FaceTrack> tracks, float x, float y);

}  // namespace ajp::render
