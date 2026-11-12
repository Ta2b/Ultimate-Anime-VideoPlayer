// La peau des visages pour le grade HDR anime (docs/hdr-anime.md, « Peau des visages »). La pré-analyse des
// plans trouve les visages d'une image sur quelques-unes (media/face_cascade.h), lit la teinte de leur peau,
// et chaque plan garde la palette de ses personnages. Le grade resserre alors sa plage de peau (skin_like,
// 3-anime.hook) sur ces teintes : une lampe chaude, une bougie, un coucher de soleil, de la même famille
// orangée que la peau sans être celle d'un visage du plan, redeviennent des lumières.
//
// La teinte et la saturation se mesurent comme le grade les mesure : sur la lumière linéaire BT.2020, teinte
// de 0 (rouge) à 60° (jaune) quand R ≥ V ≥ B, saturation (max − min) / max. Elles ne dépendent pas de la
// luminance : la palette d'un visage éclairé vaut pour le même visage dans une autre lumière du plan.
#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace ajp::media {

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
// Les doublons (IoU > iou) retirés, le meilleur score gardé ; triés par score décroissant.
std::vector<FaceBox> suppress_faces(std::vector<FaceBox> boxes, float iou = 0.45F);

// Une teinte de peau : teinte en degrés, saturation de 0 à 1.
struct SkinTone {
  float hue = 0.0F;
  float sat = 0.0F;
  friend bool operator==(const SkinTone&, const SkinTone&) = default;
};

// La teinte d'un point en lumière linéaire BT.2020 ; rien hors de l'ordre R ≥ V ≥ B ou au noir.
std::optional<SkinTone> skin_tone_of(float r, float g, float b);
// La plage large de la peau, la même que skin_like du grade : teinte de 10 à 52°, saturation de 0,08 à 0,70,
// douce aux bords. Une palette hors de cette plage n'entre pas dans le plan : la palette ne fait que
// resserrer la plage, jamais l'élargir.
float skin_wide(const SkinTone& tone);

// La palette d'un visage : la médiane par composante des points de sa peau ; vide sans point.
std::optional<SkinTone> face_palette(std::span<const SkinTone> samples);

// Les teintes de peau d'un plan, au plus kMaxPlanSkins.
inline constexpr std::size_t kMaxPlanSkins = 3;
struct PlanSkins {
  std::array<SkinTone, kMaxPlanSkins> tones{};
  std::size_t count = 0;
  std::span<const SkinTone> view() const { return {tones.data(), count}; }
  friend bool operator==(const PlanSkins&, const PlanSkins&) = default;
};

// Les palettes d'un plan, tirées de celles de ses visages : regroupées par teinte voisine (6° et 30 % de
// saturation), un groupe retenu s'il est vu au moins deux fois (un faux visage d'une image ne compte pas),
// les plus vus d'abord ; chaque palette est la médiane de son groupe.
PlanSkins plan_skins(std::span<const SkinTone> faces);

}  // namespace ajp::media
