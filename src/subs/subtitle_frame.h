// Ce que subs remet à render : pour un pts de sortie, un atlas R8 de couverture
// et la liste des rectangles à y prélever. render les dessine après le grade, à
// la résolution de sortie, au blanc subtitle_white.
//
// Contrat de dessin, dans l'ordre de la liste (l'ordre de libass) :
//   pour chaque pixel p de rect.src, couverture c = atlas[p] / 255,
//   opacité o = c × rect.color.a / 255,
//   sortie = sortie × (1 − o) + rect.color.rgb × o, en (dst_x, dst_y) + (p − src.origine).
// Les couleurs sont des valeurs sRGB 8 bits, alpha droit (non prémultiplié).
//
// Téléversement : render garde l'atlas de sa dernière image (son shared_ptr) et
// ne re-téléverse que si le pointeur ou la génération change. Un contenu
// inchangé, même déplacé, revient avec le même atlas et de nouveaux rectangles.
//
// Recyclage : un atlas est partagé en lecture seule (shared_ptr). Quand la
// dernière référence tombe, il rentre dans la boîte du thread subs, qui le
// réutilise. Au plus kMaxAtlasesInFlight atlas existent ; subs garde le dernier
// pour les images inchangées, donc render ne doit pas en tenir plus de
// kMaxAtlasesInFlight − 1 à la fois, sous peine d'attendre indéfiniment.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/types.h"

namespace ajp::subs {

// Atlas vivants au plus, chez subs et chez render réunis (estimation du plan).
inline constexpr std::size_t kMaxAtlasesInFlight = 8;
// Côté maximal d'un atlas, en pixels : la taille 2D garantie par Vulkan.
inline constexpr int kMaxAtlasSide = 4096;
// Rectangles au plus dans une image de sous-titres.
inline constexpr std::size_t kMaxRectsPerFrame = 16384;
// Demandes de rendu sans réponse au plus, par demandeur : c'est aussi la borne
// de la boîte de subs pour ces demandes, et le minimum de celle de render.
inline constexpr std::size_t kMaxPendingRequests = 8;

struct Rgba8 {
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
  std::uint8_t a = 0;  // opacité : 255 opaque
  friend bool operator==(const Rgba8&, const Rgba8&) = default;
};

struct AtlasBox {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  friend bool operator==(const AtlasBox&, const AtlasBox&) = default;
};

// Un morceau de sous-titre : la couverture src de l'atlas, posée en (dst_x, dst_y).
struct SubtitleRect {
  AtlasBox src;   // dans l'atlas
  int dst_x = 0;  // dans l'image de sortie, en pixels ; même taille que src
  int dst_y = 0;
  Rgba8 color;
  friend bool operator==(const SubtitleRect&, const SubtitleRect&) = default;
};

// Couverture R8 empaquetée, une ligne de width octets après l'autre.
struct SubtitleAtlas {
  std::uint64_t generation = 0;  // change à chaque contenu neuf : render ne re-téléverse que si elle change
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;  // width × height ; hors des rectangles, tout est à 0
};

struct SubtitleFrame {
  std::int64_t pts_us = 0;  // horodatage de l'image de sortie demandée
  std::uint32_t epoch = 0;  // celle de la demande ; render jette une époque passée
  int width = 0;            // résolution de sortie à laquelle les rectangles sont placés
  int height = 0;
  std::shared_ptr<const SubtitleAtlas> atlas;  // nullptr : rien à dessiner à ce pts
  std::vector<SubtitleRect> rects;
};

// Ce que subs appelle pour livrer, depuis son thread ; render l'implémente par sa boîte.
class SubtitleSink {
 public:
  virtual ~SubtitleSink() = default;
  // Exactement une réponse par demande de rendu, dans l'ordre des demandes.
  // La borne de render doit couvrir kMaxPendingRequests réponses : false est
  // une violation du contrat, la réponse est perdue et journalisée.
  virtual bool try_push(SubtitleFrame&& frame) = 0;
  // Refus sans réponse propre : piste illisible, paquet hors plage, police
  // refusée, ou cause d'une réponse vide. À afficher tel quel.
  virtual void refused(const Refusal& refusal) = 0;
};

}  // namespace ajp::subs
