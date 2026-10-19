// Noyaux de la chaîne exécutés par un backend sur ses propres tenseurs : le contrat de
// kernels/spec.h, que la référence CPU de kernels/ref/ fixe pour chaque port.
//
// Chaque opération s'ajoute à la file du backend, dans l'ordre des appels et des enqueue() de
// moteur ; aucune n'attend côté hôte. Les tenseurs d'image sont en NCHW, lot 1 : RGB planaire
// 1 × 3 × h × w, en F16 ou F32 ; un tenseur de sortie décrit la mémoire écrite. Un format ou une taille hors
// contrat rend un refus nommé (FormatUnsupported, ShapeInvalid, FieldOutOfRange) avant toute soumission,
// jamais une conversion silencieuse.
//
// Une opération qui lit une DecodedFrame (ingest, rife_pad, measure) en garde une copie de
// frame.data jusqu'à ce que sa lecture soit achevée côté GPU, constaté sans attente hôte à un
// appel suivant, et toutes à la destruction du backend : l'appelant peut relâcher la sienne
// au retour, l'image ne revient au pool du décodeur qu'après la lecture. Une image Vulkan se
// lit entre begin_vulkan_access et end_vulkan_access (core/frame.h).
#pragma once

#include <cstdint>
#include <expected>
#include <optional>

#include "core/frame.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/tensor.h"
#include "kernels/spec.h"

namespace ajp::gpu {

// Ingestion d'une image décodée : quelle conversion, selon l'étage qui la lit.
struct IngestSpec {
  // Chaîne d'agrandissement : kChainChromaFilter, matrice, plage et siting de la source.
  // RIFE : kRifeChromaFilter et kRifeMatrix, siting gauche (rife_cuda.py), quelle que soit
  // l'étiquette de la source.
  kernels::Filter chroma = kernels::kChainChromaFilter;
  bool rife_conversion = false;
};

// Mouvement estimé au milieu d'une paire, sorti du moteur RIFE (kernels/spec/warp.h).
struct Motion {
  Tensor flow;           // 1 × 4 × fh × fw : f0x, f0y, f1x, f1y, en pixels de la fenêtre
  Tensor mask;           // 1 × 1 × fh × fw : poids de a, dans [0, 1]
  kernels::Window area;  // la fenêtre de l'image dans les plans remplis
};

// Synchronisation d'une écriture dans le ring (protocole de core/ring.h).
struct RingWrite {
  std::uint64_t wait_value = 0;    // SlotFree::semaphore_value ; 0 au premier tour
  std::uint64_t signal_value = 0;  // RingFrame::ready_value, impaire, > wait_value
};

// Mesures d'une paire de sources (kernels/spec/detect.h), lues sur la luma non remplie.
struct PairScores {
  SourceId left;
  SourceId right;
  double scene_score = 0.0;       // kernels::is_scene_cut(scene_score)
  std::uint32_t worst_block = 0;  // pire bloc 32 × 32, en 1/256 de niveau 8 bits
  bool identical = false;         // luma égale au bit : somme des écarts nulle
};

class KernelOps {
 public:
  virtual ~KernelOps() = default;

  // YUV décodé (formats et ColorInfo de core/frame.h) → RGB planaire de même taille, attendu
  // derrière l'événement ready_event / ready_value de l'image (kernels::ref::yuv_to_rgb).
  virtual std::expected<void, Refusal> ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                              const Tensor& rgb) = 0;

  // Rééchantillonnage séparable à la taille de out (kernels::ref::resize). Area réduit
  // seulement : un agrandissement par Area est refusé.
  virtual std::expected<void, Refusal> resample(const Tensor& in, kernels::Filter filter,
                                                const Tensor& out) = 0;

  // Entrée de RIFE : remplissage mod-64 centré en noir studio, en YUV, puis la conversion de
  // RIFE (ref::rife_pad puis ref::rife_prepare). out mesure rife_geometry(w, h) rempli.
  virtual std::expected<void, Refusal> rife_pad(const DecodedFrame& frame, const Tensor& padded) = 0;

  // Sortie de RIFE : la fenêtre de rife_geometry(w, h), w × h étant la taille de out
  // (ref::rife_crop sur RGB).
  virtual std::expected<void, Refusal> rife_crop(const Tensor& padded, const Tensor& out) = 0;

  // Warp arrière et mélange de a et b à l'instant t, 0 < t < 1 (ref::warp_blend_at) ;
  // a, b et out ont la taille de sortie, le mouvement celle de la fenêtre de RIFE.
  virtual std::expected<void, Refusal> warp_blend(const Tensor& a, const Tensor& b, const Motion& motion,
                                                  float t, const Tensor& out) = 0;

  // RGB → RGBA16 UNORM du slot (ref::to_rgba16), rgb ayant la taille du slot. Le slot est
  // importé au premier appel de sa génération ; une autre taille ou un autre format est refusé.
  // Le premier appel d'une génération plus récente relâche les imports des précédentes, une
  // fois leurs écritures achevées côté GPU : un import gardé retiendrait la mémoire du ring.
  virtual std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                                  const RingWrite& sync) = 0;

  // Lance la mesure de coupe de scène et de pire bloc entre deux sources de même format et de
  // même taille. Le résultat est relu sans attente, comme k_flow_max : il sert à la paire
  // suivante, par previous_scores().
  virtual std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) = 0;

  // La dernière mesure terminée, une seule fois ; rien si aucune n'a fini depuis l'appel
  // précédent. N'attend jamais côté hôte.
  virtual std::optional<PairScores> previous_scores() = 0;
};

}  // namespace ajp::gpu
