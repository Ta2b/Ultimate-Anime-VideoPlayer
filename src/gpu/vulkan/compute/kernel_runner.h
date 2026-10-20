// Noyaux GLSL de kernels/spec.h enregistrés dans un Recorder : chaque appel vérifie ses vues,
// prépare ses tables, puis ajoute ses dispatchs (barrières comprises). Rien ne s'exécute avant
// la soumission de l'enregistrement, qui peut ensuite être rejoué sans rien recalculer.
// Refus communs : InputMismatch (taille, format ou disposition hors contrat), FieldOutOfRange
// (instant, pic ou noir hors de leur plage), FeatureUnavailable (float16 sans stockage 16 bits),
// ceux du Recorder et du cache de pipelines.
#pragma once

#include <cstdint>
#include <span>

#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/image.h"
#include "gpu/vulkan/compute/kernel_types.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/recording.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/weights.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/resample.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::vk {

class KernelRunner {
 public:
  KernelRunner(const VkContext& context, PipelineCache& pipelines)
      : context_(&context), pipelines_(&pipelines) {}

  // YUV → tenseur RGB 3 plans de mêmes dimensions (ref yuv_to_rgb) : chroma 4:2:0 rééchantillonnée
  // par chroma, depuis le siting de color.
  Done ingest(Recorder& recorder, const YuvView& src, const ColorSpec& color, const TensorView& dst,
              kernels::Filter chroma = kernels::kChainChromaFilter);

  // Tenseur RGB → YUV quantifié du format de dst (ref rgb_to_yuv), chroma réduite vers le siting
  // de color (kOutputSiting pour la chaîne, kRifeSiting pour RIFE).
  Done output_yuv(Recorder& recorder, const TensorView& src, const ColorSpec& color, const YuvView& dst,
                  kernels::Filter chroma = kernels::kChainChromaFilter);

  // Tenseur RGB → image RGBA16 UNORM de mêmes dimensions, alpha opaque (ref to_rgba16).
  Done output_rgba16(Recorder& recorder, const TensorView& src, const StorageImage& dst);

  // Rééchantillonnage séparable d'un tenseur (ref resize) : dimensions paires dans [2, 16384].
  Done resize(Recorder& recorder, const TensorView& src, const TensorView& dst, kernels::Filter filter);
  // Même passe avec les tables de l'appelant (wx : largeurs, wy : hauteurs), par exemple celles de
  // la réduction fusionnée à l'ingestion ; des tables d'autres dimensions sont refusées.
  Done resize(Recorder& recorder, const TensorView& src, const TensorView& dst, const WeightTable& wx,
              const WeightTable& wy);

  // Remplissage mod-64 centré, bords noir studio (ref rife_pad) : padded a la taille remplie
  // de rife_geometry(src) et le format de src.
  Done rife_pad(Recorder& recorder, const YuvView& src, const YuvView& padded);
  // Recadrage de la fenêtre de rife_geometry(dst) dans une sortie remplie (ref rife_crop).
  Done rife_crop(Recorder& recorder, const YuvView& padded, const YuvView& dst);
  Done rife_crop(Recorder& recorder, const TensorView& padded, const TensorView& dst);
  // Plans constants 6 à 10 du tenseur d'entrée de RIFE (11 plans pw × ph), t dans [0, 1].
  Done rife_constants(Recorder& recorder, const TensorView& input, float t);

  // a et b déformés par le mouvement et mélangés à l'instant t de ]0, 1[ (ref warp_blend_at ;
  // t = 1/2 : warp_blend). Tous les tenseurs de même précision.
  // poses.keep : l'instant de chaque pixel réglé d'après poses.previous (ref warp_blend_poses).
  Done warp_blend(Recorder& recorder, const TensorView& a, const TensorView& b, const MotionView& motion,
                  float t, const TensorView& dst, const PoseView& poses = {});

  // Extensions de la tranche verticale (gpu/kernel_ops.h), sur des tenseurs RGB déjà ingérés.
  //
  // Plans 0 à 5 de l'entrée de RIFE : a et b (w × h) dans la fenêtre de rife_geometry(w, h) de
  // input (planes plans : 11 pour RIFE, dont les plans 6 à 10 restent à rife_constants ; 6 pour
  // GMFSS), black autour.
  Done rife_frames(Recorder& recorder, const TensorView& a, const TensorView& b, const TensorView& input,
                   float black, std::uint32_t planes = kernels::kRifeInputChannels);
  // Softmax splatting de GMFSS (ref soft_splat) : source (C plans w × h) poussée par motion (3 plans
  // fx, fy, z de même taille) à l'instant s de [0, 1], additionnée dans sums (C + 1 plans float,
  // remis à zéro à chaque exécution, le dernier pour les poids), puis normalisée dans out (C plans,
  // précision de source). Refus FeatureUnavailable : appareil sans additions atomiques de float.
  Done soft_splat(Recorder& recorder, const TensorView& source, const TensorView& motion, float s,
                  const BufferView& sums, const TensorView& out);
  // Chaque élément de tensor vaut value.
  Done fill(Recorder& recorder, const TensorView& tensor, float value);
  // Écarts de luma BT.709 tirée du RGB par bloc 32 × 32 : sums reçoit la somme de chaque bloc
  // (block_count float), worst[0] le pire bloc en 1/256 de niveau (levels pas entre noir et
  // blanc), remis à zéro à chaque exécution.
  Done rgb_pair_diff(Recorder& recorder, const TensorView& a, const TensorView& b, const BufferView& sums,
                     const BufferView& worst, float levels);
  // Plus grand |déplacement| du mouvement dans sa fenêtre, en bits de float dans result[0].
  Done flow_max(Recorder& recorder, const MotionView& motion, const BufferView& result);

  // Fonction de transfert ou conversion de primaires, pixel par pixel (ref transfer).
  Done transfer(Recorder& recorder, TransferOp op, const TensorView& src, const TensorView& dst,
                const DisplayParams& display = {});

  // Coupe de scène sur les lumas de a et b : une somme partielle float par groupe 16 × 16 dans
  // partials (scene_partial_count valeurs), à additionner par scene_sum après l'exécution.
  Done scene_diff(Recorder& recorder, const YuvView& a, const YuvView& b, const BufferView& partials);
  // Dessins tenus : result[0] reçoit le pire écart, result[1 + by × blocs_x + bx] celui de
  // chaque bloc 32 × 32 (1 + block_count valeurs uint32) ; result[0] est remis à 0 à chaque
  // exécution.
  Done block_diff(Recorder& recorder, const YuvView& a, const YuvView& b, const BufferView& result);

  static std::uint32_t scene_partial_count(std::uint32_t width, std::uint32_t height);
  static std::uint32_t block_count(std::uint32_t width, std::uint32_t height);

 private:
  const VkContext* context_;
  PipelineCache* pipelines_;
};

// Somme en double des sommes partielles de scene_diff. Le score de RIFE la divise par l'aire
// remplie pw × ph (kernels::rife_geometry), puis se compare avec kernels::is_scene_cut.
double scene_sum(std::span<const float> partials);

}  // namespace ajp::gpu::vk
