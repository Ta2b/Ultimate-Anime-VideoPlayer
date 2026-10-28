// Vocabulaire des noyaux : tenseurs RGB planaires, images YUV rangées dans un tampon, couleur,
// opérations de transfert. Chaque vue se vérifie avant tout dispatch ; une taille, un format ou
// une disposition hors contrat est refusée (InputMismatch), jamais corrigée.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string_view>

#include "core/types.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/caps.h"
#include "gpu/vulkan/compute/refuse.h"
#include "kernels/spec/color.h"
#include "kernels/spec/resample.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::vk {

// Élément des tenseurs des réseaux : float, ou float16 (variantes « _f16 », stockage 16 bits).
enum class Precision : std::uint8_t { F32, F16 };

constexpr std::uint32_t element_bytes(Precision p) {
  return p == Precision::F16 ? 2U : 4U;
}

// Tenseur NCHW sans lot : planes plans width × height contigus, ligne par ligne.
struct TensorView {
  BufferView buffer;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t planes = 3;
  Precision precision = Precision::F32;
};

constexpr VkDeviceSize tensor_bytes(std::uint32_t width, std::uint32_t height, std::uint32_t planes,
                                    Precision precision) {
  return VkDeviceSize{width} * height * planes * element_bytes(precision);
}

// Un plan YUV dans la vue : début et pas de ligne en octets, multiples de 4 (les noyaux lisent
// et écrivent des mots de 32 bits). Le bourrage d'une ligne écrite vaut 0.
struct PlaneLayout {
  std::uint32_t offset = 0;
  std::uint32_t pitch = 0;
};

// cb porte CbCr entrelacés pour NV12, P010 et P016 ; cr ne sert alors pas.
struct YuvLayout {
  PlaneLayout luma;
  PlaneLayout cb;
  PlaneLayout cr;
  std::uint32_t bytes = 0;  // taille totale de la disposition serrée
};

struct YuvView {
  BufferView buffer;
  kernels::Format format = kernels::Format::Nv12;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  YuvLayout layout;
};

// Les plans d'une image décodée lus en place : la VkImage multiplan de FFmpeg (NV12, P010 ou
// P016) vue plan par plan, luma en views[0], CbCr entrelacés en views[1], chacune liée en image
// échantillonnée (planes.glsl). Les vues restent à l'appelant.
struct YuvPlanes {
  std::array<VkImageView, 2> views{};
  kernels::Format format = kernels::Format::Nv12;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

// Couleur résolue d'une image YUV (effective_matrix, effective_range, effective_siting).
struct ColorSpec {
  kernels::Matrix matrix = kernels::Matrix::Bt709;
  kernels::Range range = kernels::Range::Limited;
  kernels::Siting siting = kernels::Siting::Left;
};

// Numéros de AJP_TRANSFER_OP dans transfer.comp.
enum class TransferOp : std::uint8_t {
  PqEotf,
  PqInverseEotf,
  HlgOetf,
  HlgInverseOetf,
  HlgOotf,
  SrgbEotf,
  SrgbInverseEotf,
  Bt1886Eotf,
  Bt1886InverseEotf,
  Bt709ToBt2020,
  Bt2020ToBt709,
};

// Écran des opérations HLG OOTF et BT.1886, validé par hlg_display et bt1886_display.
struct DisplayParams {
  float hlg_peak = 1000.0F;
  float bt1886_white = 1.0F;
  float bt1886_black = 0.0F;
};

// Mouvement estimé par RIFE : 4 plans (f0x, f0y, f1x, f1y), un masque d'un plan de même taille,
// et la fenêtre de l'image dans ces plans.
struct MotionView {
  TensorView flow;
  TensorView mask;
  kernels::Window area;
};

// Garder les poses (kernels/spec/poses.h) : previous est le flot au milieu de la paire précédente,
// mêmes dimensions et même précision que celui du mouvement ; nul, la paire précédente était tenue.
struct PoseView {
  bool keep = false;
  const TensorView* previous = nullptr;
};

// Disposition serrée : luma, puis Cb (ou CbCr), puis Cr, chaque pas arrondi à 4 octets.
// Refus InputMismatch : dimensions hors contrat (paires dans [2, 16384] en 4:2:0).
std::expected<YuvLayout, Refusal> tight_layout(kernels::Format format, std::uint32_t width,
                                               std::uint32_t height);

// Dimensions du plan de chroma (cw, ch) du format.
std::uint32_t chroma_width(kernels::Format format, std::uint32_t width);
std::uint32_t chroma_height(kernels::Format format, std::uint32_t height);

// Une image YUV conforme : format connu, dimensions du format, chaque plan aligné sur 4 octets,
// pas suffisant, tout dans la vue (moins de 4 Gio, les noyaux indexent en 32 bits).
Done check_yuv(const YuvView& view, std::string_view what);

// Un tenseur conforme : planes plans, dimensions dans [1, 16384], vue assez grande et de moins
// de 4 Gio, float16 seulement si le stockage 16 bits est activé (FeatureUnavailable sinon).
Done check_tensor(const TensorView& tensor, std::uint32_t planes, const Features& enabled,
                  std::string_view what);

// Des plans conformes : format entrelacé (NV12, P010, P016), dimensions du format, deux vues.
Done check_planes(const YuvPlanes& planes, std::string_view what);

// Deux vues aux mêmes dimensions.
Done check_same_size(std::uint32_t w0, std::uint32_t h0, std::uint32_t w1, std::uint32_t h1,
                     std::string_view what);

}  // namespace ajp::gpu::vk
