// Image décodée et description de couleur : ce que media remet à gpu.
// Fait partie de core_types : aucune dépendance hors bibliothèque standard.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>

#include "core/types.h"

namespace ajp {

// Horodatage inconnu, partout où un pts tient dans un std::int64_t (DecodedFrame, ring,
// paquets). media traduit AV_NOPTS_VALUE de FFmpeg en kNoPts ; un module qui préfère
// std::optional (core/cadence) passe par known_pts et pts_or_unknown, les deux seules
// conversions. Un calcul sur un pts doit d'abord écarter kNoPts.
inline constexpr std::int64_t kNoPts = std::numeric_limits<std::int64_t>::min();

constexpr std::optional<std::int64_t> known_pts(std::int64_t pts_us) {
  if (pts_us == kNoPts) return std::nullopt;
  return pts_us;
}

constexpr std::int64_t pts_or_unknown(std::optional<std::int64_t> pts_us) {
  return pts_us.value_or(kNoPts);
}

enum class PixelFormat : std::uint8_t {
  Nv12,         // 8 bits, chroma entrelacée
  P010,         // 10 bits dans 16, chroma entrelacée
  P016,         // 16 bits, chroma entrelacée
  Yuv420p,      // 8 bits planaire
  Yuv420p10,    // 10 bits dans 16, planaire
  Yuv444p16,    // repli prouvé du ring
  Rgba16Unorm,  // format par défaut du ring
  Rgba8Unorm,
};

enum class ColorMatrix : std::uint8_t { Bt601, Bt709, Bt2020Ncl, Rgb };
enum class ColorRange : std::uint8_t { Limited, Full };
enum class Transfer : std::uint8_t { Bt1886, Srgb, Linear, Pq, Hlg };
enum class Primaries : std::uint8_t { Bt709, Bt601_625, Bt601_525, Bt2020, DciP3 };
enum class ChromaSiting : std::uint8_t { Left, Center, TopLeft };

// Métadonnées HDR10 : affichage de mastering (SMPTE ST 2086) et niveaux de lumière.
struct HdrMetadata {
  std::array<float, 6> primaries_xy{};  // R, G, B en (x, y)
  std::array<float, 2> white_xy{};
  float max_luminance = 0.0F;  // nits
  float min_luminance = 0.0F;  // nits
  float max_cll = 0.0F;        // nits, 0 si inconnu
  float max_fall = 0.0F;       // nits, 0 si inconnu
};

struct ColorInfo {
  ColorMatrix matrix = ColorMatrix::Bt709;
  ColorRange range = ColorRange::Limited;
  Transfer transfer = Transfer::Bt1886;
  Primaries primaries = Primaries::Bt709;
  ChromaSiting siting = ChromaSiting::Left;
  std::optional<HdrMetadata> hdr;
};

// Où vivent les pixels d'une image.
enum class FrameMemory : std::uint8_t { Host, Cuda, Vulkan, D3d11, VaApi, VideoToolbox };

// Propriétaire des pixels. media l'implémente autour d'une référence AVFrame ;
// la dernière copie de DecodedFrame relâchée rend le tampon à son pool.
class FrameData {
 public:
  virtual ~FrameData() = default;
  virtual FrameMemory memory() const = 0;
  virtual int plane_count() const = 0;
  // Mémoire hôte seulement : octets et pas de ligne du plan i.
  virtual std::span<const std::byte> host_plane(int plane) const = 0;
  virtual std::size_t host_stride(int plane) const = 0;
  // Mémoire GPU : pointeur natif du plan i (CUdeviceptr, VkImage, ID3D11Texture2D…).
  virtual std::uintptr_t native_plane(int plane) const = 0;
  // Mémoire GPU linéaire (Cuda) : pas de ligne du plan i, en octets. 0 pour une mémoire
  // opaque, dont le pilote seul connaît la disposition (VkImage, texture D3D11, surface
  // VA-API non exportée, CVPixelBuffer), et pour l'hôte, qui a host_stride.
  virtual std::size_t native_pitch(int plane) const = 0;
  // Ce qu'il faut attendre côté GPU avant de lire, 0 si rien : CUevent du flux de décodage
  // (Cuda), ou VkSemaphore timeline de l'AVVkFrame (Vulkan).
  virtual std::uintptr_t ready_event() const = 0;
  // Vulkan : la valeur du sémaphore timeline de ready_event qui marque la fin du décodage,
  // lue à la livraison. 0 pour un CUevent ou sans attente. Le lecteur garde l'image tant que
  // sa propre lecture n'est pas achevée côté GPU : FFmpeg peut la réécrire dès qu'elle revient
  // au pool.
  virtual std::uint64_t ready_value() const = 0;
};

struct DecodedFrame {
  SourceId id;  // identité explicite : le pool recycle les adresses
  int width = 0;
  int height = 0;
  PixelFormat format = PixelFormat::Nv12;
  ColorInfo color;
  std::int64_t pts_us = kNoPts;  // horodatage de présentation, microsecondes ; kNoPts si inconnu
  std::int64_t duration_us = 0;  // durée nominale, microsecondes
  std::uint32_t epoch = 0;       // incrémenté à chaque Flush ; une image d'une époque passée est jetée
  std::shared_ptr<const FrameData> data;
};

// Ce que media appelle pour livrer une image ; gpu l'implémente par sa boîte.
// false : la borne est atteinte, le producteur attend un Credit.
class FrameSink {
 public:
  virtual ~FrameSink() = default;
  virtual bool try_push(DecodedFrame&& frame) = 0;
};

}  // namespace ajp
