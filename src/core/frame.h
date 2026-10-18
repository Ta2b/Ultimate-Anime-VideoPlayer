// Image décodée et description de couleur : ce que media remet à gpu.
// Fait partie de core_types : aucune dépendance hors bibliothèque standard.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

#include "core/types.h"

namespace ajp {

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
  // Événement à attendre côté GPU avant de lire (CUevent, VkSemaphore), 0 si aucun.
  virtual std::uintptr_t ready_event() const = 0;
};

struct DecodedFrame {
  SourceId id;  // identité explicite : le pool recycle les adresses
  int width = 0;
  int height = 0;
  PixelFormat format = PixelFormat::Nv12;
  ColorInfo color;
  std::int64_t pts_us = 0;       // horodatage de présentation, microsecondes
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
