// L'export vu par l'écran : chaque image du ring rendue par libplacebo à la taille de sortie, dans la
// disposition de l'encodeur (YUV 4:2:0, 10 bits en P010 ou en plans séparés), avec ses meilleurs
// filtres de mise à l'échelle, puis copiée sans attendre dans des tampons que l'encodeur lit tels
// quels. Ni grade, ni HDR, ni sous-titres, ni interface : l'image du ring telle que la chaîne l'a
// faite, au moins aussi fine qu'à l'écran.
//
// Trois chemins. Pour NVENC, des tampons d'appareil exportés vers CUDA : les plans y sont copiés
// sur le GPU, puis d'une copie CUDA dans une image du pool de l'encodeur, et le tampon repart
// aussitôt ; rien ne repasse par l'hôte. (L'encodeur garde chaque image le temps de son
// anticipation, des dizaines : lue en place, la poignée de tampons s'épuisait et l'export se
// figeait.) Pour Vulkan Video sur
// l'appareil de l'écran, le rendu se fait directement dans les plans des images de l'encodeur,
// synchronisé par leur sémaphore (placebo_export_vulkan.cpp). Sinon, des tampons hôte relus par le
// GPU, que l'encodeur lit (FFV1) ou charge sur ses surfaces (VA-API, Vulkan sans partage).
//
// Le slot repart au thread gpu dès le rendu soumis : la copie lit les textures de sortie, pas le
// ring. Les tampons tournent : rendu et copie → prêt → lu par l'encodeur → libre.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

#include "app/play/placebo_screen.h"
#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "core/types.h"
#include "export/export_writer.h"
#include "media/cuda_driver.h"

struct AVBufferRef;
struct AVFrame;

namespace ajp::app::play {

// Ordinal CUDA de l'appareil de rendu, pour l'encodeur qui lit la mémoire CUDA (NVENC) ; -1 sans
// CUDA : GPU d'un autre fabricant, pilote absent, ou appareil introuvable par son UUID.
int export_cuda_ordinal(const GpuDevice& device);

struct FrameExportSetup {
  int width = 0;  // taille encodée, paire
  int height = 0;
  exporter::Layout layout = exporter::Layout::P010;
  ColorInfo color;  // celle des images du ring, gardée telle quelle
  // Le contexte d'images CUDA de l'encodeur (ExportWriter::cuda_frames(), emprunté) : tampons
  // d'appareil exportés vers CUDA, copiés dans ses images. Si l'export ou l'import échoue, tampons
  // hôte comme pour les autres encodeurs : l'encodeur charge lui-même ses images.
  AVBufferRef* cuda_frames = nullptr;
  // Le contexte d'images de l'encodeur Vulkan Video sur l'appareil de l'écran
  // (ExportWriter::vulkan_frames(), emprunté) : les images sont rendues dans ses plans et confiées
  // telles quelles. Si leurs plans ne se rendent pas sur ce GPU, tampons hôte.
  AVBufferRef* vulkan_frames = nullptr;
  // Tampons de conversion (images de l'encodeur Vulkan rendues et pas encore confiées, pour
  // vulkan_frames), de 1 à FrameExport::kMaxBuffers : 11 au profil « débit » (export_depth.h).
  std::size_t buffers = 7;
};

class FrameExport {
 public:
  static constexpr std::size_t kMaxBuffers = 12;

  // Refus : format de texture absent ou sans rendu ni relecture (FeatureUnavailable), mémoire.
  static std::expected<std::unique_ptr<FrameExport>, Refusal> open(PlaceboScreen& screen,
                                                                   const FrameExportSetup& setup);
  // Attend les copies en cours. Les tampons encore lus par l'encodeur doivent être rendus avant :
  // l'écriture (ExportWriter) part la première.
  ~FrameExport();
  FrameExport(const FrameExport&) = delete;
  FrameExport& operator=(const FrameExport&) = delete;

  bool cuda() const { return cuda_.has_value(); }      // les images sont copiées dans celles de l'encodeur
  bool vulkan() const { return vulkan_.has_value(); }  // rendues dans les images de l'encodeur
  bool has_room() const;                               // un tampon libre : convert() peut suivre
  // Rend l'image du ring dans un tampon libre et lance sa copie. Le slot est rendu dans tous les
  // cas : freed, à poster au thread gpu. failure : VendorCallFailed, ou SequenceOrder sans tampon libre.
  struct Converted {
    SlotFree freed;
    std::optional<Refusal> failure;
  };
  Converted convert(const RingFrame& frame);
  // La plus ancienne image copiée, dans l'ordre des convert(), confiée à push ; si push la refuse,
  // elle reste prête. false : rien de prêt, ou refusée. Refus : la copie vers l'image de
  // l'encodeur (CUDA : GpuMemoryExhausted, HwDecodeUnavailable nommant l'appel).
  std::expected<bool, Refusal> deliver(const std::function<bool(exporter::Picture&&)>& push);
  // Aucune image en copie, ni rendue, ni prête.
  bool idle() const {
    return order_.empty() && (!vulkan_ || vulkan_->pending.empty()) && (!cuda_ || cuda_->pending.empty());
  }

 private:
  struct Buffer {
    std::uint8_t* data = nullptr;  // hôte, aligné sur 4 Kio : la relecture peut l'importer sans copie
    pl_buf buf = nullptr;          // ou d'appareil, exporté vers CUDA (data nul)
    media::ImportedMemory memory;  // son adresse CUDA
    int planes_done = 0;           // relectures hôte de plans finies (rappels de libplacebo)
    std::size_t set = 0;           // le jeu de plans relu (planes_)
    bool in_flight = false;        // en copie ou prêt
    std::int64_t pts_us = 0;
    std::atomic<bool> held{false};  // lu par l'encodeur, rendu par Picture::release
  };
  // Une image de l'encodeur, rendue ou copiée, à lui confier avec son instant.
  struct Pending {
    AVFrame* frame = nullptr;
    std::int64_t pts_us = 0;
  };
  struct Cuda {
    std::shared_ptr<media::CudaDriver> driver;
    media::PrimaryContext primary;  // retenu : les tampons se rendent après l'encodeur
    std::uintptr_t stream = 0;      // flux non bloquant des copies vers les images du pool
    AVBufferRef* frames = nullptr;  // référence propre sur le contexte d'images de l'encodeur
    std::deque<Pending> pending;    // copiées, à confier dans l'ordre
  };
  // Les images de l'encodeur Vulkan Video : une image prise au pool par rendu, ses deux plans
  // enveloppés en textures (réutilisées tant que l'image revient du pool), rendue entre une
  // attente et un signal de son sémaphore, puis confiée à l'écriture avec l'image.
  struct VulkanFrames {
    AVBufferRef* frames = nullptr;  // référence propre sur le contexte d'images
    struct Wrap {
      std::uint64_t image = 0;  // VkImage
      std::array<pl_tex, 2> planes{};
    };
    std::vector<Wrap> wraps;
    std::deque<Pending> pending;  // rendues, à confier dans l'ordre
  };
  FrameExport(PlaceboScreen& screen, const FrameExportSetup& setup);
  // Les tampons d'appareil, exportés puis importés dans le contexte primaire du GPU du rendu, et
  // le flux de leurs copies vers les images de frames.
  // Refus : FeatureUnavailable (pas d'export de tampons), ceux du pilote CUDA, GpuMemoryExhausted.
  std::expected<void, Refusal> open_cuda(pl_gpu gpu, AVBufferRef* frames);
  void close_cuda(pl_gpu gpu);
  // Les tampons prêts copiés chacun dans une image du pool de l'encodeur, puis la plus ancienne
  // image copiée confiée.
  std::expected<bool, Refusal> deliver_cuda(Cuda& c, const std::function<bool(exporter::Picture&&)>& push);
  // Une image prise au pool, les plans du tampon copiés dedans, la copie finie.
  std::expected<AVFrame*, Refusal> copy_to_pool(const Cuda& c, const Buffer& buffer);
  // Les images de l'encodeur : une image d'essai prise au pool, ses plans enveloppés et rendables.
  // Refus FeatureUnavailable (format des plans, pas de rendu), VendorCallFailed.
  std::expected<void, Refusal> open_vulkan(pl_gpu gpu, AVBufferRef* frames);
  void close_vulkan(pl_gpu gpu);
  std::expected<std::array<pl_tex, 2>, Refusal> wrap_planes(VulkanFrames& v, pl_gpu gpu,
                                                            const AVFrame& frame);
  Converted convert_vulkan(VulkanFrames& v, const RingFrame& frame);
  bool deliver_vulkan(VulkanFrames& v, const std::function<bool(exporter::Picture&&)>& push);
  // Rend l'image du ring dans ces plans de sortie (un par plan de la disposition ; packed : plans
  // 10 bits dans 16, rx10) et rend le slot.
  Converted render(const RingFrame& frame, std::span<const pl_tex> targets, bool packed);
  bool ready(Buffer& buffer);

  PlaceboScreen& screen_;
  FrameExportSetup setup_;
  pl_renderer renderer_ = nullptr;
  // Deux jeux de plans de sortie, l'un après l'autre : le rendu d'une image n'attend pas la relecture
  // de la précédente (file de transfert à part sur la plupart des GPU).
  static constexpr std::size_t kPlaneSets = 2;
  std::array<std::array<pl_tex, 3>, kPlaneSets> planes_{};
  std::size_t next_set_ = 0;
  int plane_count_ = 0;
  std::array<std::size_t, 3> offsets_{};  // les plans à la suite : chroma après la luma, même pas
  std::array<std::size_t, 3> pitches_{};
  std::array<std::size_t, 3> row_bytes_{};  // octets utiles d'une ligne de chaque plan, et lignes
  std::array<int, 3> rows_{};
  std::size_t bytes_ = 0;
  bool async_ = true;  // relecture hôte par rappel ; sinon bloquante (GPU sans rappels)
  std::optional<Cuda> cuda_;
  std::optional<VulkanFrames> vulkan_;
  std::array<Buffer, kMaxBuffers> storage_;
  std::span<Buffer> buffers_;      // les setup.buffers premiers de storage_
  std::deque<std::size_t> order_;  // tampons en copie ou prêts, dans l'ordre des convert()
};

}  // namespace ajp::app::play
