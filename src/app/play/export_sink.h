// L'export par puits CUDA (étude 1, 5.3) : sous TensorRT, quand l'encodeur lit la mémoire CUDA
// (NVENC) et que le fichier a la taille de la sortie de la chaîne, l'étage gpu écrit chaque image en
// NV12 ou P010 tramé (KernelOps::write_yuv) directement dans une image du pool de l'encodeur : ni
// ring Vulkan, ni rendu libplacebo, ni copie, et l'export ne dépend plus de l'écran.
//
// Chaque slot de l'étage tient une image de l'encodeur. Le protocole des slots reste celui du ring
// (core/ring.h) : l'étage écrit le slot puis poste son RingFrame ; main attend que le GPU ait fini
// (ready, signalé par le backend), prend l'image du slot pour l'écriture et en remet une neuve à
// sa place (take), puis rend le slot. Le slot n'attend donc jamais l'encodeur : l'image confiée
// lui appartient, et revient au pool quand il la relâche.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "app/play/export_job.h"
#include "app/play/links.h"
#include "app/play/upscale_stage.h"
#include "core/frame.h"
#include "core/ring.h"
#include "core/types.h"
#include "export/encoder_choice.h"
#include "kernels/spec/color.h"

struct AVBufferRef;
struct AVFrame;

namespace ajp::app::play {

// Une image neuve de l'encodeur, ou le refus du pool.
using FrameSource = std::function<std::expected<AVFrame*, Refusal>()>;

// Les images du contexte d'images CUDA de l'encodeur (ExportWriter::cuda_frames(), emprunté).
FrameSource pool_frames(AVBufferRef* frames);

struct ExportSinkSetup {
  FrameSource frames;
  int width = 0;  // la taille encodée, celle de la sortie de la chaîne
  int height = 0;
  kernels::Format format = kernels::Format::P010;  // Nv12 ou P010 : la disposition de l'encodeur
  kernels::Matrix matrix = kernels::Matrix::Bt709;
  kernels::Range range = kernels::Range::Limited;
  kernels::Siting siting = kernels::kOutputSiting;
  std::uint32_t slots = 4;
  std::uint32_t generation = 1;
};

// Ce qui empêche le puits pour cet export, ou rien : backend sans write_yuv, encodeur qui ne lit
// pas la mémoire CUDA, disposition autre que P010 ou NV12, flux étiqueté RGB, ou taille encodée
// autre que la sortie de la chaîne (l'ancien chemin la met à l'échelle).
std::optional<std::string> sink_blocker(bool yuv_targets, bool cuda_frames, exporter::Layout layout,
                                        ColorMatrix matrix, ExportSize size, ExportSize chain);

// Le réglage du puits pour un flux étiqueté color, à la disposition layout (P010 ou NV12).
ExportSinkSetup sink_setup(FrameSource frames, ExportSize size, exporter::Layout layout,
                           const ColorInfo& color, std::uint32_t slots);

class ExportSink final : public YuvSlots {
 public:
  // Les écritures en vol attendues au plus ce temps à la destruction ; au-delà, leurs images et
  // leurs signaux ne sont pas rendus (le GPU pourrait encore y écrire).
  static constexpr std::int64_t kDrainTimeoutUs = 2'000'000;

  // Une image prise par slot. Refus : FieldOutOfRange (1 à kMaxRingSlots slots), ShapeInvalid
  // (taille impaire ou plans trop étroits), FormatUnsupported (ni NV12 ni P010), ceux de frames.
  static std::expected<std::shared_ptr<ExportSink>, Refusal> open(ExportSinkSetup setup);
  ~ExportSink() override;
  ExportSink(const ExportSink&) = delete;
  ExportSink& operator=(const ExportSink&) = delete;

  // Les slots de l'étage : indice, génération et taille, ni image ni sémaphore.
  const std::vector<RingSlotDesc>& slots() const { return descs_; }
  // Thread gpu : l'image du slot et son signal.
  gpu::YuvTarget target(std::uint32_t slot, std::uint64_t done_value) const override;

  // Thread main. L'écriture de frame finie côté GPU.
  bool ready(const RingFrame& frame) const;
  // L'image finie du slot, à confier à l'écriture (Picture::frame, rendue par elle), une neuve
  // prise à sa place : le slot peut repartir à l'étage. Refus : ceux de frames, le slot gardant
  // son image.
  std::expected<AVFrame*, Refusal> take(std::uint32_t slot);

 private:
  struct Fences {
    std::array<std::atomic<std::uint64_t>, kMaxRingSlots> done{};
  };
  explicit ExportSink(ExportSinkSetup setup);
  std::expected<void, Refusal> check(const AVFrame& frame) const;

  ExportSinkSetup setup_;
  std::vector<RingSlotDesc> descs_;
  std::unique_ptr<Fences> fences_ = std::make_unique<Fences>();
  mutable std::mutex mutex_;
  std::vector<AVFrame*> frames_;               // sous mutex_ : l'image de chaque slot
  mutable std::vector<std::uint64_t> issued_;  // sous mutex_ : dernière valeur promise par slot
};

}  // namespace ajp::app::play
