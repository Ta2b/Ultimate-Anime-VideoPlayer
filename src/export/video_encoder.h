// L'encodeur vidéo d'un export : le premier candidat d'encoder_choice.h qui s'ouvre, ses images
// reçues en mémoire hôte (chargées sur des surfaces VA-API ou Vulkan, ou en mémoire CUDA, quand il
// le demande) ou déjà en mémoire CUDA (NVENC les lit telles quelles), ses paquets rendus un par un.
// Interne au module.
#pragma once

#include <expected>
#include <functional>
#include <string>

#include "core/frame.h"
#include "core/types.h"
#include "export/av_handles.h"
#include "export/encoder_choice.h"

namespace ajp::exporter {

// Ce que l'écriture (export_writer.cpp) attend d'un encodeur ouvert. VideoEncoder l'est par
// FFmpeg ; les tests de l'écriture le sont par un encodeur factice qui garde N images avant de les
// rendre, comme NVENC avec son anticipation (export_writer_test.cpp).
class FrameEncoder {
 public:
  using PacketSink = std::function<std::expected<void, Refusal>(AVPacket&)>;

  FrameEncoder() = default;
  FrameEncoder(const FrameEncoder&) = delete;
  FrameEncoder& operator=(const FrameEncoder&) = delete;
  FrameEncoder(FrameEncoder&&) = default;
  FrameEncoder& operator=(FrameEncoder&&) = default;
  virtual ~FrameEncoder() = default;

  virtual const AVCodecContext& context() const = 0;
  virtual Layout layout() const = 0;
  virtual Upload upload() const = 0;                // comment l'encodeur ouvert reçoit ses images
  virtual AVBufferRef* frames_context() const = 0;  // nullptr pour un encodeur hôte
  virtual bool shared_device() const = 0;
  virtual const std::string& label() const = 0;  // « NVENC (hevc_nvenc) »
  // Encode une image au format de layout(), datée dans la base de temps de context() : en mémoire
  // hôte, chargée sur l'appareil si l'encodeur le demande ; ou déjà au format de l'appareil
  // (AV_PIX_FMT_CUDA, hw_frames_ctx posé sur frames_context()), envoyée telle quelle. nullptr vide
  // l'encodeur. Chaque paquet prêt passe par sink. Refus : EncodeFailed.
  virtual std::expected<void, Refusal> encode(AVFrame* frame, const PacketSink& sink) = 0;
};

struct EncoderSetup {
  int width = 0;
  int height = 0;
  int fps_num = 0;
  int fps_den = 1;
  AVRational sample_aspect{0, 1};  // celui de la source ; {0, 1} : inconnu
  ColorInfo color;
  Codec codec = Codec::Hevc;
  Quality quality = Quality::VeryHigh;
  GpuVendor vendor = GpuVendor::Other;
  // Ordinal CUDA du GPU du rendu, pour un encodeur qui lit la mémoire CUDA (Upload::Cuda) : son
  // contexte primaire, celui des noyaux et de l'export ; -1 : inconnu, l'appareil 0.
  int cuda_ordinal = -1;
  // L'appareil Vulkan du rendu (AVHWDeviceContext, emprunté), pour un encodeur Vulkan Video : ses
  // images y sont rendues (shared_device()). nullptr : l'encodeur crée le sien.
  AVBufferRef* vulkan_device = nullptr;
  // Images de l'encodeur que l'export tient hors de lui (tampons de l'appelant et file d'écriture,
  // WriterConfig) : le pool CUDA les alloue en plus de celles que NVENC garde.
  int held_frames = 10;
};

class VideoEncoder final : public FrameEncoder {
 public:
  // Refus : EncoderUnavailable, avec chaque candidat essayé et sa raison.
  static std::expected<VideoEncoder, Refusal> open(const EncoderSetup& setup);

  const AVCodecContext& context() const override { return *ctx_; }
  Layout layout() const override { return layout_; }
  Upload upload() const override { return upload_; }
  AVBufferRef* frames_context() const override { return ctx_->hw_frames_ctx; }
  // Vulkan sur l'appareil du rendu : ses images (frames_context()) se rendent dans leurs plans
  // quand le GPU le permet (COLOR_ATTACHMENT), et s'envoient telles quelles (AV_PIX_FMT_VULKAN).
  bool shared_device() const override { return shared_; }
  const std::string& label() const override { return label_; }

  std::expected<void, Refusal> encode(AVFrame* frame, const PacketSink& sink) override;

 private:
  VideoEncoder() = default;
  std::expected<void, Refusal> drain(const PacketSink& sink);

  BufferPtr device_;  // appareil CUDA, VA-API ou Vulkan de l'encodeur ; vide pour un encodeur hôte
  CodecContextPtr ctx_;
  FramePtr hw_frame_;
  PacketPtr packet_;
  Upload upload_ = Upload::Host;
  Layout layout_ = Layout::P010;
  bool shared_ = false;
  std::string label_;
};

}  // namespace ajp::exporter
