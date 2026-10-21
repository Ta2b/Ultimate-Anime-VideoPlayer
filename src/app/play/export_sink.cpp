#include "app/play/export_sink.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <format>
#include <thread>
#include <utility>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

#include "core/log.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

kernels::Matrix matrix_of(ColorMatrix matrix) {
  switch (matrix) {
    case ColorMatrix::Bt601:
      return kernels::Matrix::Bt601;
    case ColorMatrix::Bt2020Ncl:
      return kernels::Matrix::Bt2020Ncl;
    case ColorMatrix::Bt709:
    case ColorMatrix::Rgb:
      break;
  }
  return kernels::Matrix::Bt709;
}

kernels::Siting siting_of(ChromaSiting siting) {
  switch (siting) {
    case ChromaSiting::Center:
      return kernels::Siting::Center;
    case ChromaSiting::TopLeft:
      return kernels::Siting::TopLeft;
    case ChromaSiting::Left:
      break;
  }
  return kernels::Siting::Left;
}

}  // namespace

FrameSource pool_frames(AVBufferRef* frames) {
  return [frames]() -> std::expected<AVFrame*, Refusal> {
    AVFrame* frame = av_frame_alloc();
    if (frame == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "export : image de l'encodeur");
    // Le pool de l'encodeur grandit à la demande (VideoEncoder l'alloue d'avance pour l'export).
    if (const int err = av_hwframe_get_buffer(frames, frame, 0); err < 0) {
      av_frame_free(&frame);
      return refuse(RefusalCode::GpuMemoryExhausted, std::format("export : av_hwframe_get_buffer : {}", err));
    }
    return frame;
  };
}

std::optional<std::string> sink_blocker(bool yuv_targets, bool cuda_frames, exporter::Layout layout,
                                        ColorMatrix matrix, ExportSize size, ExportSize chain) {
  if (!yuv_targets) return "le backend n'écrit pas en YUV";
  if (!cuda_frames) return "l'encodeur ne lit pas la mémoire CUDA";
  if (layout != exporter::Layout::P010 && layout != exporter::Layout::Nv12)
    return "disposition de l'encodeur autre que P010 ou NV12";
  if (matrix == ColorMatrix::Rgb) return "flux étiqueté RGB";
  if (size != chain) {
    return std::format("taille encodée {}x{} autre que la sortie de la chaîne {}x{}", size.width, size.height,
                       chain.width, chain.height);
  }
  return std::nullopt;
}

ExportSinkSetup sink_setup(FrameSource frames, ExportSize size, exporter::Layout layout,
                           const ColorInfo& color, std::uint32_t slots) {
  return ExportSinkSetup{
      .frames = std::move(frames),
      .width = size.width,
      .height = size.height,
      .format = layout == exporter::Layout::Nv12 ? kernels::Format::Nv12 : kernels::Format::P010,
      .matrix = matrix_of(color.matrix),
      .range = color.range == ColorRange::Full ? kernels::Range::Full : kernels::Range::Limited,
      .siting = siting_of(color.siting),
      .slots = slots};
}

ExportSink::ExportSink(ExportSinkSetup setup) : setup_(std::move(setup)) {}

std::expected<std::shared_ptr<ExportSink>, Refusal> ExportSink::open(ExportSinkSetup setup) {
  if (setup.slots == 0 || setup.slots > kMaxRingSlots)
    return refuse(RefusalCode::FieldOutOfRange, std::format("export : 1 à {} slots", kMaxRingSlots));
  if (!kernels::valid_frame_dimension(setup.width) || !kernels::valid_frame_dimension(setup.height)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("export : {}x{}, dimensions paires attendues", setup.width, setup.height));
  }
  if (setup.format != kernels::Format::Nv12 && setup.format != kernels::Format::P010)
    return refuse(RefusalCode::FormatUnsupported, "export : NV12 ou P010 attendu");
  if (!setup.frames) return refuse(RefusalCode::FieldOutOfRange, "export : pas de source d'images");
  std::shared_ptr<ExportSink> sink(new ExportSink(std::move(setup)));
  const ExportSinkSetup& s = sink->setup_;
  for (std::uint32_t i = 0; i < s.slots; ++i) {
    auto frame = s.frames();
    if (!frame) return std::unexpected(std::move(frame.error()));
    sink->frames_.push_back(*frame);  // rendue par le destructeur, même refusée
    if (auto ok = sink->check(**frame); !ok) return std::unexpected(std::move(ok.error()));
    sink->descs_.push_back(
        RingSlotDesc{.index = i, .generation = s.generation, .width = s.width, .height = s.height});
  }
  sink->issued_.assign(s.slots, 0);
  return sink;
}

ExportSink::~ExportSink() {
  // Les écritures promises et pas encore signalées : le GPU écrit peut-être encore leurs images.
  const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(kDrainTimeoutUs);
  bool drained = false;
  while (true) {
    drained = true;
    {
      const std::scoped_lock lock(mutex_);
      for (std::size_t i = 0; i < issued_.size(); ++i)
        drained = drained && fences_->done[i].load(std::memory_order_acquire) >= issued_[i];
    }
    if (drained || std::chrono::steady_clock::now() >= until) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!drained) {
    // Un rappel du backend peut encore venir : ses signaux et les images restent, perdus.
    log_channel("app")->warn("export : écritures YUV jamais signalées, leurs images ne sont pas rendues");
    [[maybe_unused]] const Fences* kept = fences_.release();
    return;
  }
  for (AVFrame*& frame : frames_) av_frame_free(&frame);
}

std::expected<void, Refusal> ExportSink::check(const AVFrame& frame) const {
  const std::size_t row =
      static_cast<std::size_t>(setup_.width) * static_cast<std::size_t>(kernels::info(setup_.format).bytes);
  for (int p = 0; p < 2; ++p) {
    if (frame.data[p] == nullptr || frame.linesize[p] <= 0 ||
        static_cast<std::size_t>(frame.linesize[p]) < row)
      return refuse(RefusalCode::ShapeInvalid, std::format("export : plan {} de l'image de l'encodeur", p));
  }
  if (frame.width != setup_.width || frame.height != setup_.height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("export : image de l'encodeur {}x{} au lieu de {}x{}", frame.width,
                              frame.height, setup_.width, setup_.height));
  }
  return {};
}

gpu::YuvTarget ExportSink::target(std::uint32_t slot, std::uint64_t done_value) const {
  const std::scoped_lock lock(mutex_);
  if (slot >= frames_.size()) return {};  // write_yuv refuse une cible vide
  const AVFrame& frame = *frames_[slot];
  issued_[slot] = std::max(issued_[slot], done_value);
  // En mémoire CUDA, data est l'adresse d'appareil : FFmpeg ne la déréférence pas.
  return gpu::YuvTarget{.luma = std::bit_cast<std::uintptr_t>(frame.data[0]),
                        .chroma = std::bit_cast<std::uintptr_t>(frame.data[1]),
                        .luma_pitch = static_cast<std::size_t>(frame.linesize[0]),
                        .chroma_pitch = static_cast<std::size_t>(frame.linesize[1]),
                        .width = setup_.width,
                        .height = setup_.height,
                        .format = setup_.format,
                        .matrix = setup_.matrix,
                        .range = setup_.range,
                        .siting = setup_.siting,
                        .dither = kernels::Dither::Bayer8,
                        .done = &fences_->done[slot],
                        .done_value = done_value};
}

bool ExportSink::ready(const RingFrame& frame) const {
  return frame.slot < setup_.slots &&
         fences_->done[frame.slot].load(std::memory_order_acquire) >= frame.ready_value;
}

std::expected<AVFrame*, Refusal> ExportSink::take(std::uint32_t slot) {
  if (slot >= setup_.slots)
    return refuse(RefusalCode::FieldOutOfRange, std::format("export : slot {}", slot));
  auto fresh = setup_.frames();
  if (!fresh) return std::unexpected(std::move(fresh.error()));
  if (auto ok = check(**fresh); !ok) {
    av_frame_free(&*fresh);
    return std::unexpected(std::move(ok.error()));
  }
  const std::scoped_lock lock(mutex_);
  return std::exchange(frames_[slot], *fresh);
}

}  // namespace ajp::app::play
