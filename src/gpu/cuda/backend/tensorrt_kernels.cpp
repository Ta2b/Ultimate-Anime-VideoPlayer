// KernelOps du backend TensorRT : ingestion des images NVDEC ou du décodage logiciel, et écriture
// du ring.
#include <algorithm>
#include <array>
#include <format>
#include <span>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/host_upload.h"
#include "gpu/cuda/backend/slice_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/color.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::unexpected<Refusal> not_here(std::string_view kernel) {
  return refuse(RefusalCode::FormatUnsupported,
                std::format("{} : pas encore dans le backend TensorRT de la tranche verticale", kernel));
}

// Les formats lus par l'ingestion : ceux de NVDEC, et le 4:2:0 planaire du décodage logiciel.
std::optional<kernels::Format> ingest_format(PixelFormat format, bool host) {
  switch (format) {
    case PixelFormat::Nv12:
      return kernels::Format::Nv12;
    case PixelFormat::P010:
      return kernels::Format::P010;
    case PixelFormat::P016:
      return kernels::Format::P016;
    case PixelFormat::Yuv420p:
      if (host) return kernels::Format::Yuv420p;
      return std::nullopt;
    case PixelFormat::Yuv420p10:
      if (host) return kernels::Format::Yuv420p10;
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

std::optional<kernels::Matrix> matrix_of(ColorMatrix matrix) {
  switch (matrix) {
    case ColorMatrix::Bt601:
      return kernels::Matrix::Bt601;
    case ColorMatrix::Bt709:
      return kernels::Matrix::Bt709;
    case ColorMatrix::Bt2020Ncl:
      return kernels::Matrix::Bt2020Ncl;
    case ColorMatrix::Rgb:
      return std::nullopt;
  }
  return std::nullopt;
}

// Tenseur RGB planaire 1 × 3 × h × w attendu par la chaîne.
bool rgb_tensor(const Tensor& tensor, int width, int height) {
  return tensor.type == TensorType::F16 && tensor.shape.dims[0] == 1 && tensor.shape.dims[1] == 3 &&
         tensor.shape.dims[2] == height && tensor.shape.dims[3] == width && tensor.handle != 0;
}

void* address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::expected<void, Refusal> TensorRtBackend::hold(const DecodedFrame& frame) {
  // Rend au pool de FFmpeg les images dont l'ingestion est finie ; s'il n'y a plus de place,
  // attend la plus ancienne (cas rare : le GPU a plus de kHeldFrames images de retard).
  Held* free = nullptr;
  for (Held& held : held_) {
    if (held.data && cudaEventQuery(held.done) == cudaSuccess) held.data.reset();
    if (!held.data && free == nullptr) free = &held;
  }
  if (free == nullptr) {
    free = &held_[0];
    if (auto ok = wait_event(free->done); !ok) return ok;
    free->data.reset();
  }
  free->data = frame.data;
  return cuda_check(cudaEventRecord(free->done, stream_), "cudaEventRecord");
}

std::expected<void, Refusal> TensorRtBackend::ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                                     const Tensor& rgb) {
  const bool host = frame.data && frame.data->memory() == FrameMemory::Host;
  if (!frame.data || (!host && frame.data->memory() != FrameMemory::Cuda)) {
    return refuse(RefusalCode::FormatUnsupported,
                  "ingestion : image CUDA (NVDEC) ou hôte (décodage logiciel) attendue");
  }
  const auto format = ingest_format(frame.format, host);
  const auto matrix = matrix_of(frame.color.matrix);
  if (!format || !matrix || spec.rife_conversion) {
    return refuse(RefusalCode::FormatUnsupported,
                  host
                      ? "ingestion hôte : NV12, P010, P016 ou YUV 4:2:0 8 ou 10 bits, chaîne d'agrandissement"
                      : "ingestion : NV12, P010 ou P016 en YUV, chaîne d'agrandissement");
  }
  // Tenseur à la taille de l'image, ou plus petit : réduction Area fusionnée (slice_kernels.h).
  // Extension de ce backend au contrat de KernelOps::ingest, signalée pour l'intégration.
  const int out_width = static_cast<int>(rgb.shape.dims[3]);
  const int out_height = static_cast<int>(rgb.shape.dims[2]);
  if (out_width < 1 || out_height < 1 || out_width > frame.width || out_height > frame.height ||
      !rgb_tensor(rgb, out_width, out_height)) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("ingestion : tenseur FP16 1×3×h×w attendu, au plus {}×{} (Area ne fait que réduire)",
                    frame.height, frame.width));
  }
  pair_levels_ = frame.color.range == ColorRange::Full ? 255.0F : 219.0F;  // pas de luma de la source
  const kernels::YuvScale scale = kernels::yuv_scale(
      *format, frame.color.range == ColorRange::Full ? kernels::Range::Full : kernels::Range::Limited);
  const kernels::Coefficients k = kernels::coefficients(*matrix);
  IngestParams params{
      .luma =
          reinterpret_cast<const void*>(frame.data->native_plane(0)),  // NOLINT(performance-no-int-to-ptr)
      .chroma =
          reinterpret_cast<const void*>(frame.data->native_plane(1)),  // NOLINT(performance-no-int-to-ptr)
      .luma_pitch = frame.data->native_pitch(0),
      .chroma_pitch = frame.data->native_pitch(1),
      .width = frame.width,
      .height = frame.height,
      .out_width = out_width,
      .out_height = out_height,
      .reduce_kind = spec.reduce.kind == ReduceKind::Triangle   ? kReduceTriangle
                     : spec.reduce.kind == ReduceKind::Cubic    ? kReduceCubic
                     : spec.reduce.kind == ReduceKind::Spline36 ? kReduceSpline36
                                                                : kReduceArea,
      .reduce_b = spec.reduce.b,
      .reduce_c = spec.reduce.c,
      .sixteen_bits = kernels::info(*format).bytes == 2,
      .yoff = scale.yoff,
      .yscale = scale.yscale,
      .coff = scale.coff,
      .cscale = scale.cscale,
      .kr = k.kr,
      .kb = k.kb,
      .chroma_x = frame.color.siting == ChromaSiting::Center ? 0.5F : 0.0F,
      .chroma_y = frame.color.siting == ChromaSiting::TopLeft ? 0.0F : 0.5F,
      .rgb = address(rgb),
  };
  if (!host && (params.luma == nullptr || params.chroma == nullptr || params.luma_pitch == 0 ||
                params.chroma_pitch == 0)) {
    return refuse(RefusalCode::FormatUnsupported, "ingestion : plans ou pas de ligne CUDA absents");
  }
  std::optional<HostUploads::Layout> layout;
  if (host) {
    layout = HostUploads::layout(frame.format, frame.width, frame.height);
    if (!layout) return refuse(RefusalCode::FormatUnsupported, "ingestion hôte : format sans disposition");
  }
  if (auto ok = bind(); !ok) return ok;
  const std::size_t rows = ingest_rows_bytes(params);
  if (rows > ingest_rows_size_) {
    // cudaFree attend la fin de tout ce que l'appareil exécute, l'ingestion précédente comprise.
    if (ingest_rows_ != nullptr) cudaFree(ingest_rows_);
    ingest_rows_ = nullptr;
    ingest_rows_size_ = 0;
    if (auto ok = cuda_check(cudaMalloc(reinterpret_cast<void**>(&ingest_rows_), rows), "cudaMalloc"); !ok)
      return ok;
    ingest_rows_size_ = rows;
  }
  params.rows = ingest_rows_;
  if (auto ok = order({}, {rgb.handle, reinterpret_cast<std::uintptr_t>(&ingest_rows_)}); !ok) return ok;
  if (host) {
    // Copie épinglée puis envoi sur la file de transfert, attendu par la voie courante.
    if (!uploads_) uploads_ = std::make_unique<HostUploads>();
    auto planes = uploads_->upload(*frame.data, *layout, stream_, stage("envoi de l'image décodée"));
    if (!planes) return std::unexpected(std::move(planes.error()));
    params.luma = planes->luma;
    params.chroma = planes->chroma;
    params.chroma_v = planes->chroma_v;
    params.luma_pitch = planes->luma_pitch;
    params.chroma_pitch = planes->chroma_pitch;
  } else if (frame.data->ready_event() != 0) {
    auto* ready =
        reinterpret_cast<cudaEvent_t>(frame.data->ready_event());  // NOLINT(performance-no-int-to-ptr)
    if (auto ok = cuda_check(cudaStreamWaitEvent(stream_, ready, 0), "cudaStreamWaitEvent"); !ok) return ok;
  }
  chain_timing_->begin(stream_);
  GpuTiming& timing = stage("ingestion");
  timing.begin(stream_);
  if (auto ok = cuda_check(launch_ingest(params, stream_), "ingest<<<>>>"); !ok) return ok;
  timing.end(stream_);
  // Une image hôte est déjà copiée : elle repart au décodeur sans attendre le GPU.
  if (host) return uploads_->consumed(stream_);
  return hold(frame);
}

std::optional<KernelOps::DeviceMemory> TensorRtBackend::device_memory() {
  if (!bind()) return std::nullopt;
  std::size_t free = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) return std::nullopt;
  return DeviceMemory{.used = total - free, .total = total};
}

void TensorRtBackend::release_retired() {
  if (!retired_.empty() && cudaEventQuery(retired_done_) == cudaSuccess) retired_.clear();
}

std::expected<ImportedSlot*, Refusal> TensorRtBackend::slot(const RingSlotDesc& desc) {
  if (desc.index >= 64)
    return refuse(RefusalCode::FieldOutOfRange, std::format("slot {} : 64 au plus", desc.index));
  if (desc.generation < generation_) {
    return refuse(RefusalCode::SequenceOrder, std::format("slot {} : génération {} remplacée par {}",
                                                          desc.index, desc.generation, generation_));
  }
  if (desc.generation > generation_) {
    // Ring recréé (kernel_ops.h) : les imports de l'ancienne génération se relâchent quand leurs
    // écritures déjà enfilées sont finies, constaté sans attente hôte aux appels suivants.
    for (auto& old : slots_) {
      if (old) retired_.push_back(std::move(old));
    }
    slots_.clear();
    generation_ = desc.generation;
    if (auto ok = cuda_check(cudaEventRecord(retired_done_, stream_), "cudaEventRecord"); !ok) {
      return std::unexpected(ok.error());
    }
  }
  release_retired();
  if (slots_.size() <= desc.index) slots_.resize(desc.index + 1);
  std::unique_ptr<ImportedSlot>& imported = slots_[desc.index];
  if (!imported) {
    auto fresh = ImportedSlot::import(desc);
    if (!fresh) return std::unexpected(std::move(fresh.error()));
    imported = std::move(*fresh);
  }
  const RingSlotDesc& known = imported->desc();
  if (known.width != desc.width || known.height != desc.height || known.format != desc.format) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("slot {} : taille ou format changés sans nouvelle génération", desc.index));
  }
  return imported.get();
}

std::expected<ImportedSlot*, Refusal> TensorRtBackend::begin_write(const Tensor& rgb,
                                                                   const RingSlotDesc& desc,
                                                                   const RingWrite& sync,
                                                                   std::uintptr_t scratch) {
  if (!rgb_tensor(rgb, desc.width, desc.height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("slot {} : tenseur FP16 1×3×{}×{} attendu",
                                                         desc.index, desc.height, desc.width));
  }
  if (auto ok = bind(); !ok) return std::unexpected(std::move(ok.error()));
  const std::array scratch_writes{scratch};
  if (auto ok = order(std::span(&rgb.handle, 1), std::span(scratch_writes).first(scratch != 0 ? 1 : 0)); !ok)
    return std::unexpected(std::move(ok.error()));
  return begin_slot(desc, sync);
}

std::expected<ImportedSlot*, Refusal> TensorRtBackend::begin_slot(const RingSlotDesc& desc,
                                                                  const RingWrite& sync) {
  if (sync.signal_value <= sync.wait_value || sync.signal_value % 2 == 0) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("slot {} : valeur à signaler {} impaire et > {} attendue", desc.index,
                              sync.signal_value, sync.wait_value));
  }
  // Ring recréé : les écritures de l'ancienne génération, sur toutes les voies, avant le relâche
  // de ses imports (slot).
  if (desc.generation > generation_) {
    if (auto ok = join(); !ok) return std::unexpected(std::move(ok.error()));
  }
  auto imported = slot(desc);
  if (!imported) return std::unexpected(std::move(imported.error()));
  if (sync.wait_value != 0) {
    if (auto ok = (*imported)->wait(sync.wait_value, stream_); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  return *imported;
}

std::expected<void, Refusal> TensorRtBackend::write_ring(const Tensor& rgb, const RingSlotDesc& desc,
                                                         const RingWrite& sync) {
  auto imported = begin_write(rgb, desc, sync);
  if (!imported) return std::unexpected(std::move(imported.error()));
  const WriteParams params{
      .rgb = address(rgb), .width = desc.width, .height = desc.height, .surface = (*imported)->surface()};
  GpuTiming& timing = stage("écriture du ring");
  timing.begin(stream_);
  if (auto ok = cuda_check(launch_write_rgba16(params, stream_), "write_rgba16<<<>>>"); !ok) return ok;
  timing.end(stream_);
  return (*imported)->signal(sync.signal_value, stream_);
}

std::expected<void, Refusal> TensorRtBackend::resample(const Tensor&, kernels::Filter, const Tensor&) {
  return not_here("resample");
}
std::expected<void, Refusal> TensorRtBackend::rife_pad(const DecodedFrame&, const Tensor&) {
  return not_here("rife_pad");
}
std::expected<void, Refusal> TensorRtBackend::measure(const DecodedFrame&, const DecodedFrame&) {
  return not_here("measure");
}

}  // namespace ajp::gpu::cuda
