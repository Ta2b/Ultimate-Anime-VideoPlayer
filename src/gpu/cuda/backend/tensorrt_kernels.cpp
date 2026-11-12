// KernelOps du backend TensorRT : ingestion des images NVDEC ou du décodage logiciel (4:2:0, 4:2:2 et
// 4:4:4, chroma suréchantillonnée au filtre demandé), et écriture du ring.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda.h>
#include <format>
#include <span>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/deband_kernels.h"
#include "gpu/cuda/backend/host_upload.h"
#include "gpu/cuda/backend/slice_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/color.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> not_here(std::string_view kernel) {
  return refuse(RefusalCode::FormatUnsupported, std::format("{} : pas dans le backend TensorRT", kernel));
}

// Les formats lus par l'ingestion : ceux de NVDEC (4:2:0 entrelacé), et le YUV planaire du décodage logiciel
// en 4:2:0, 4:2:2 et 4:4:4 (B01-03). scale : le format dont la normalisation vaut (Yuv422p16 range ses
// codes comme Yuv444p16) ; chroma : la disposition de la chroma (slice_kernels.h).
struct IngestFormat {
  kernels::Format scale = kernels::Format::Nv12;
  int chroma = kChroma420;
};

std::optional<IngestFormat> ingest_format(PixelFormat format, bool host) {
  switch (format) {
    case PixelFormat::Nv12:
      return IngestFormat{.scale = kernels::Format::Nv12};
    case PixelFormat::P010:
      return IngestFormat{.scale = kernels::Format::P010};
    case PixelFormat::P016:
      return IngestFormat{.scale = kernels::Format::P016};
    case PixelFormat::Yuv420p:
      if (host) return IngestFormat{.scale = kernels::Format::Yuv420p};
      return std::nullopt;
    case PixelFormat::Yuv420p10:
      if (host) return IngestFormat{.scale = kernels::Format::Yuv420p10};
      return std::nullopt;
    case PixelFormat::Yuv422p16:
      if (host) return IngestFormat{.scale = kernels::Format::Yuv444p16, .chroma = kChroma422};
      return std::nullopt;
    case PixelFormat::Yuv444p16:
      if (host) return IngestFormat{.scale = kernels::Format::Yuv444p16, .chroma = kChroma444};
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

const char* format_refusal(bool host) {
  return host ? "ingestion hôte : NV12, P010, P016, YUV 4:2:0 8 ou 10 bits, 4:2:2 ou 4:4:4 16 bits"
              : "ingestion : NV12, P010 ou P016 en YUV, chaîne d'agrandissement";
}

kernels::Siting siting_of(ChromaSiting siting) {
  switch (siting) {
    case ChromaSiting::Left:
      return kernels::Siting::Left;
    case ChromaSiting::Center:
      return kernels::Siting::Center;
    case ChromaSiting::TopLeft:
      return kernels::Siting::TopLeft;
  }
  return kernels::Siting::Left;
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

kernels::Range range_of(ColorRange range) {
  return range == ColorRange::Full ? kernels::Range::Full : kernels::Range::Limited;
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
    free = &*std::ranges::min_element(held_, {}, &Held::sequence);
    if (auto ok = wait_event(free->done); !ok) return ok;
    free->data.reset();
  }
  free->data = frame.data;
  free->sequence = ++held_sequence_;
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
  if (!format || !matrix) return refuse(RefusalCode::FormatUnsupported, format_refusal(host));
  // Tenseur à la taille de l'image, ou plus petit : réduction Area fusionnée (slice_kernels.h).
  const int out_width = static_cast<int>(rgb.shape.dims[3]);
  const int out_height = static_cast<int>(rgb.shape.dims[2]);
  if (out_width < 1 || out_height < 1 || out_width > frame.width || out_height > frame.height ||
      !rgb_tensor(rgb, out_width, out_height)) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("ingestion : tenseur FP16 1×3×h×w attendu, au plus {}×{} (Area ne fait que réduire)",
                    frame.height, frame.width));
  }
  pair_levels_ = kernels::pair_levels(range_of(frame.color.range));  // pas de luma de la source
  const kernels::YuvScale scale = kernels::yuv_scale(format->scale, range_of(frame.color.range));
  const kernels::Coefficients k = kernels::coefficients(*matrix);
  pair_k_ = k;  // la luma des mesures de paire est celle de la source (B03-06)
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
      .sixteen_bits = kernels::info(format->scale).bytes == 2,
      .yoff = scale.yoff,
      .yscale = scale.yscale,
      .coff = scale.coff,
      .cscale = scale.cscale,
      .kr = k.kr,
      .kb = k.kb,
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
  // La chroma au filtre demandé, comme l'ingestion Vulkan (B03-02).
  if (auto ok = chroma_tables(params, format->chroma, siting_of(frame.color.siting), spec.chroma); !ok)
    return ok;
  const std::size_t rows = ingest_rows_bytes(params);
  if (rows > ingest_rows_size_) {
    // Toutes les voies peuvent encore lire l'ancien tampon : il part en retrait, libéré quand elles
    // l'ont quitté, sans arrêter l'appareil en cours de lecture.
    if (ingest_rows_ != nullptr) free_after_lanes(ingest_rows_);
    ingest_rows_ = nullptr;
    ingest_rows_size_ = 0;
    if (auto ok = cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&ingest_rows_), rows, stream_),
                             "cudaMallocAsync");
        !ok)
      return ok;
    ingest_rows_size_ = rows;
  }
  params.rows = ingest_rows_;
  // Débandage : à même taille, dans l'ingestion elle-même (étape 5.4) ; réduite, l'ingestion écrit dans le
  // tampon de travail et le débandage dans rgb.
  const bool same_size = out_width == frame.width && out_height == frame.height;
  params.deband = spec.deband && same_size;
  const bool deband_pass = spec.deband && !same_size;
  const std::size_t deband_bytes =
      std::size_t{3} * static_cast<std::size_t>(out_width) * static_cast<std::size_t>(out_height) * 2U;
  if (deband_pass) {
    if (auto ok = reserve_deband(deband_bytes); !ok) return ok;
    params.rgb = deband_out_;
  }
  // Les lignes de chroma (4:2:0) sont réécrites à chaque image : une ressource écrite, comme rows.
  const std::array<std::uintptr_t, 4> writes{rgb.handle, reinterpret_cast<std::uintptr_t>(&ingest_rows_),
                                             reinterpret_cast<std::uintptr_t>(&chroma_.rows),
                                             reinterpret_cast<std::uintptr_t>(&deband_out_)};
  const std::array<std::uintptr_t, 1> reads{reinterpret_cast<std::uintptr_t>(&deband_out_)};
  if (auto ok =
          order(std::span(reads).first(deband_pass ? 1 : 0), std::span(writes).first(deband_pass ? 4 : 3));
      !ok)
    return ok;
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
  chain_span_->start(rgb.handle, stream_);  // la source, jusqu'à la fin de son SPAN (0.2)
  GpuTiming& timing = stage("ingestion");
  timing.begin(stream_);
  if (auto ok = cuda_check(launch_ingest(params, stream_), "ingest<<<>>>"); !ok) return ok;
  // Tenue dès l'ingestion enfilée, avant le débandage (release_source, A01-05).
  if (auto ok = release_source(host, frame); !ok) return ok;
  if (deband_pass) {
    const DebandParams deband{
        .in = deband_out_, .out = address(rgb), .width = out_width, .height = out_height};
    if (auto ok = cuda_check(launch_deband(deband, stream_), "deband<<<>>>"); !ok) return ok;
  }
  timing.end(stream_);
  return {};
}

std::expected<void, Refusal> TensorRtBackend::release_source(bool host, const DecodedFrame& frame) {
  // La source est tenue dès que l'ingestion qui la lit est enfilée : un débandage refusé ensuite ne la
  // rend pas au décodeur (ni le tampon épinglé à l'envoi suivant) pendant cette lecture (A01-05). Une
  // image hôte est déjà copiée : elle repart au décodeur sans attendre le GPU.
  if (host) return uploads_->consumed(stream_);
  return hold(frame);
}

std::expected<void, Refusal> TensorRtBackend::reserve_deband(std::size_t bytes) {
  if (bytes <= deband_out_size_) return {};
  // L'ancien tampon part en retrait, comme celui de l'ingestion.
  if (deband_out_ != nullptr) free_after_lanes(deband_out_);
  deband_out_ = nullptr;
  deband_out_size_ = 0;
  if (auto ok = cuda_check(cudaMallocAsync(&deband_out_, bytes, stream_), "cudaMallocAsync"); !ok) return ok;
  deband_out_size_ = bytes;
  return {};
}

std::optional<KernelOps::DeviceMemory> TensorRtBackend::device_memory() {
  if (!bind()) return std::nullopt;
  std::size_t free = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) return std::nullopt;
  std::size_t used = total - free;
  // Le pool garde la mémoire rendue (seuil de libération au maximum) : réservée mais libre, elle sert à la
  // prochaine allocation et n'est pas « pleine » pour l'OSD (B37-03).
  cudaMemPool_t pool = nullptr;
  std::uint64_t reserved = 0;
  std::uint64_t taken = 0;
  if (cudaDeviceGetDefaultMemPool(&pool, device_) != cudaSuccess ||
      cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) != cudaSuccess ||
      cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &taken) != cudaSuccess) {
    static_cast<void>(clear_error());  // un refus du pool ne reste pas au prochain appel
  } else if (reserved > taken) {
    used -= std::min(used, static_cast<std::size_t>(reserved - taken));
  }
  return DeviceMemory{.used = used, .total = total};
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
    log(BackendLogLevel::Info,
        std::format(
            "anneau : slot {} {}x{} importé dans CUDA, mémoire {} (décalage {}, {} octets)",  // i18n-ok
            desc.index, desc.width, desc.height, desc.dedicated ? "dédiée" : "partagée", desc.memory_offset,
            desc.memory_size));
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
  if (auto ok = bind_joined(&desc); !ok) return std::unexpected(std::move(ok.error()));
  const std::array scratch_writes{scratch};
  if (auto ok = order(std::span(&rgb.handle, 1), std::span(scratch_writes).first(scratch != 0 ? 1 : 0)); !ok)
    return std::unexpected(std::move(ok.error()));
  return begin_slot(desc, sync);
}

std::expected<void, Refusal> TensorRtBackend::bind_joined(const RingSlotDesc* desc) {
  if (auto ok = bind(); !ok) return ok;
  // Ring recréé : les écritures de l'ancienne génération, sur toutes les voies, avant le relâche de
  // ses imports (slot). Faite avant order() de l'opération qui écrit le premier slot de la
  // génération : au milieu, l'événement de l'opération serait posé avant ses derniers noyaux, et une
  // autre voie qui attend ce rang lirait une sortie à moitié écrite (A01-02).
  if (desc == nullptr || desc->generation <= generation_) return {};
  return join();
}

std::expected<ImportedSlot*, Refusal> TensorRtBackend::begin_slot(const RingSlotDesc& desc,
                                                                  const RingWrite& sync) {
  if (sync.signal_value <= sync.wait_value || sync.signal_value % 2 == 0) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("slot {} : valeur à signaler {} impaire et > {} attendue", desc.index,
                              sync.signal_value, sync.wait_value));
  }
  // La jointure d'un ring recréé est déjà faite (bind_joined, avant order de l'opération).
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
  return finish_slot(**imported, sync);
}

std::expected<void, Refusal> TensorRtBackend::finish_slot(const ImportedSlot& slot, const RingWrite& sync) {
  return slot.signal(sync.signal_value, stream_);
}

// À forme égale, un rééchantillonnage est l'identité : une copie appareil → appareil, dans l'ordre de la
// file (DistilDRBA, côté tenu). À forme différente, le backend ne sait pas faire (refus not_here).
std::expected<void, Refusal> TensorRtBackend::resample(const Tensor& in, kernels::Filter, const Tensor& out) {
  if (in.handle == 0 || out.handle == 0 || in.type != out.type || in.shape != out.shape ||
      in.type != TensorType::F16) {
    return not_here("resample");
  }
  const std::size_t bytes =
      static_cast<std::size_t>(in.shape.dims[0] * in.shape.dims[1] * in.shape.dims[2] * in.shape.dims[3]) *
      sizeof(std::uint16_t);
  if (auto ok = bind(); !ok) return ok;
  if (auto ok = order({in.handle}, {out.handle}); !ok) return ok;
  const CUresult copied = cuMemcpyDtoDAsync(static_cast<CUdeviceptr>(out.handle) + out.offset,
                                            static_cast<CUdeviceptr>(in.handle) + in.offset, bytes, stream_);
  if (copied != CUDA_SUCCESS)
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("cuMemcpyDtoDAsync : erreur {}", static_cast<int>(copied)));
  return {};
}

}  // namespace ajp::gpu::cuda
