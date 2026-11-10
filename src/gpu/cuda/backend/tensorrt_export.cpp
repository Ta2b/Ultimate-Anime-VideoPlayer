// Export par puits CUDA du backend TensorRT (KernelOps::write_yuv, export_kernels.h) : la sortie
// de la chaîne écrite en NV12 ou P010 dans l'image du pool de NVENC, sur la voie courante, puis
// l'image dite finie par un rappel hôte du flux.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <format>
#include <new>
#include <span>
#include <string>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"
#include "gpu/cuda/backend/export_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/ref/resample.h"
#include "kernels/spec/color.h"

namespace ajp::gpu::cuda {
namespace {

bool rgb_tensor(const Tensor& tensor, int width, int height) {
  return tensor.type == TensorType::F16 && tensor.shape.dims[0] == 1 && tensor.shape.dims[1] == 3 &&
         tensor.shape.dims[2] == height && tensor.shape.dims[3] == width && tensor.handle != 0;
}

void* address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

void* device_pointer(std::uintptr_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

// Le signal de YuvTarget, porté jusqu'au rappel hôte qui le range puis le libère.
struct DoneSignal {
  std::atomic<std::uint64_t>* done = nullptr;
  std::uint64_t value = 0;
};

void signal_done(void* opaque) {
  auto* signal = static_cast<DoneSignal*>(opaque);
  signal->done->store(signal->value, std::memory_order_release);
  delete signal;  // NOLINT(cppcoreguidelines-owning-memory) : alloué par write_yuv pour ce rappel
}

// Une table de poids copiée sur l'appareil : début de chaque sortie et poids, à la suite. Sur la file de la
// voie, dans le pool (étape 11.4) : ni cudaMalloc ni cudaMemcpy, qui attendent l'appareil entier ; la file
// est attendue avant de rendre la main, les tables de l'hôte meurent au retour.
std::expected<void, Refusal> upload(const kernels::ref::Weights& table, int*& start, float*& weights,
                                    cudaStream_t stream) {
  const std::size_t starts = table.start.size() * sizeof(int);
  const std::size_t taps = table.w.size() * sizeof(float);
  if (auto ok =
          cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&start), starts, stream), "cudaMallocAsync");
      !ok)
    return ok;
  if (auto ok =
          cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&weights), taps, stream), "cudaMallocAsync");
      !ok)
    return ok;
  if (auto ok = cuda_check(cudaMemcpyAsync(start, table.start.data(), starts, cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync");
      !ok) {
    return ok;
  }
  if (auto ok = cuda_check(cudaMemcpyAsync(weights, table.w.data(), taps, cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync");
      !ok) {
    return ok;
  }
  return cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

}  // namespace

void TensorRtBackend::release_yuv_tables(bool behind_lanes) {
  // Les écritures qui lisent ces tables peuvent être en vol : en lecture, rendues derrière toutes les voies
  // (free_after_lanes) ; à la destruction, cudaFree attend la fin de tout ce que l'appareil exécute.
  for (void* p : {static_cast<void*>(yuv_.x_start), static_cast<void*>(yuv_.x_weights),
                  static_cast<void*>(yuv_.y_start), static_cast<void*>(yuv_.y_weights), yuv_.rows}) {
    if (p == nullptr) continue;
    if (behind_lanes) {
      free_after_lanes(p);
    } else {
      cuda_warn(cudaFree(p), "cudaFree");
    }
  }
  yuv_ = YuvTables{};
}

std::expected<void, Refusal> TensorRtBackend::yuv_tables(int width, int height, kernels::Siting siting) {
  if (yuv_.rows != nullptr && yuv_.width == width && yuv_.height == height && yuv_.siting == siting)
    return {};
  release_yuv_tables(true);
  const kernels::ChromaShift shift = kernels::chroma_shift_down(siting);
  const auto wx = kernels::ref::compute_weights(width, width / 2, shift.x, kernels::kChainChromaFilter);
  const auto wy = kernels::ref::compute_weights(height, height / 2, shift.y, kernels::kChainChromaFilter);
  if (!wx || !wy) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("write_yuv : table de chroma {}", wx ? wy.error().detail : wx.error().detail));
  }
  if (wx->taps > kMaxYuvTaps || wy->taps > kMaxYuvTaps) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("write_yuv : {} et {} prises, {} au plus", wx->taps, wy->taps, kMaxYuvTaps));
  }
  if (auto ok = upload(*wx, yuv_.x_start, yuv_.x_weights, stream_); !ok) return ok;
  if (auto ok = upload(*wy, yuv_.y_start, yuv_.y_weights, stream_); !ok) return ok;
  if (auto ok =
          cuda_check(cudaMallocAsync(&yuv_.rows, yuv_rows_bytes(width, height), stream_), "cudaMallocAsync");
      !ok)
    return ok;
  yuv_.width = width;
  yuv_.height = height;
  yuv_.siting = siting;
  yuv_.x_taps = wx->taps;
  yuv_.y_taps = wy->taps;
  return {};
}

std::expected<void, Refusal> TensorRtBackend::write_yuv(const Tensor& rgb, const YuvTarget& target) {
  const bool ten_bit = target.format == kernels::Format::P010;
  if (!ten_bit && target.format != kernels::Format::Nv12)
    return refuse(RefusalCode::FormatUnsupported, "write_yuv : cible NV12 ou P010 attendue");
  if (!kernels::valid_frame_dimension(target.width) || !kernels::valid_frame_dimension(target.height)) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("write_yuv : cible {}x{}, dimensions paires attendues", target.width, target.height));
  }
  if (!rgb_tensor(rgb, target.width, target.height)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("write_yuv : tenseur FP16 1×3×{}×{} attendu", target.height, target.width));
  }
  if (target.luma == 0 || target.chroma == 0 || target.done == nullptr)
    return refuse(RefusalCode::FieldOutOfRange, "write_yuv : plans ou signal de la cible absents");
  // Deux échantillons par écriture : adresses et pas alignés sur leur paire.
  const std::size_t pair = ten_bit ? 4 : 2;
  const std::size_t row = static_cast<std::size_t>(target.width) * pair / 2;
  if (target.luma_pitch < row || target.chroma_pitch < row || target.luma % pair != 0 ||
      target.chroma % pair != 0 || target.luma_pitch % pair != 0 || target.chroma_pitch % pair != 0) {
    return refuse(RefusalCode::ShapeInvalid, "write_yuv : plans de la cible trop étroits ou mal alignés");
  }
  if (auto ok = bind(); !ok) return ok;
  if (auto ok = yuv_tables(target.width, target.height, target.siting); !ok) return ok;
  // Les lignes réduites sont partagées par toutes les écritures : une seule à la fois, toutes voies.
  if (auto ok = order({rgb.handle}, {reinterpret_cast<std::uintptr_t>(&yuv_)}); !ok) return ok;

  const kernels::Coefficients k = kernels::coefficients(target.matrix);
  const kernels::YuvScale scale = kernels::yuv_scale(target.format, target.range);
  const kernels::Quantizer q = kernels::quantizer(target.format);
  const YuvWriteParams params{.rgb = address(rgb),
                              .width = target.width,
                              .height = target.height,
                              .luma = device_pointer(target.luma),
                              .chroma = device_pointer(target.chroma),
                              .luma_pitch = target.luma_pitch,
                              .chroma_pitch = target.chroma_pitch,
                              .ten_bit = ten_bit,
                              .dither = target.dither == kernels::Dither::Bayer8,
                              .kr = k.kr,
                              .kb = k.kb,
                              .yoff = scale.yoff,
                              .yscale = scale.yscale,
                              .coff = scale.coff,
                              .cscale = scale.cscale,
                              .qdiv = q.qdiv,
                              .qmax = q.qmax,
                              .x_start = yuv_.x_start,
                              .x_weights = yuv_.x_weights,
                              .x_taps = yuv_.x_taps,
                              .y_start = yuv_.y_start,
                              .y_weights = yuv_.y_weights,
                              .y_taps = yuv_.y_taps,
                              .rows = static_cast<float2*>(yuv_.rows)};
  GpuTiming& timing = stage("écriture YUV de l'export");
  timing.begin(stream_);
  if (auto ok = cuda_check(launch_write_yuv(params, stream_), "write_yuv<<<>>>"); !ok) return ok;
  timing.end(stream_);
  // Le signal écrit par le GPU derrière les noyaux (barrière mémoire comprise) : ni rappel hôte ni fil du
  // pilote à réveiller, et un contexte perdu ne laisse pas un rappel en suspens.
  if (const std::uintptr_t device = mapped_fence(target.done); device != 0) {
    const CUresult r =
        cuStreamWriteValue64(stream_, static_cast<CUdeviceptr>(device),
                             static_cast<cuuint64_t>(target.done_value), CU_STREAM_WRITE_VALUE_DEFAULT);
    if (r == CUDA_SUCCESS) {
      // WDDM (12.3) : le lot de lancements part tout de suite, signal compris ; sinon il attendrait le
      // prochain appel qui force l'envoi, une image plus tard.
      push_launches();
      return {};
    }
    // Refusé : le rappel hôte, comme sans bloc enregistré.
  }
  auto* signal = new (std::nothrow) DoneSignal{.done = target.done, .value = target.done_value};
  if (signal == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "write_yuv : signal : mémoire");
  if (auto ok = cuda_check(cudaLaunchHostFunc(stream_, signal_done, signal), "cudaLaunchHostFunc"); !ok) {
    delete signal;  // NOLINT(cppcoreguidelines-owning-memory) : le rappel ne viendra pas
    return ok;
  }
  push_launches();  // WDDM : le rappel part avec ce lot, pas avec l'image suivante
  return {};
}

std::uintptr_t TensorRtBackend::mapped_fence(const void* done) const {
  const auto* at = static_cast<const std::byte*>(done);
  for (const MappedFences& block : mapped_fences_) {
    if (at >= block.host && at + sizeof(std::uint64_t) <= block.host + block.bytes)
      return block.device + static_cast<std::uintptr_t>(at - block.host);
  }
  return 0;
}

bool TensorRtBackend::map_fences(std::span<std::byte> block) {
  if (block.empty() || !bind()) return false;
  if (mapped_fence(block.data()) != 0) return true;
  if (cudaHostRegister(block.data(), block.size(), cudaHostRegisterMapped | cudaHostRegisterPortable) !=
      cudaSuccess) {
    static_cast<void>(clear_error());
    return false;
  }
  void* device = nullptr;
  const auto give_up = [&] {
    static_cast<void>(clear_error());
    cuda_warn(cudaHostUnregister(block.data()), "cudaHostUnregister");
    return false;
  };
  if (cudaHostGetDevicePointer(&device, block.data(), 0) != cudaSuccess) return give_up();
  // Un essai : réécrire la valeur déjà là. Un appareil ou un pilote sans opérations mémoire de flux 64 bits
  // le refuse, le backend garde alors le rappel hôte.
  std::uint64_t current = 0;
  std::memcpy(&current, block.data(), sizeof(current));
  if (cuStreamWriteValue64(stream_, static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device)),
                           current, CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS ||
      cudaStreamSynchronize(stream_) != cudaSuccess) {
    return give_up();
  }
  mapped_fences_.push_back(
      {.host = block.data(), .bytes = block.size(), .device = reinterpret_cast<std::uintptr_t>(device)});
  return true;
}

void TensorRtBackend::unmap_fences(std::span<std::byte> block) {
  const auto it =
      std::ranges::find_if(mapped_fences_, [&](const MappedFences& m) { return m.host == block.data(); });
  if (it == mapped_fences_.end()) return;
  if (bind()) {
    // Les écritures de signaux encore en vol visent ce bloc : attendues sur chaque voie.
    for (cudaStream_t stream : streams_) {
      if (stream != nullptr) cuda_warn(cudaStreamSynchronize(stream), "cudaStreamSynchronize (signaux)");
    }
  }
  cuda_warn(cudaHostUnregister(it->host), "cudaHostUnregister");
  mapped_fences_.erase(it);
}

std::expected<void, Refusal> TensorRtBackend::health() {
  if (auto ok = bind(); !ok) return ok;
  for (cudaStream_t stream : streams_) {
    if (stream == nullptr) continue;
    const cudaError_t state = cudaStreamQuery(stream);
    if (state == cudaSuccess || state == cudaErrorNotReady) continue;
    if (sticky(state)) return std::unexpected(cuda_refusal(state, "état de l'appareil (cudaStreamQuery)"));
    // Une erreur d'appel passée s'efface ; une qui survit à sa lecture est un contexte perdu (A01-01).
    if (const cudaError_t kept = clear_error(); kept != cudaSuccess)
      return std::unexpected(lost_refusal(kept, "état de l'appareil (cudaStreamQuery)"));
  }
  return {};
}

}  // namespace ajp::gpu::cuda
