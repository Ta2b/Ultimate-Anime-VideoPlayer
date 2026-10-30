// RTX HDR (TrueHDR, Maxine sous Linux) dans le backend TensorRT de la tranche verticale.
#include "gpu/cuda/backend/tensorrt_true_hdr.h"

#include <atomic>
#include <format>
#include <string>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/hdr_kernels.h"
#include "vendor_hdr/open_true_hdr.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

void* address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

// Les refus du pont, au format de vendor_hdr.
vendor_hdr::Done bridge_check(cudaError_t error, std::string_view call) {
  if (error == cudaSuccess) return {};
  return std::unexpected(
      Refusal{RefusalCode::VendorCallFailed, std::format("{} : {}", call, cudaGetErrorString(error))});
}

}  // namespace

LegacyBridge::~LegacyBridge() {
  if (input_ready_ != nullptr) cudaEventDestroy(input_ready_);
  if (output_ready_ != nullptr) cudaEventDestroy(output_ready_);
}

std::expected<void, Refusal> LegacyBridge::create() {
  if (auto ok =
          cuda_check(cudaEventCreateWithFlags(&input_ready_, cudaEventDisableTiming), "cudaEventCreate");
      !ok)
    return ok;
  return cuda_check(cudaEventCreateWithFlags(&output_ready_, cudaEventDisableTiming), "cudaEventCreate");
}

vendor_hdr::Done LegacyBridge::hand_to_legacy() {
  if (auto ok = bridge_check(cudaEventRecord(input_ready_, stream_), "cudaEventRecord"); !ok) return ok;
  return bridge_check(cudaStreamWaitEvent(cudaStreamLegacy, input_ready_, 0), "cudaStreamWaitEvent(flux 0)");
}

vendor_hdr::Done LegacyBridge::take_from_legacy() {
  if (auto ok = bridge_check(cudaEventRecord(output_ready_, cudaStreamLegacy), "cudaEventRecord(flux 0)");
      !ok)
    return ok;
  return bridge_check(cudaStreamWaitEvent(stream_, output_ready_, 0), "cudaStreamWaitEvent");
}

TensorRtBackend::TrueHdrState::~TrueHdrState() {
  effect.reset();  // l'effet avant ses tampons
  if (rgba8 != nullptr) cudaFree(rgba8);
  if (rgb10 != nullptr) cudaFree(rgb10);
  if (luma_accum != nullptr) cudaFree(luma_accum);
  if (luma_host != nullptr) cudaFreeHost(luma_host);
}

std::expected<void, Refusal> TensorRtBackend::open_true_hdr(const vendor_hdr::InstalledSdk& sdk,
                                                            const vendor_hdr::TrueHdrConfig& config) {
  if (auto ok = bind(); !ok) return ok;
  true_hdr_.reset();
  auto state = std::make_unique<TrueHdrState>();
  state->width = config.width;
  state->height = config.height;
  const auto width_bytes = static_cast<std::size_t>(config.width) * 4;
  const auto rows = static_cast<std::size_t>(config.height);
  if (auto ok = cuda_check(cudaMallocPitch(&state->rgba8, &state->rgba8_pitch, width_bytes, rows),
                           "cudaMallocPitch");
      !ok)
    return ok;
  if (auto ok = cuda_check(cudaMallocPitch(&state->rgb10, &state->rgb10_pitch, width_bytes, rows),
                           "cudaMallocPitch");
      !ok)
    return ok;
  // Luminance moyenne de la source pour le grade adaptatif : un échec ne gêne pas RTX HDR (la mesure reste
  // absente).
  if (cudaMalloc(reinterpret_cast<void**>(&state->luma_accum), sizeof(float)) == cudaSuccess &&
      cudaMemset(state->luma_accum, 0, sizeof(float)) == cudaSuccess &&
      cudaHostAlloc(reinterpret_cast<void**>(&state->luma_host), sizeof(float), cudaHostAllocMapped) ==
          cudaSuccess &&
      cudaHostGetDevicePointer(reinterpret_cast<void**>(&state->luma_device), state->luma_host, 0) ==
          cudaSuccess) {
    *state->luma_host = -1.0F;  // pas encore de mesure
  } else {
    static_cast<void>(cudaGetLastError());
    if (state->luma_accum != nullptr) cudaFree(state->luma_accum);
    if (state->luma_host != nullptr) cudaFreeHost(state->luma_host);
    state->luma_accum = nullptr;
    state->luma_host = nullptr;
    state->luma_device = nullptr;
  }
  state->lane = lane_;
  state->bridge = std::make_unique<LegacyBridge>(stream_);
  if (auto ok = state->bridge->create(); !ok) return ok;
  // Maxine prend le contexte courant : le contexte primaire du backend, rendu courant par bind().
  auto effect = vendor_hdr::open_true_hdr(sdk, config);
  if (!effect) return refuse(effect.error().code, effect.error().detail);
  if ((*effect)->input_pixel() != vendor_hdr::TrueHdrPixel::Rgba8) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("RTX HDR : entrée {} attendue par le SDK, RGBA8 seulement ici",
                              vendor_hdr::name((*effect)->input_pixel())));
  }
  state->effect = std::move(*effect);
  true_hdr_ = std::move(state);
  return {};
}

std::expected<void, Refusal> TensorRtBackend::set_true_hdr(const vendor_hdr::TrueHdrParams& params) {
  if (!true_hdr_) return refuse(RefusalCode::HdrUnavailable, "RTX HDR : open_true_hdr n'a pas été appelé");
  auto valid = vendor_hdr::validate(params);
  if (!valid) return std::unexpected(std::move(valid.error()));
  return true_hdr_->effect->set_params(*valid);
}

std::expected<void, Refusal> TensorRtBackend::write_ring_true_hdr(
    const Tensor& rgb, const vendor_hdr::InputPreparation& prepared, const RingSlotDesc& desc,
    const RingWrite& sync) {
  if (!true_hdr_) return refuse(RefusalCode::HdrUnavailable, "RTX HDR : open_true_hdr n'a pas été appelé");
  if (desc.width != true_hdr_->width || desc.height != true_hdr_->height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("RTX HDR ouvert pour {}x{}, slot en {}x{}", true_hdr_->width, true_hdr_->height,
                              desc.width, desc.height));
  }
  const LaneScope on_bridge(*this, true_hdr_->lane);
  auto imported = begin_write(rgb, desc, sync, reinterpret_cast<std::uintptr_t>(true_hdr_.get()));
  if (!imported) return std::unexpected(std::move(imported.error()));
  TrueHdrState& hdr = *true_hdr_;
  hdr.prepared = prepared;
  const ToRgba8Params to8{.rgb = address(rgb),
                          .width = hdr.width,
                          .height = hdr.height,
                          .seed = hdr.frames++,
                          .out = hdr.rgba8,
                          .pitch = hdr.rgba8_pitch};
  GpuTiming& timing = stage("RTX HDR (tramage, TrueHDR, écriture)");
  timing.begin(stream_);
  if (auto ok = cuda_check(launch_to_rgba8(to8, stream_), "to_rgba8<<<>>>"); !ok) return ok;
  if (hdr.luma_device != nullptr) {
    const MeanLumaParams luma{.rgb = address(rgb),
                              .width = hdr.width,
                              .height = hdr.height,
                              .accum = hdr.luma_accum,
                              .out = hdr.luma_device};
    if (auto ok = cuda_check(launch_mean_luma(luma, stream_), "mean_luma<<<>>>"); !ok) return ok;
  }
  const vendor_hdr::TrueHdrInput in{
      .image = {.device = reinterpret_cast<std::uintptr_t>(hdr.rgba8),
                .pitch = hdr.rgba8_pitch,
                .width = hdr.width,
                .height = hdr.height,
                .pixel = vendor_hdr::TrueHdrPixel::Rgba8},
      .prepared = {.matrix = prepared.matrix, .range = prepared.range, .dithered = true}};
  const vendor_hdr::TrueHdrImage out{.device = reinterpret_cast<std::uintptr_t>(hdr.rgb10),
                                     .pitch = hdr.rgb10_pitch,
                                     .width = hdr.width,
                                     .height = hdr.height,
                                     .pixel = vendor_hdr::TrueHdrPixel::Rgb10A2};
  if (auto ok = hdr.effect->process(in, out, *hdr.bridge); !ok)
    return refuse(ok.error().code, ok.error().detail);
  const Rgb10a2ToSurfaceParams back{.in = hdr.rgb10,
                                    .pitch = hdr.rgb10_pitch,
                                    .width = hdr.width,
                                    .height = hdr.height,
                                    .surface = (*imported)->surface()};
  if (auto ok = cuda_check(launch_rgb10a2_to_surface(back, stream_), "rgb10a2_to_surface<<<>>>"); !ok)
    return ok;
  timing.end(stream_);
  return (*imported)->signal(sync.signal_value, stream_);
}

std::optional<float> TensorRtBackend::scene_luma() const {
  if (!true_hdr_ || true_hdr_->luma_host == nullptr) return std::nullopt;
  const float value = std::atomic_ref<float>(*true_hdr_->luma_host).load(std::memory_order_relaxed);
  if (!(value >= 0.0F)) return std::nullopt;  // -1 : aucune image mesurée
  return value;
}

std::expected<std::vector<float>, Refusal> TensorRtBackend::true_hdr_ramp(
    std::span<const std::uint8_t> levels) {
  if (!true_hdr_) return refuse(RefusalCode::HdrUnavailable, "RTX HDR : open_true_hdr n'a pas été appelé");
  TrueHdrState& hdr = *true_hdr_;
  const int bands = static_cast<int>(levels.size());
  if (bands == 0 || hdr.width / bands < 8)
    return refuse(RefusalCode::ShapeInvalid, std::format("rampe de {} marches sur {} px", bands, hdr.width));
  if (auto ok = bind(); !ok) return std::unexpected(ok.error());
  const LaneScope on_bridge(*this, hdr.lane);
  if (auto ok = order({}, {reinterpret_cast<std::uintptr_t>(true_hdr_.get())}); !ok)
    return std::unexpected(ok.error());
  // Une bande verticale par niveau, assez large pour que son centre ne voie pas ses voisines.
  const int band = hdr.width / bands;
  auto* rgba8 = static_cast<std::uint8_t*>(hdr.rgba8);
  for (int i = 0; i < bands; ++i) {
    const int x0 = i * band;
    const int x1 = i + 1 == bands ? hdr.width : x0 + band;
    if (auto ok = cuda_check(
            cudaMemset2DAsync(rgba8 + static_cast<std::ptrdiff_t>(x0) * 4, hdr.rgba8_pitch,
                              levels[static_cast<std::size_t>(i)], static_cast<std::size_t>(x1 - x0) * 4,
                              static_cast<std::size_t>(hdr.height), stream_),
            "cudaMemset2DAsync");
        !ok)
      return std::unexpected(ok.error());
  }
  const vendor_hdr::TrueHdrInput in{
      .image = {.device = reinterpret_cast<std::uintptr_t>(hdr.rgba8),
                .pitch = hdr.rgba8_pitch,
                .width = hdr.width,
                .height = hdr.height,
                .pixel = vendor_hdr::TrueHdrPixel::Rgba8},
      .prepared = {.matrix = hdr.prepared.matrix, .range = hdr.prepared.range, .dithered = true}};
  const vendor_hdr::TrueHdrImage out{.device = reinterpret_cast<std::uintptr_t>(hdr.rgb10),
                                     .pitch = hdr.rgb10_pitch,
                                     .width = hdr.width,
                                     .height = hdr.height,
                                     .pixel = vendor_hdr::TrueHdrPixel::Rgb10A2};
  if (auto ok = hdr.effect->process(in, out, *hdr.bridge); !ok)
    return refuse(ok.error().code, ok.error().detail);
  std::vector<std::uint32_t> row(static_cast<std::size_t>(hdr.width));
  const auto* middle =
      static_cast<const std::uint8_t*>(hdr.rgb10) +
      static_cast<std::ptrdiff_t>(hdr.height / 2) * static_cast<std::ptrdiff_t>(hdr.rgb10_pitch);
  if (auto ok = cuda_check(cudaMemcpyAsync(row.data(), middle, row.size() * sizeof(std::uint32_t),
                                           cudaMemcpyDeviceToHost, stream_),
                           "cudaMemcpyAsync");
      !ok)
    return std::unexpected(ok.error());
  if (auto ok = cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); !ok)
    return std::unexpected(ok.error());
  std::vector<float> pq;
  pq.reserve(levels.size());
  for (int i = 0; i < bands; ++i) {
    const std::uint32_t pixel = row[(static_cast<std::size_t>(i) * static_cast<std::size_t>(band)) +
                                    static_cast<std::size_t>(band / 2)];
    pq.push_back(static_cast<float>(pixel & 0x3FFU) / 1023.0F);  // R : les dix bits de poids faible
  }
  return pq;
}

}  // namespace ajp::gpu::cuda
