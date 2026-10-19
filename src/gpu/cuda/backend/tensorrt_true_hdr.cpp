// RTX HDR (TrueHDR, Maxine sous Linux) dans le backend TensorRT de la tranche verticale.
#include "gpu/cuda/backend/tensorrt_true_hdr.h"

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

std::expected<void, Refusal> TensorRtBackend::write_ring_true_hdr(
    const Tensor& rgb, const vendor_hdr::InputPreparation& prepared, const RingSlotDesc& desc,
    const RingWrite& sync) {
  if (!true_hdr_) return refuse(RefusalCode::HdrUnavailable, "RTX HDR : open_true_hdr n'a pas été appelé");
  if (desc.width != true_hdr_->width || desc.height != true_hdr_->height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("RTX HDR ouvert pour {}x{}, slot en {}x{}", true_hdr_->width, true_hdr_->height,
                              desc.width, desc.height));
  }
  auto imported = begin_write(rgb, desc, sync);
  if (!imported) return std::unexpected(std::move(imported.error()));
  TrueHdrState& hdr = *true_hdr_;
  const ToRgba8Params to8{.rgb = address(rgb),
                          .width = hdr.width,
                          .height = hdr.height,
                          .seed = hdr.frames++,
                          .out = hdr.rgba8,
                          .pitch = hdr.rgba8_pitch};
  if (auto ok = cuda_check(launch_to_rgba8(to8, stream_), "to_rgba8<<<>>>"); !ok) return ok;
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
  return (*imported)->signal(sync.signal_value, stream_);
}

}  // namespace ajp::gpu::cuda
