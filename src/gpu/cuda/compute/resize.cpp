#include "gpu/cuda/compute/resize.h"

#include <format>
#include <utility>

#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kMaxChannels = 16;

}  // namespace

Result<ResizePlan> ResizePlan::create(DevicePool& pool, const ResizeSpec& spec, cudaStream_t stream) {
  if (auto ok = check_frame(spec.src_width, spec.src_height, "source du rééchantillonnage"); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = check_frame(spec.dst_width, spec.dst_height, "cible du rééchantillonnage"); !ok) {
    return std::unexpected(ok.error());
  }
  if (spec.channels < 1 || spec.channels > kMaxChannels) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("rééchantillonnage : {} canaux hors de [1, {}]", spec.channels, kMaxChannels));
  }
  auto wx = DeviceWeights::create(pool, spec.src_width, spec.dst_width, 0.0, spec.filter, stream);
  if (!wx) return std::unexpected(wx.error());
  auto wy = DeviceWeights::create(pool, spec.src_height, spec.dst_height, 0.0, spec.filter, stream);
  if (!wy) return std::unexpected(wy.error());
  const std::size_t tmp = static_cast<std::size_t>(spec.channels) * static_cast<std::size_t>(spec.dst_width) *
                          static_cast<std::size_t>(spec.src_height);
  auto buffer = pool.allocate(tmp * sizeof(float), "rééchantillonnage : passe horizontale");
  if (!buffer) return std::unexpected(buffer.error());
  ResizePlan plan;
  plan.spec_ = spec;
  plan.wx_ = std::move(*wx);
  plan.wy_ = std::move(*wy);
  plan.tmp_ = std::move(*buffer);
  return plan;
}

Result<> ResizePlan::run(const Tensor& in, const Tensor& out, cudaStream_t stream) const {
  if (auto ok = check_tensor(in, spec_.src_width, spec_.src_height, spec_.channels, "rééchantillonnage");
      !ok) {
    return ok;
  }
  if (auto ok = check_tensor(out, spec_.dst_width, spec_.dst_height, spec_.channels, "rééchantillonnage");
      !ok) {
    return ok;
  }
  if (auto ok = cuda_check(dev::rows(in, spec_.channels, wx_.view(), tmp_.as<float>(), stream),
                           "lancement de rows");
      !ok) {
    return ok;
  }
  return cuda_check(
      dev::columns(tmp_.as<const float>(), spec_.dst_width, spec_.channels, wy_.view(), out, stream),
      "lancement de columns");
}

}  // namespace ajp::gpu::cuda
