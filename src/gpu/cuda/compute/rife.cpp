#include "gpu/cuda/compute/rife.h"

#include <format>

#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda {
namespace {

Result<> check_input(const Tensor& input) {
  for (const int n : {input.width, input.height}) {
    if (n < kernels::kRifeAlign || n > kernels::kMaxDimension || n % kernels::kRifeAlign != 0) {
      return refuse(RefusalCode::ShapeInvalid, std::format("entrée RIFE {}x{} : multiples de {} attendus",
                                                           input.width, input.height, kernels::kRifeAlign));
    }
  }
  return check_tensor(input, input.width, input.height, kernels::kRifeInputChannels, "entrée RIFE");
}

}  // namespace

Result<> rife_constants(const Tensor& input, cudaStream_t stream) {
  if (auto ok = check_input(input); !ok) return ok;
  return cuda_check(dev::rife_constants(channels(input, kernels::kRifeChannelMeshX, 4), stream),
                    "lancement de rife_constants");
}

Result<> rife_time(const Tensor& input, float t, cudaStream_t stream) {
  if (auto ok = check_input(input); !ok) return ok;
  if (!(t >= 0.0F && t <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("instant RIFE {} hors de [0, 1]", t));
  const std::size_t count = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);
  const Tensor plane = channels(input, kernels::kRifeChannelTime, 1);
  return cuda_check(dev::fill(plane.ptr, plane.elem, count, t, stream), "lancement de fill (instant RIFE)");
}

Result<> rife_crop(const Tensor& padded, const Tensor& out, cudaStream_t stream) {
  if (!kernels::valid_plane_dimension(out.width) || !kernels::valid_plane_dimension(out.height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("sortie RIFE {}x{} hors de [1, {}]", out.width,
                                                         out.height, kernels::kMaxDimension));
  }
  const kernels::RifeGeometry g = kernels::rife_geometry(out.width, out.height);
  if (auto ok = check_tensor(padded, g.padded_width, g.padded_height, 3, "sortie RIFE remplie"); !ok)
    return ok;
  if (auto ok = check_tensor(out, g.width, g.height, 3, "sortie RIFE recadrée"); !ok) return ok;
  return cuda_check(dev::crop(padded, g.left, g.top, out, stream), "lancement de crop");
}

}  // namespace ajp::gpu::cuda
