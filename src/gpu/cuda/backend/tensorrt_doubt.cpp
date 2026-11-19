// Fondu local du mouvement de RIFE (doubt_kernels.h, kernels/spec/motion_doubt.h) dans le backend TensorRT.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/doubt_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "kernels/spec/motion_doubt.h"

namespace kirari::gpu::cuda {
namespace {

bool fp16_planes(const Tensor& tensor, std::int64_t channels) {
  return tensor.type == TensorType::F16 && tensor.handle != 0 && tensor.shape.dims[0] == 1 &&
         tensor.shape.dims[1] == channels && tensor.shape.dims[2] > 0 && tensor.shape.dims[3] > 0;
}

const void* address(const Tensor& tensor) {
  return reinterpret_cast<const void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

// L'entrée et la sortie : flot 1 × 4 et masque 1 × 1 FP16, tous aux mêmes plans, fenêtre dans les plans.
bool same_motion_shapes(const Motion& motion, const Motion& out) {
  if (!fp16_planes(motion.flow, 4) || !fp16_planes(motion.mask, 1) || !fp16_planes(out.flow, 4) ||
      !fp16_planes(out.mask, 1))
    return false;
  const auto& dims = motion.flow.shape.dims;
  if (out.flow.shape.dims != dims || motion.mask.shape.dims[2] != dims[2] ||
      motion.mask.shape.dims[3] != dims[3] || out.mask.shape.dims != motion.mask.shape.dims)
    return false;
  const kernels::Window& w = motion.area;
  return w.x >= 0 && w.y >= 0 && w.width >= 1 && w.height >= 1 && w.x + w.width <= dims[3] &&
         w.y + w.height <= dims[2];
}

}  // namespace

std::expected<void, Refusal> TensorRtBackend::motion_doubt(const Motion& motion,
                                                           std::span<const std::uint8_t> weights, int cols,
                                                           int rows, float fade, const Motion& out) {
  if (cols < 1 || rows < 1 || cols > kernels::kDoubtMaxCells || rows > kernels::kDoubtMaxCells ||
      weights.size() != static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("motion_doubt : grille {}x{} de {} poids", cols, rows, weights.size()));
  }
  if (!(fade >= 0.0F && fade <= 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("motion_doubt : masque du fondu {}", fade));
  if (!same_motion_shapes(motion, out)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "motion_doubt : flot FP16 1×4 et masque 1×1 de mêmes plans, fenêtre dedans, attendus");
  }
  if (auto ok = bind(); !ok) return ok;
  MotionDoubtParams params{.flow = address(motion.flow),
                           .mask = address(motion.mask),
                           .out_flow = const_cast<void*>(address(out.flow)),  // NOLINT : écrit
                           .out_mask = const_cast<void*>(address(out.mask)),  // NOLINT : écrit
                           .plane_width = static_cast<int>(motion.flow.shape.dims[3]),
                           .plane_height = static_cast<int>(motion.flow.shape.dims[2]),
                           .window_x = motion.area.x,
                           .window_y = motion.area.y,
                           .window_width = motion.area.width,
                           .window_height = motion.area.height,
                           .cols = cols,
                           .rows = rows,
                           .fade = fade};
  std::copy(weights.begin(), weights.end(), params.weights);
  if (auto ok = order({motion.flow.handle, motion.mask.handle}, {out.flow.handle, out.mask.handle}); !ok)
    return ok;
  return cuda_check(launch_motion_doubt(params, stream_), "motion_doubt<<<>>>");
}

}  // namespace kirari::gpu::cuda
