// La composition par calques sous TensorRT (KernelOps::compose_layers, kernels/spec/layers.h) : le noyau de
// layer_kernels.cu sur le flux du backend, mesuré au profil de fin.
#include <cstdint>
#include <format>

#include "core/types.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/layer_kernels.h"
#include "gpu/cuda/backend/tensorrt_backend.h"

namespace kirari::gpu::cuda {
namespace {

bool rgb16(const Tensor& tensor) {
  return tensor.type == TensorType::F16 && tensor.handle != 0 && tensor.shape.dims[0] == 1 &&
         tensor.shape.dims[1] == 3 && tensor.shape.dims[2] > 0 && tensor.shape.dims[3] > 0;
}

bool same_size(const Tensor& x, const Tensor& y) {
  return x.shape.dims[2] == y.shape.dims[2] && x.shape.dims[3] == y.shape.dims[3];
}

void* address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

bool grid_ok(const kernels::LayerGrid& g) {
  return g.fits(g.cols, g.rows) && g.tile_w > 0.0F && g.tile_h > 0.0F;
}

}  // namespace

std::expected<void, Refusal> TensorRtBackend::compose_layers(const LayerJob& job) {
  const kernels::LayerParams& p = job.params;
  const bool two = p.count == 2;
  if (!rgb16(job.out) || !rgb16(job.a) || !same_size(job.out, job.a) ||
      (two && (!rgb16(job.b) || !same_size(job.out, job.b))))
    return refuse(RefusalCode::ShapeInvalid, "compose_layers : images RGB FP16 de même taille attendues");
  if (job.out.handle == job.a.handle || (two && job.out.handle == job.b.handle))
    return refuse(RefusalCode::ShapeInvalid, "compose_layers : la sortie est l'une des sources");
  if (p.count < 1 || p.count > 2 || !grid_ok(p.grid) ||
      (p.source_grid.cols != 0 && !grid_ok(p.source_grid)) || !(p.guard_high > p.guard_low))
    return refuse(RefusalCode::FieldOutOfRange, std::format("compose_layers : {} sources, grille {} × {}",
                                                            p.count, p.grid.cols, p.grid.rows));
  if (auto ok = bind(); !ok) return ok;
  LayerArgs args = layer_args(p);
  args.out = address(job.out);
  args.a = address(job.a);
  args.b = two ? address(job.b) : nullptr;
  args.width = static_cast<int>(job.out.shape.dims[3]);
  args.height = static_cast<int>(job.out.shape.dims[2]);
  if (two) {
    if (auto ok = order({job.out.handle, job.a.handle, job.b.handle}, {job.out.handle}); !ok) return ok;
  } else {
    if (auto ok = order({job.out.handle, job.a.handle}, {job.out.handle}); !ok) return ok;
  }
  GpuTiming& timing = stage(two ? "calques" : "plaque de décor");
  timing.begin(stream_);
  auto launched = cuda_check(launch_compose_layers(args, stream_), "compose_layers<<<>>>");
  timing.end(stream_);
  return launched;
}

}  // namespace kirari::gpu::cuda
