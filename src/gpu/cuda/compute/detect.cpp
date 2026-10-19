#include "gpu/cuda/compute/detect.h"

#include <cstring>
#include <format>
#include <utility>

#include "gpu/cuda/compute/kernels/launch.h"
#include "kernels/spec/detect.h"

namespace ajp::gpu::cuda {
namespace {

dev::Luma luma(const YuvPlanes& image) {
  return {.plane = image.planes[0],
          .sample = sample_of(image.format),
          .width = image.width,
          .height = image.height};
}

}  // namespace

Result<PairMeter> PairMeter::create(DevicePool& pool, kernels::Format format, int width, int height) {
  if (!kernels::valid_plane_dimension(width) || !kernels::valid_plane_dimension(height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("mesure de paire : {}x{} hors de [1, {}]", width,
                                                         height, kernels::kMaxDimension));
  }
  PairMeter meter;
  meter.format_ = format;
  meter.width_ = width;
  meter.height_ = height;
  meter.tiles_ = dev::tile_count(width, height);
  auto partials =
      pool.allocate(static_cast<std::size_t>(meter.tiles_) * sizeof(double), "mesure de paire : tuiles");
  if (!partials) return std::unexpected(partials.error());
  auto result = pool.allocate(sizeof(dev::PairResult), "mesure de paire : résultat");
  if (!result) return std::unexpected(result.error());
  auto host = PinnedBuffer::create(sizeof(dev::PairResult));
  if (!host) return std::unexpected(host.error());
  auto ready = Event::create();
  if (!ready) return std::unexpected(ready.error());
  meter.partials_ = std::move(*partials);
  meter.result_ = std::move(*result);
  meter.host_ = std::move(*host);
  meter.ready_ = std::move(*ready);
  return meter;
}

Result<> PairMeter::measure(const YuvPlanes& a, const YuvPlanes& b, cudaStream_t stream) {
  if (auto ok = check_luma(a, format_, width_, height_, "mesure de paire (a)"); !ok) return ok;
  if (auto ok = check_luma(b, format_, width_, height_, "mesure de paire (b)"); !ok) return ok;
  auto* result = result_.as<dev::PairResult>();
  const kernels::RifeGeometry g = kernels::rife_geometry(width_, height_);
  const double area = static_cast<double>(g.padded_width) * static_cast<double>(g.padded_height);
  if (auto ok =
          cuda_check(cudaMemsetAsync(result, 0, sizeof(dev::PairResult), stream), "cudaMemsetAsync (paire)");
      !ok) {
    return ok;
  }
  if (auto ok =
          cuda_check(dev::pair_tiles(luma(a), luma(b), kernels::scene_norm(format_),
                                     kernels::held_scale(format_), partials_.as<double>(), result, stream),
                     "lancement de pair_tiles");
      !ok) {
    return ok;
  }
  if (auto ok = cuda_check(dev::pair_finish(partials_.as<const double>(), tiles_, area, result, stream),
                           "lancement de pair_finish");
      !ok) {
    return ok;
  }
  if (auto ok = cuda_check(
          cudaMemcpyAsync(host_.data(), result, sizeof(dev::PairResult), cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync (paire)");
      !ok) {
    return ok;
  }
  if (auto ok = ready_.record(stream); !ok) return ok;
  pending_ = true;
  return {};
}

Result<std::optional<PairScores>> PairMeter::previous() {
  if (!pending_) return std::nullopt;
  auto done = ready_.done();
  if (!done) return std::unexpected(done.error());
  if (!*done) return std::nullopt;
  pending_ = false;
  dev::PairResult r;
  std::memcpy(&r, host_.data(), sizeof(r));
  return PairScores{.scene_score = r.score, .worst_block = r.worst, .identical = r.identical != 0};
}

}  // namespace ajp::gpu::cuda
