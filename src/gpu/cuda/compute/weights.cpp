#include "gpu/cuda/compute/weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <utility>

namespace ajp::gpu::cuda {
namespace {

std::size_t at(int i) {
  return static_cast<std::size_t>(i);
}

double kernel(kernels::Filter filter, double x) {
  return filter == kernels::Filter::Bilinear ? kernels::bilinear(x) : kernels::spline36(x);
}

// Spline36 et bilinéaire : compute_filter de zimg.
HostWeights kernel_weights(int src, int dst, double shift, kernels::Filter filter) {
  const double scale = static_cast<double>(dst) / static_cast<double>(src);
  const double step = std::min(scale, 1.0);
  const int taps = std::max(static_cast<int>(std::ceil(kernels::support(filter) / step)) * 2, 1);

  HostWeights w{.taps = taps, .src = src, .dst = dst, .start = std::vector<int>(at(dst)), .w = {}};
  w.w.resize(at(dst) * at(taps));
  std::vector<double> row(at(taps));
  for (int i = 0; i < dst; ++i) {
    const double pos = (i + 0.5) / scale + shift;
    const double begin = std::floor(pos - taps / 2.0 + 0.5) + 0.5;
    double sum = 0.0;
    for (int j = 0; j < taps; ++j) {
      row[at(j)] = kernel(filter, (begin + j - pos) * step);
      sum += row[at(j)];
    }
    w.start[at(i)] = static_cast<int>(begin - 0.5);
    for (int j = 0; j < taps; ++j) w.w[at(i) * at(taps) + at(j)] = static_cast<float>(row[at(j)] / sum);
  }
  return w;
}

// Area : recouvrement exact de l'empreinte [i src / dst, (i + 1) src / dst].
HostWeights area_weights(int src, int dst) {
  const auto footprint = [src, dst](int i) { return static_cast<double>(i) * src / dst; };
  int taps = 1;
  for (int i = 0; i < dst; ++i) {
    const int x0 = static_cast<int>(std::floor(footprint(i)));
    const int x1 = std::min(src, static_cast<int>(std::ceil(footprint(i + 1))));
    taps = std::max(taps, x1 - x0);
  }

  HostWeights w{.taps = taps, .src = src, .dst = dst, .start = std::vector<int>(at(dst)), .w = {}};
  w.w.assign(at(dst) * at(taps), 0.0F);
  for (int i = 0; i < dst; ++i) {
    const double a = footprint(i);
    const double b = footprint(i + 1);
    const int x0 = static_cast<int>(std::floor(a));
    const int x1 = std::min(src, static_cast<int>(std::ceil(b)));
    w.start[at(i)] = x0;
    for (int sx = x0; sx < x1; ++sx) {
      const double overlap = std::min(b, sx + 1.0) - std::max(a, static_cast<double>(sx));
      w.w[at(i) * at(taps) + at(sx - x0)] = static_cast<float>(overlap / (b - a));
    }
  }
  return w;
}

template <class T>
Result<DeviceBuffer> copy_to_device(DevicePool& pool, const std::vector<T>& host, const char* what,
                                    cudaStream_t stream) {
  auto buffer = pool.allocate(host.size() * sizeof(T), what);
  if (!buffer) return std::unexpected(buffer.error());
  if (auto ok = cuda_check(cudaMemcpyAsync(buffer->data(), host.data(), host.size() * sizeof(T),
                                           cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync (table de poids)");
      !ok) {
    return std::unexpected(ok.error());
  }
  return buffer;
}

}  // namespace

Result<HostWeights> compute_weights(int src, int dst, double shift, kernels::Filter filter) {
  if (!kernels::valid_plane_dimension(src) || !kernels::valid_plane_dimension(dst)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("poids : {} vers {} hors de [1, {}]", src, dst, kernels::kMaxDimension));
  }
  if (!std::isfinite(shift) || std::abs(shift) > 1.0) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("poids : décalage {} hors de [-1, 1]", shift));
  }
  if (filter != kernels::Filter::Area) return kernel_weights(src, dst, shift, filter);
  if (dst > src)
    return refuse(RefusalCode::ShapeInvalid, std::format("Area : {} vers {} agrandirait", src, dst));
  if (shift != 0.0)
    return refuse(RefusalCode::FieldOutOfRange, std::format("Area : décalage {} non nul", shift));
  return area_weights(src, dst);
}

Result<DeviceWeights> DeviceWeights::upload(DevicePool& pool, const HostWeights& weights,
                                            cudaStream_t stream) {
  DeviceWeights out;
  auto start = copy_to_device(pool, weights.start, "table de poids (débuts)", stream);
  if (!start) return std::unexpected(start.error());
  auto w = copy_to_device(pool, weights.w, "table de poids", stream);
  if (!w) return std::unexpected(w.error());
  // Les vecteurs hôtes ne vivent pas au-delà de l'appel.
  if (auto ok = cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize (table de poids)"); !ok) {
    return std::unexpected(ok.error());
  }
  out.start_ = std::move(*start);
  out.w_ = std::move(*w);
  out.view_ = {.start = out.start_.as<const int>(),
               .w = out.w_.as<const float>(),
               .taps = weights.taps,
               .src = weights.src,
               .dst = weights.dst};
  return out;
}

Result<DeviceWeights> DeviceWeights::create(DevicePool& pool, int src, int dst, double shift,
                                            kernels::Filter filter, cudaStream_t stream) {
  return compute_weights(src, dst, shift, filter).and_then([&](const HostWeights& weights) {
    return upload(pool, weights, stream);
  });
}

}  // namespace ajp::gpu::cuda
