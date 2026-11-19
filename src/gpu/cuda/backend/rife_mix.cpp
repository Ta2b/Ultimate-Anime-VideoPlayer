// Le problème du mélange du RIFE natif (rife_mix.h).
#include "gpu/cuda/backend/rife_mix.h"

#include <utility>

namespace kirari::gpu::cuda::rife_mix {

std::vector<std::vector<int>> candidates(const native_mix::Grid& times) {
  std::vector<std::vector<int>> out(times.size());
  for (std::size_t p = 0; p < times.size(); ++p) {
    out[p].push_back(0);
    for (std::size_t o = 1; o < times[p].size(); ++o)
      if (times[p][o] < times[p][0]) out[p].push_back(static_cast<int>(o));
  }
  return out;
}

native_mix::Problem problem(native_mix::Grid times, std::vector<double> peak, double min_db) {
  const std::size_t pieces = times.size();
  const std::size_t groups = peak.size();
  std::vector<std::vector<int>> cands = candidates(times);
  return native_mix::Problem{.times = std::move(times),
                             .cands = std::move(cands),
                             .base = native_mix::Options(pieces, 0),
                             .seeds = {},
                             .peak = std::move(peak),
                             .floor = std::vector<double>(groups, 0.0),
                             .min_db = min_db};
}

native_mix::Eval eval(const std::vector<double>& mse) {
  return native_mix::Eval{.want = mse, .base = mse};
}

native_mix::Options kept(const native_mix::Mix& mix, std::size_t pieces) {
  return mix.held ? mix.options : native_mix::Options(pieces, 0);
}

}  // namespace kirari::gpu::cuda::rife_mix
