// Les options des morceaux du GMFSS natif (gmfss_mix.h).
#include "gpu/cuda/backend/gmfss_mix.h"

#include <utility>

namespace kirari::gpu::cuda::gmfss_mix {

std::vector<std::vector<int>> candidates(const std::vector<bool>& heads) {
  std::vector<std::vector<int>> out;
  for (const bool head : heads) {
    std::vector<int> rungs{kF16, kMixed, kF32};
    if (head) rungs.insert(rungs.begin() + 1, kHeadF32);
    out.push_back(std::move(rungs));
  }
  return out;
}

native_mix::Options uniform(std::size_t pieces, int rung) {
  return native_mix::Options(pieces, rung);
}

std::vector<RifeNativePrecision> precisions(const native_mix::Options& options) {
  std::vector<RifeNativePrecision> out;
  out.reserve(options.size());
  for (const int o : options) out.push_back(kLadder[static_cast<std::size_t>(o)]);
  return out;
}

native_mix::Problem problem(native_mix::Grid times, const std::vector<bool>& heads, std::vector<double> peak,
                            std::vector<double> floor, double min_db) {
  return native_mix::Problem{.times = std::move(times),
                             .cands = candidates(heads),
                             .base = uniform(heads.size(), kF32),
                             .seeds = {uniform(heads.size(), kF16), uniform(heads.size(), kMixed)},
                             .peak = std::move(peak),
                             .floor = std::move(floor),
                             .min_db = min_db};
}

std::string mix_name(const std::vector<std::string>& pieces, const native_mix::Options& options) {
  std::string name;
  for (std::size_t p = 0; p < options.size() && p < pieces.size(); ++p) {
    if (p > 0) name += ", ";
    name += pieces[p] + " " + kLadderNames[static_cast<std::size_t>(options[p])];
  }
  return name;
}

}  // namespace kirari::gpu::cuda::gmfss_mix
