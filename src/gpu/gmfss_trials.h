#pragma once
// Options GMFSS internes au lecteur et à son backend, dans BackendInit::trials.
#include <cstdint>

namespace kirari::gpu {

inline constexpr std::uint32_t kTrialGmflowFast = 1U << 12U;
inline constexpr std::uint32_t kTrialGmfssDirectSplats = 1U << 16U;

}  // namespace kirari::gpu
