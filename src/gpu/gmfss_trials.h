#pragma once
// Options GMFSS internes au lecteur et à son backend, dans BackendInit::trials.
#include <cstdint>

namespace kirari::gpu {

inline constexpr std::uint32_t kTrialGmflowFast = 1U << 12U;

}  // namespace kirari::gpu
