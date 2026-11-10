// Le plus grand E4M3 fini, à une seule place (A02-09) : sans std::expected, pour que span_fp8_net.cu (nvcc)
// l'inclue avec quantize.h.
#pragma once

namespace ajp::gpu::span_fp8 {

inline constexpr float kE4m3Max = 448.0F;

}  // namespace ajp::gpu::span_fp8
