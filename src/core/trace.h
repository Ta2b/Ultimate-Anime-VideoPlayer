// Plages de trace pour Nsight Systems (NVTX). Sans KIRARI_WITH_NVTX (CMake : KIRARI_WITH_CUDA et les
// en-têtes NVTX3 du kit CUDA), KIRARI_TRACE ne produit rien : aucun coût, aucune dépendance.
#pragma once

#if defined(KIRARI_WITH_NVTX) && __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
namespace kirari::core {
struct TraceDomain {
  static constexpr const char* name{"kirari"};
};
}  // namespace kirari::core
#define KIRARI_TRACE_JOIN2(a, b) a##b
#define KIRARI_TRACE_JOIN(a, b) KIRARI_TRACE_JOIN2(a, b)
#define KIRARI_TRACE(label)                                                                            \
  nvtx3::scoped_range_in<kirari::core::TraceDomain> KIRARI_TRACE_JOIN(kirari_trace_range_, __LINE__) { \
    label                                                                                              \
  }
#else
#define KIRARI_TRACE(label) ((void)0)
#endif
