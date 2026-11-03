// Plages de trace pour Nsight Systems (NVTX). Sans AJP_WITH_NVTX (CMake : AJP_WITH_CUDA et les
// en-têtes NVTX3 du kit CUDA), AJP_TRACE ne produit rien : aucun coût, aucune dépendance.
#pragma once

#if defined(AJP_WITH_NVTX) && __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
namespace ajp::core {
struct TraceDomain {
  static constexpr const char* name{"ajp"};
};
}  // namespace ajp::core
#define AJP_TRACE_JOIN2(a, b) a##b
#define AJP_TRACE_JOIN(a, b) AJP_TRACE_JOIN2(a, b)
#define AJP_TRACE(label)                                                                      \
  nvtx3::scoped_range_in<ajp::core::TraceDomain> AJP_TRACE_JOIN(ajp_trace_range_, __LINE__) { \
    label                                                                                     \
  }
#else
#define AJP_TRACE(label) ((void)0)
#endif
