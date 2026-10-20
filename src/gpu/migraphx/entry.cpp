// Points d'entrée de ajp_backend_migraphx (gpu/backend.h) : le backend Vulkan, ses réseaux par
// MIGraphX sur un GPU AMD (entry_points.h).
#include "gpu/migraphx/migraphx_runtime.h"
#include "gpu/vulkan/backend/entry_points.h"

namespace {

ajp::gpu::vk::BackendOptions migraphx_options() {
  return {.runtime = &ajp::gpu::migraphx_rt::make_runtime};
}

}  // namespace

AJP_VULKAN_BACKEND_ENTRY_POINTS("MIGraphX", migraphx_options)
