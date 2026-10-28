// Points d'entrée de ajp_backend_migraphx sous Windows (gpu/backend.h) : le backend Vulkan, ses
// réseaux par MIGraphX de Windows ML sur un GPU AMD RDNA 3 ou plus récent (entry_points.h).
#include "gpu/vulkan/backend/entry_points.h"
#include "gpu/winml/winml_runtime.h"

namespace {

ajp::gpu::vk::BackendOptions winml_options() {
  return {.runtime = &ajp::gpu::winml_rt::make_runtime};
}

}  // namespace

AJP_VULKAN_BACKEND_ENTRY_POINTS("MIGraphX (Windows ML)", winml_options)
