// Points d'entrée de kirari_backend_coreml (gpu/backend.h) : le backend Vulkan, ses réseaux par
// Core ML sur un Mac Apple Silicon (entry_points.h).
#include "gpu/coreml/coreml_runtime.h"
#include "gpu/vulkan/backend/entry_points.h"

namespace {

kirari::gpu::vk::BackendOptions coreml_options() {
  return {.runtime = &kirari::gpu::coreml_rt::make_runtime};
}

}  // namespace

KIRARI_VULKAN_BACKEND_ENTRY_POINTS("Core ML", coreml_options)
