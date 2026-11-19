// Points d'entrée de kirari_backend_vulkan (gpu/backend.h), seuls symboles exportés de la
// bibliothèque (entry_points.h).
#include "gpu/vulkan/backend/entry_points.h"

namespace {

kirari::gpu::vk::BackendOptions vulkan_options() {
  return {};
}

}  // namespace

KIRARI_VULKAN_BACKEND_ENTRY_POINTS("Vulkan", vulkan_options)
