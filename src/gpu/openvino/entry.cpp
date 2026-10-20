// Points d'entrée de ajp_backend_openvino (gpu/backend.h) : le backend Vulkan, ses réseaux par
// OpenVINO sur un GPU Intel (entry_points.h).
#include "gpu/openvino/openvino_runtime.h"
#include "gpu/vulkan/backend/entry_points.h"

namespace {

ajp::gpu::vk::BackendOptions openvino_options() {
  return {.runtime = &ajp::gpu::openvino_rt::make_runtime};
}

}  // namespace

AJP_VULKAN_BACKEND_ENTRY_POINTS("OpenVINO", openvino_options)
