// Le fil des étapes différées d'un réseau Core ML (RuntimeNet::run_deferred), partagé par les
// réseaux d'ONNX Runtime (CoremlNet) et l'interpolation d'Apple (coreml_vt.h) : celui de tous les
// runtimes de l'hôte (vulkan/backend/deferred_steps.h), son fil nommé « coreml » suivi de label.
#pragma once

#include <string>
#include <utility>

#include "gpu/vulkan/backend/deferred_steps.h"

namespace ajp::gpu::coreml_rt {

class DeferredSteps : public vk::net::DeferredSteps {
 public:
  DeferredSteps(const std::string& label, Compute compute)
      : vk::net::DeferredSteps("coreml" + label, "Core ML", std::move(compute)) {}
};

}  // namespace ajp::gpu::coreml_rt
