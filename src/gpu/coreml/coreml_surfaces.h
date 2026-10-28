// Les IOSurface des tenseurs de la chaîne (coreml_direct.h, allocate_surface) pour le code qui les
// donne lui-même au GPU, par Metal (les conversions de l'interpolation d'Apple, vt_gpu.h).
// Objective-C++, sur un Mac seulement.
#pragma once

#include <IOSurface/IOSurfaceRef.h>
#include <cstddef>
#include <memory>
#include <optional>

#include "gpu/coreml/coreml_direct.h"
#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::coreml_rt {

// Une IOSurface de surfaces prêtée au GPU tant que l'objet vit, comme pour une prédiction de Core
// ML direct : le CPU la déverrouille (ce que lui et Vulkan y ont écrit est visible au GPU), puis
// la reprend à la destruction (ce que le GPU y a écrit revient à la chaîne).
class LentSurface {
 public:
  LentSurface(std::shared_ptr<Surfaces> surfaces, const std::byte* data, IOSurfaceRef surface);
  LentSurface(const LentSurface&) = delete;
  LentSurface& operator=(const LentSurface&) = delete;
  LentSurface(LentSurface&& other) noexcept;
  LentSurface& operator=(LentSurface&&) = delete;
  ~LentSurface();
  IOSurfaceRef get() const { return surface_; }

 private:
  std::shared_ptr<Surfaces> surfaces_;
  const std::byte* data_;
  IOSurfaceRef surface_;  // retenue
};

// L'IOSurface qui commence à b.data et porte ce tenseur (son type, sa forme, sa taille), prêtée ;
// nullopt : b n'en est pas une de surfaces (mémoire ordinaire, FP32, autre forme), ou surfaces nul.
std::optional<LentSurface> lend_surface(const std::shared_ptr<Surfaces>& surfaces,
                                        const vk::net::HostBinding& b);

}  // namespace ajp::gpu::coreml_rt
