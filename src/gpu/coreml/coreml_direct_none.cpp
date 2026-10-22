// Hors d'un Mac (le runtime construit sous Linux pour ses tests) : pas de Core ML à appeler, ni
// d'IOSurface à allouer.
#include "gpu/coreml/coreml_direct.h"

namespace ajp::gpu::coreml_rt {

class Surfaces {};

std::shared_ptr<Surfaces> make_surfaces() {
  return std::make_shared<Surfaces>();
}

std::expected<std::unique_ptr<vk::net::HostMemory>, Refusal> allocate_surface(
    const std::shared_ptr<Surfaces>& /*surfaces*/, std::uint64_t /*bytes*/, std::uint64_t /*alignment*/,
    TensorType /*type*/, const Shape& /*shape*/) {
  return std::unexpected(
      Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "IOSurface : absentes de ce système"});
}

std::expected<std::unique_ptr<DirectModel>, Refusal> open_direct(const std::filesystem::path& /*compiled*/,
                                                                 Units /*units*/,
                                                                 std::span<const std::string> /*inputs*/,
                                                                 std::span<const std::string> /*outputs*/,
                                                                 std::shared_ptr<Surfaces> /*surfaces*/) {
  return std::unexpected(
      Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "Core ML direct : absent de ce système"});
}

}  // namespace ajp::gpu::coreml_rt
