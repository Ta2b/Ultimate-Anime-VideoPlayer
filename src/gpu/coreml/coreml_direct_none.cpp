// Hors d'un Mac (le runtime construit sous Linux pour ses tests) : pas de Core ML à appeler.
#include "gpu/coreml/coreml_direct.h"

namespace ajp::gpu::coreml_rt {

std::expected<std::unique_ptr<DirectModel>, Refusal> open_direct(const std::filesystem::path& /*compiled*/,
                                                                 Units /*units*/,
                                                                 std::span<const std::string> /*inputs*/,
                                                                 std::span<const std::string> /*outputs*/) {
  return std::unexpected(
      Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "Core ML direct : absent de ce système"});
}

}  // namespace ajp::gpu::coreml_rt
