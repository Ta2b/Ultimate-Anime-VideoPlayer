// Hors d'un Mac (le runtime construit sous Linux pour ses tests) : pas d'interpolation d'Apple.
#include "gpu/coreml/coreml_vt.h"

namespace ajp::gpu::coreml_rt {
namespace {

std::unexpected<Refusal> unavailable() {
  return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable,
                                 .detail = "interpolation d'Apple : VideoToolbox absent de ce système"});
}

}  // namespace

std::expected<std::string, Refusal> vt_probe(int /*width*/, int /*height*/) {
  return unavailable();
}

std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> vt_net(vk::net::IoNames /*names*/, int /*width*/,
                                                                    int /*height*/) {
  return unavailable();
}

}  // namespace ajp::gpu::coreml_rt
