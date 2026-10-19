#include "core/plan.h"

#include <bit>
#include <string_view>

namespace ajp::core {
namespace {

class Fnv1a {
 public:
  void bytes(std::string_view data) {
    for (char c : data) {
      state_ ^= static_cast<std::uint8_t>(c);
      state_ *= kPrime;
    }
    integer(data.size());  // sépare les chaînes consécutives
  }
  void integer(std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      state_ ^= (value >> (8 * i)) & 0xFFU;
      state_ *= kPrime;
    }
  }
  void real(double value) { integer(std::bit_cast<std::uint64_t>(value)); }
  std::uint64_t value() const { return state_; }

 private:
  static constexpr std::uint64_t kPrime = 0x100000001b3ULL;
  std::uint64_t state_ = 0xcbf29ce484222325ULL;
};

std::uint64_t u(auto value) {
  return static_cast<std::uint64_t>(value);
}

}  // namespace

int UpscaleStep::factor() const {
  int total = 1;
  for (const ModelPass& pass : passes) total *= pass.factor;
  return total;
}

std::uint64_t compute_hash(const Plan& plan) {
  Fnv1a h;
  h.integer(plan.downscale.has_value());
  if (plan.downscale) {
    h.integer(u(plan.downscale->filter));
    h.integer(u(plan.downscale->height));
  }
  h.integer(plan.upscale.has_value());
  if (plan.upscale) {
    h.integer(plan.upscale->passes.size());
    for (const ModelPass& pass : plan.upscale->passes) {
      h.bytes(pass.model);
      h.integer(u(pass.factor));
    }
  }
  h.integer(plan.levels.size());
  for (const InterpLevel& level : plan.levels) {
    h.integer(u(level.factor.num));
    h.integer(u(level.factor.den));
    h.bytes(level.model);
    h.integer(u(level.policy));
    h.integer(u(level.pacing));
    h.integer(u(level.schedule));
  }
  h.integer(u(plan.hdr.kind));
  h.integer(u(plan.ring));
  h.integer(u(plan.output_width));
  h.integer(u(plan.output_height));
  h.real(plan.output_fps);
  h.integer(plan.cadence.has_value());
  if (plan.cadence) {
    h.integer(u(plan.cadence->factor.num));
    h.integer(u(plan.cadence->factor.den));
    h.integer(u(plan.cadence->grid));
    h.integer(u(plan.cadence->vsyncs_per_frame));
    h.real(plan.cadence->speed);
    h.real(plan.cadence->output_fps);
  }
  return h.value();
}

}  // namespace ajp::core
