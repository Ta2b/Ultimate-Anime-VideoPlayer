#include "present/output.h"

namespace ajp::present {

OutputHint first_hint(const core::DisplayInfo& display) {
  if (!display.hdr) return OutputHint::Sdr;
  return display.space == core::OutputSpace::ScRgb ? OutputHint::ScRgb : OutputHint::Pq;
}

OutputState classify_output(OutputHint requested, SurfaceTransfer obtained) {
  OutputState state{.hdr_requested = requested != OutputHint::Sdr};
  switch (obtained) {
    case SurfaceTransfer::Pq:
      state.space = core::OutputSpace::Pq;
      break;
    case SurfaceTransfer::Linear:
      state.space = core::OutputSpace::ScRgb;
      state.grade = false;
      break;
    case SurfaceTransfer::Other:
      state.space = core::OutputSpace::Srgb;
      break;
  }
  state.hdr_lost = state.hdr_requested && obtained != SurfaceTransfer::Pq;
  return state;
}

std::optional<OutputHint> fallback_hint(OutputHint requested, SurfaceTransfer obtained) {
  if (requested == OutputHint::Pq && obtained == SurfaceTransfer::Other) return OutputHint::ScRgb;
  return std::nullopt;
}

}  // namespace ajp::present
