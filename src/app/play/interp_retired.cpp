// Les choix d'interpolation retirés du menu (interp_retired.h).
#include "app/play/interp_retired.h"

namespace kirari::app::play {

namespace {

constexpr std::string_view kRife426 = "rife_v4.26.onnx";
constexpr std::string_view kRife425Lite = "rife_v4.25_lite.onnx";

}  // namespace

std::string_view nearest_rife_model(std::string_view model) {
  if (model == kRife426) return kRife425File;
  if (model == kRife425Lite) return kRife422LiteFile;
  return model;
}

void retire_interpolation(PlaySettings& settings) {
  settings.rife_model = std::string(nearest_rife_model(settings.rife_model));
  if (settings.motion_engine == MotionEngine::Gmfss || settings.gmfss_middle) {
    settings.motion_engine = MotionEngine::Rife;
    settings.rife_model = std::string(kRife425File);
    settings.gmfss_middle = false;
  }
}

}  // namespace kirari::app::play
