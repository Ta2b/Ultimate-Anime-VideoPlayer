#include "prepare/trtexec_progress.h"

#include <array>

#include "core/text.h"

namespace ajp::prepare {
namespace {

struct Marker {
  std::string_view text;  // sous-chaîne cherchée dans la ligne
  float fraction;
  std::string_view stage;
};

// Les noms d'étape sont des clés anglaises : l'interface les traduit (app/play/build_progress.h,
// build_stage_text). Dans l'ordre d'apparition. Le choix des tactiques occupe l'essentiel du temps (22 s pour
// RIFE 4.26, 53 à 64 s pour SPAN sur la 4070 Ti).
constexpr std::array kMarkers = {
    Marker{"&&&& RUNNING", 0.02F, "Starting trtexec"},
    Marker{"Start parsing network model", 0.05F, "Reading the model"},
    Marker{"Finished parsing network model", 0.10F, "Choosing tactics"},
    Marker{"Total Activation Memory", 0.85F, "Planning memory"},
    Marker{"Engine generation completed", 0.90F, "Serializing the engine"},
    Marker{"Engine built in", 0.95F, "Engine built"},
    Marker{"&&&& PASSED", 1.0F, "Done"},
};

}  // namespace

std::string TrtexecProgress::reported_version(std::string_view line) {
  if (line.find("&&&& RUNNING") == std::string_view::npos) return {};
  constexpr std::string_view kTag = "[TensorRT v";
  const std::size_t at = line.find(kTag);
  if (at == std::string_view::npos) return {};
  const std::size_t start = at + kTag.size();
  std::size_t end = start;
  while (end < line.size() && line[end] >= '0' && line[end] <= '9') ++end;
  if (end == start || end >= line.size() || line[end] != ']') return {};
  return "v" + std::string(line.substr(start, end - start));
}

std::optional<BuildStage> TrtexecProgress::feed(std::string_view line) {
  if (line.find("&&&& FAILED") != std::string_view::npos) failed_ = true;
  if (line.find("&&&& PASSED") != std::string_view::npos) passed_ = true;
  if (line.find("[E]") != std::string_view::npos) last_error_ = core::trim(line);
  for (const Marker& marker : kMarkers) {
    if (line.find(marker.text) == std::string_view::npos || marker.fraction <= fraction_) continue;
    fraction_ = marker.fraction;
    return BuildStage{.fraction = marker.fraction, .stage = std::string(marker.stage)};
  }
  return std::nullopt;
}

}  // namespace ajp::prepare
