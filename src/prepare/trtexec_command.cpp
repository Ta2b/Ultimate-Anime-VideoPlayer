// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:202-208, 812-860,
// 862-890 et 1313-1320. sanitize_trt11 garde l'algorithme du fork, espaces
// compris, et dit en plus ce qu'il retire.
#include "prepare/trtexec_command.h"

#include <algorithm>
#include <array>
#include <format>
#include <ranges>

#include "prepare/process.h"

namespace ajp::prepare {
namespace {

bool is_fixed(const gpu::ShapeSet& shapes) {
  return shapes.min.dims == shapes.opt.dims && shapes.opt.dims == shapes.max.dims;
}

// Formes de trtexec : --optShapes seul pour un moteur statique (le défaut du
// fork), les trois profils pour un moteur dynamique.
std::string profile_flags(std::string_view input, const gpu::ShapeSet& shapes) {
  if (is_fixed(shapes)) return std::format(" --optShapes={}:{}", input, shape_text(shapes.opt));
  return std::format(" --minShapes={0}:{1} --optShapes={0}:{2} --maxShapes={0}:{3}", input,
                     shape_text(shapes.min), shape_text(shapes.opt), shape_text(shapes.max));
}

bool positive(const gpu::Shape& shape) {
  return std::ranges::all_of(shape.dims, [](std::int64_t d) { return d > 0; });
}

bool not_above(const gpu::Shape& low, const gpu::Shape& high) {
  for (std::size_t i = 0; i < low.dims.size(); ++i) {
    if (low.dims[i] > high.dims[i]) return false;
  }
  return true;
}

}  // namespace

std::string shape_text(const gpu::Shape& shape) {
  return std::format("{}x{}x{}x{}", shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]);
}

std::string shapes_label(const gpu::ShapeSet& shapes) {
  const auto size = [](const gpu::Shape& s) { return std::format("{}x{}", s.dims[3], s.dims[2]); };
  const std::string low = size(shapes.min);
  const std::string high = size(shapes.max);
  return low == high ? size(shapes.opt) : low + "-" + high;
}

std::expected<void, Refusal> check_shapes(const gpu::ShapeSet& shapes) {
  if (!positive(shapes.min) || !positive(shapes.opt) || !positive(shapes.max)) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, "formes : chaque dimension doit être positive"});
  }
  if (!not_above(shapes.min, shapes.opt) || !not_above(shapes.opt, shapes.max)) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange,
                std::format("formes : min {} ≤ opt {} ≤ max {} attendu", shape_text(shapes.min),
                            shape_text(shapes.opt), shape_text(shapes.max))});
  }
  return {};
}

std::string engine_settings(Recipe recipe, std::string_view input_name,
                            const std::optional<gpu::ShapeSet>& shapes) {
  const std::string profile = shapes ? profile_flags(input_name, *shapes) : "";
  switch (recipe) {
    case Recipe::Upscaler:
      // Le préréglage « Dynamic » du Manager ajoute ses trois profils après
      // --skipInference (MainWindowViewModel.cs:1606-1610, InsertBeforeTactics) :
      // même ordre ici, pour que la clé d'un moteur dynamique du fork reste la sienne.
      if (shapes && !is_fixed(*shapes)) return "--builderOptimizationLevel=5 --skipInference" + profile;
      return "--builderOptimizationLevel=5" + profile + " --skipInference";
    case Recipe::Fp32Islands:
      return "--stronglyTyped" + profile +
             " --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw"
             " --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference";
  }
  return {};
}

SanitizedSettings sanitize_trt11(std::string_view settings) {
  SanitizedSettings result{.kept = std::string(settings)};
  std::string& s = result.kept;
  constexpr std::array<std::string_view, 5> kDropped = {"--fp16", "--bf16", "--int8", "--best",
                                                        "--buildOnly"};
  for (const std::string_view flag : kDropped) {
    std::size_t pos = 0;
    while ((pos = s.find(flag, pos)) != std::string::npos) {
      std::size_t end = pos + flag.size();
      if (end < s.size() && s[end] != ' ') {  // drapeau entier seulement : pas --fp16xyz
        pos = end;
        continue;
      }
      while (end < s.size() && s[end] == ' ') ++end;
      s.erase(pos, end - pos);
      result.removed.emplace_back(flag);
    }
  }
  constexpr std::array<std::string_view, 3> kDroppedValues = {
      "--inputIOFormats=", "--outputIOFormats=", "--tacticSources="};
  for (const std::string_view flag : kDroppedValues) {
    std::size_t pos = 0;
    while ((pos = s.find(flag)) != std::string::npos) {
      const std::size_t space = s.find(' ', pos);
      const std::size_t end = space == std::string::npos ? s.size() : space + 1;
      result.removed.push_back(s.substr(pos, (space == std::string::npos ? s.size() : space) - pos));
      s.erase(pos, end - pos);
    }
  }
  constexpr std::string_view kWorkspace = "--workspace=";
  if (const std::size_t pos = s.find(kWorkspace); pos != std::string::npos) {
    const std::size_t value_start = pos + kWorkspace.size();
    const std::size_t value_end = s.find(' ', value_start);
    const std::size_t end = value_end == std::string::npos ? s.size() : value_end;
    const std::string value = s.substr(value_start, end - value_start);
    const std::string replacement = "--memPoolSize=workspace:" + value + "M";
    result.removed.push_back(s.substr(pos, end - pos) + " → " + replacement);
    s.replace(pos, end - pos, replacement);
  }
  return result;
}

std::filesystem::path timing_cache_for(const std::filesystem::path& engine) {
  std::filesystem::path cache = engine;
  cache += ".timing.cache";
  return cache;
}

std::vector<std::string> trtexec_arguments(const TrtexecPaths& paths, std::string_view command_settings) {
  std::vector<std::string> args{"--onnx=" + utf8_path(paths.onnx), "--saveEngine=" + utf8_path(paths.engine)};
  for (auto&& word : std::views::split(command_settings, ' ')) {
    if (!word.empty()) args.emplace_back(word.begin(), word.end());
  }
  args.push_back("--timingCacheFile=" + utf8_path(paths.timing_cache));
  return args;
}

}  // namespace ajp::prepare
