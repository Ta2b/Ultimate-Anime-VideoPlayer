#include "gpu/cuda/backend/fused_keys.h"

#include <charconv>
#include <cstddef>
#include <format>
#include <system_error>

namespace ajp::gpu::cuda {
namespace {

constexpr std::string_view kRifePrefix = "rife-fused:";
constexpr std::string_view kDrbaPrefix = "drba-fused:";

std::optional<int> parse_int(std::string_view text) {
  int value = 0;
  const auto [stop, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || stop != text.data() + text.size()) return std::nullopt;
  return value;
}

std::vector<std::string_view> split(std::string_view text, char separator) {
  std::vector<std::string_view> parts;
  while (true) {
    const std::size_t at = text.find(separator);
    parts.push_back(text.substr(0, at));
    if (at == std::string_view::npos) return parts;
    text.remove_prefix(at + 1);
  }
}

// Un nom de fichier seul : la clé ne sort jamais du dossier du cache.
bool plain_file_name(std::string_view part) {
  return !part.empty() && part.find_first_of("/\\") == std::string_view::npos && part != "." && part != "..";
}

}  // namespace

std::string rife_fused_artifact_key(const RifeFusedKey& key) {
  std::string reduce;
  for (int b = 0; b < key.plan.blocks; ++b)
    reduce += std::format("{}{}", b == 0 ? "" : ",", key.plan.reduce[static_cast<std::size_t>(b)]);
  std::string engines;
  for (const std::string& engine : key.engines)
    engines += std::format("{}{}", engines.empty() ? "" : ",", engine);
  return std::format("{}{}x{}:{}:{}:{}", kRifePrefix, key.height, key.width, key.motion ? "motion" : "image",
                     reduce, engines);
}

std::optional<RifeFusedKey> rife_fused_artifact_key_parse(std::string_view text) {
  if (!text.starts_with(kRifePrefix)) return std::nullopt;
  text.remove_prefix(kRifePrefix.size());
  const std::vector<std::string_view> fields = split(text, ':');
  if (fields.size() != 4) return std::nullopt;
  const std::vector<std::string_view> size = split(fields[0], 'x');
  if (size.size() != 2) return std::nullopt;
  const auto height = parse_int(size[0]);
  const auto width = parse_int(size[1]);
  if (!height || !width || (fields[1] != "image" && fields[1] != "motion")) return std::nullopt;
  RifeFusedKey key{.height = *height, .width = *width, .motion = fields[1] == "motion"};
  for (const std::string_view part : split(fields[2], ',')) {
    const auto d = parse_int(part);
    if (!d || key.plan.blocks >= kernels::kFusedMaxBlocks) return std::nullopt;
    key.plan.reduce[static_cast<std::size_t>(key.plan.blocks++)] = *d;
  }
  for (const std::string_view part : split(fields[3], ',')) {
    if (!plain_file_name(part)) return std::nullopt;
    key.engines.emplace_back(part);
  }
  if (!kernels::rife_fused_size_valid(key.plan, key.height, key.width) ||
      key.engines.size() != static_cast<std::size_t>(key.plan.blocks) + 1)
    return std::nullopt;
  return key;
}

std::string drba_fused_artifact_key(const DrbaFusedKey& key) {
  std::string engines;
  for (const std::string& engine : key.engines)
    engines += std::format("{}{}", engines.empty() ? "" : ",", engine);
  return std::format("{}{}x{}:{}:{}", kDrbaPrefix, key.height, key.width, key.plan.tail, engines);
}

std::optional<DrbaFusedKey> drba_fused_artifact_key_parse(std::string_view text) {
  if (!text.starts_with(kDrbaPrefix)) return std::nullopt;
  text.remove_prefix(kDrbaPrefix.size());
  const std::vector<std::string_view> fields = split(text, ':');
  if (fields.size() != 3) return std::nullopt;
  const std::vector<std::string_view> size = split(fields[0], 'x');
  if (size.size() != 2) return std::nullopt;
  const auto height = parse_int(size[0]);
  const auto width = parse_int(size[1]);
  const auto tail = parse_int(fields[1]);
  if (!height || !width || !tail) return std::nullopt;
  DrbaFusedKey key{.height = *height, .width = *width, .plan = {.tail = *tail}};
  for (const std::string_view part : split(fields[2], ',')) {
    if (!plain_file_name(part)) return std::nullopt;
    key.engines.emplace_back(part);
  }
  if (!kernels::drba_fused_size_valid(key.plan, key.height, key.width) ||
      key.engines.size() != static_cast<std::size_t>(kernels::kDrbaFusedBlocks) + 1)
    return std::nullopt;
  return key;
}

}  // namespace ajp::gpu::cuda
