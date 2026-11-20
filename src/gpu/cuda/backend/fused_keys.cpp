#include "gpu/cuda/backend/fused_keys.h"

#include <charconv>
#include <cstddef>
#include <format>
#include <system_error>

namespace kirari::gpu::cuda {
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
  const std::string_view output =
      key.motion ? "motion" : (key.warp == kernels::RifeWarp::CatmullRom ? "image-cubic" : "image");
  std::string text =
      std::format("{}{}x{}:{}:{}:{}", kRifePrefix, key.height, key.width, output, reduce, engines);
  if (!key.encode_body.empty()) text += std::format(":{}", key.encode_body);
  return text;
}

std::optional<RifeFusedKey> rife_fused_artifact_key_parse(std::string_view text) {
  if (!text.starts_with(kRifePrefix)) return std::nullopt;
  text.remove_prefix(kRifePrefix.size());
  const std::vector<std::string_view> fields = split(text, ':');
  if (fields.size() != 4 && fields.size() != 5) return std::nullopt;
  const std::vector<std::string_view> size = split(fields[0], 'x');
  if (size.size() != 2) return std::nullopt;
  const auto height = parse_int(size[0]);
  const auto width = parse_int(size[1]);
  if (!height || !width || (fields[1] != "image" && fields[1] != "motion" && fields[1] != "image-cubic"))
    return std::nullopt;
  RifeFusedKey key{
      .height = *height,
      .width = *width,
      .motion = fields[1] == "motion",
      .warp = fields[1] == "image-cubic" ? kernels::RifeWarp::CatmullRom : kernels::RifeWarp::Bilinear};
  for (const std::string_view part : split(fields[2], ',')) {
    const auto d = parse_int(part);
    if (!d || key.plan.blocks >= kernels::kFusedMaxBlocks) return std::nullopt;
    key.plan.reduce[static_cast<std::size_t>(key.plan.blocks++)] = *d;
  }
  for (const std::string_view part : split(fields[3], ',')) {
    if (!plain_file_name(part)) return std::nullopt;
    key.engines.emplace_back(part);
  }
  if (fields.size() == 5) {
    if (!plain_file_name(fields[4]) || !key.motion || key.height != 1088 || key.width != 1920)
      return std::nullopt;
    key.encode_body = fields[4];
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

constexpr std::string_view kTfixPrefix = "tfix-fused:";

std::string tfix_fused_artifact_key(const TfixFusedKey& key) {
  std::string engines;
  for (const std::string& engine : key.engines)
    engines += std::format("{}{}", engines.empty() ? "" : ",", engine);
  return std::format("{}{}:{}x{}:{}:{}", kTfixPrefix, key.window ? "window" : "encoder", key.height,
                     key.width, key.half ? 1 : 0, engines);
}

std::optional<TfixFusedKey> tfix_fused_artifact_key_parse(std::string_view text) {
  if (!text.starts_with(kTfixPrefix)) return std::nullopt;
  text.remove_prefix(kTfixPrefix.size());
  const std::vector<std::string_view> fields = split(text, ':');
  if (fields.size() != 4 || (fields[0] != "window" && fields[0] != "encoder") ||
      (fields[2] != "0" && fields[2] != "1"))
    return std::nullopt;
  const std::vector<std::string_view> size = split(fields[1], 'x');
  if (size.size() != 2) return std::nullopt;
  const auto height = parse_int(size[0]);
  const auto width = parse_int(size[1]);
  if (!height || !width) return std::nullopt;
  TfixFusedKey key{
      .window = fields[0] == "window", .height = *height, .width = *width, .half = fields[2] == "1"};
  for (const std::string_view part : split(fields[3], ',')) {
    if (!plain_file_name(part)) return std::nullopt;
    key.engines.emplace_back(part);
  }
  if (!kernels::tfix_size_valid(key.height, key.width) || key.engines.size() != (key.window ? 4U : 1U))
    return std::nullopt;
  return key;
}

namespace {

constexpr std::string_view kGmfssPrefix = "gmfss-fused:";
constexpr std::string_view kGmfssPairPrefix = "gmfss-pair:";

std::string gmfss_key(std::string_view prefix, const GmfssFusedKey& key) {
  std::string engines;
  for (const std::string& engine : key.engines)
    engines += std::format("{}{}", engines.empty() ? "" : ",", engine);
  return std::format("{}{}x{}:{}", prefix, key.height, key.width, engines);
}

std::optional<GmfssFusedKey> gmfss_key_parse(std::string_view prefix, std::string_view text) {
  if (!text.starts_with(prefix)) return std::nullopt;
  text.remove_prefix(prefix.size());
  const std::vector<std::string_view> fields = split(text, ':');
  if (fields.size() != 2) return std::nullopt;
  const std::vector<std::string_view> size = split(fields[0], 'x');
  if (size.size() != 2) return std::nullopt;
  const auto height = parse_int(size[0]);
  const auto width = parse_int(size[1]);
  if (!height || !width || *height < 32 || *width < 32 || *height % 32 != 0 || *width % 32 != 0)
    return std::nullopt;
  GmfssFusedKey key{.height = *height, .width = *width};
  for (const std::string_view part : split(fields[1], ',')) {
    if (!plain_file_name(part)) return std::nullopt;
    key.engines.emplace_back(part);
  }
  if (key.engines.size() != 2) return std::nullopt;
  return key;
}

}  // namespace

std::string gmfss_fused_artifact_key(const GmfssFusedKey& key) {
  return gmfss_key(kGmfssPrefix, key);
}

std::optional<GmfssFusedKey> gmfss_fused_artifact_key_parse(std::string_view text) {
  return gmfss_key_parse(kGmfssPrefix, text);
}

std::string gmfss_pair_artifact_key(const GmfssFusedKey& key) {
  return gmfss_key(kGmfssPairPrefix, key);
}

std::optional<GmfssFusedKey> gmfss_pair_artifact_key_parse(std::string_view text) {
  return gmfss_key_parse(kGmfssPairPrefix, text);
}

}  // namespace kirari::gpu::cuda
