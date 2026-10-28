#include "app/play/gpu_tier.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <utility>

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;

std::unexpected<Refusal> refuse(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, std::move(detail)});
}

// Minuscules, sans « (tm) » ni « (r) » : « Intel(R) Arc(tm) A770 » devient « intel arc a770 ».
std::string normalized(std::string_view name) {
  std::string out;
  out.reserve(name.size());
  for (const char c : name) out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  for (const std::string_view mark : {"(tm)", "(r)"}) {
    for (std::size_t at = out.find(mark); at != std::string::npos; at = out.find(mark)) out.erase(at, mark.size());
  }
  return out;
}

std::optional<GpuVendor> vendor_of(std::string_view key) {
  if (key == "nvidia") return GpuVendor::Nvidia;
  if (key == "amd") return GpuVendor::Amd;
  if (key == "intel") return GpuVendor::Intel;
  if (key == "apple") return GpuVendor::Apple;
  return std::nullopt;
}

}  // namespace

std::string_view tier_label(Tier tier) {
  switch (tier) {
    case Tier::Economy: return "Économe";
    case Tier::Light: return "Léger";
    case Tier::Normal: return "Normal";
    case Tier::Fast: return "Rapide";
    case Tier::Ultra: return "Ultra";
  }
  return "Normal";
}

std::expected<TierTable, Refusal> parse_tier_table(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed) return refuse(std::string("gpu_tiers.toml mal formé : ") + std::string(parsed.error().description()));
  const toml::array* rules = parsed.table()["rules"].as_array();
  if (rules == nullptr) return refuse("gpu_tiers.toml : `rules` manque");
  TierTable table;
  for (const toml::node& node : *rules) {
    const toml::table* row = node.as_table();
    if (row == nullptr) return refuse("gpu_tiers.toml : une règle n'est pas une table");
    const auto vendor = vendor_of(row->get("vendor") != nullptr ? row->get("vendor")->value_or(std::string_view())
                                                                : std::string_view());
    if (!vendor) return refuse("gpu_tiers.toml : fabricant inconnu (nvidia, amd, intel, apple)");
    const std::int64_t tier = row->get("tier") != nullptr ? row->get("tier")->value_or(std::int64_t{-1}) : -1;
    if (tier < 0 || tier >= kTierCount) return refuse("gpu_tiers.toml : tranche hors de 0 à 4");
    const toml::array* contains = row->get_as<toml::array>("contains");
    if (contains == nullptr || contains->empty()) return refuse("gpu_tiers.toml : `contains` manque");
    TierRule rule{.vendor = *vendor, .tier = static_cast<Tier>(tier)};
    for (const toml::node& part : *contains) {
      const auto text_part = part.value<std::string>();
      if (!text_part || text_part->empty()) return refuse("gpu_tiers.toml : `contains` n'accepte que du texte non vide");
      rule.contains.push_back(normalized(*text_part));
    }
    rule.integrated = row->get("integrated") != nullptr && row->get("integrated")->value_or(false);
    table.rules.push_back(std::move(rule));
  }
  return table;
}

const TierTable& builtin_tier_table() {
  static const TierTable table = [] {
    auto parsed = parse_tier_table(kGpuTiersToml);
    return parsed ? *std::move(parsed) : TierTable{};
  }();
  return table;
}

TierEstimate estimate_tier(const TierTable& table, const GpuFacts& gpu) {
  const std::string name = normalized(gpu.name);
  for (const TierRule& rule : table.rules) {
    if (rule.vendor != gpu.vendor) continue;
    const bool all = std::ranges::all_of(rule.contains, [&](const std::string& part) { return name.contains(part); });
    if (all) return {.tier = rule.tier, .matched = true, .integrated = rule.integrated};
  }
  // Inconnue : ce que la carte sait faire. Les unités matricielles font les réseaux de neurones ; sans
  // elles, la VRAM dit la classe de la carte.
  const bool integrated = gpu.vendor == GpuVendor::Apple || (gpu.vendor != GpuVendor::Nvidia && gpu.vram_bytes <= kGiB);
  if (gpu.vram_bytes != 0 && gpu.vram_bytes < 2 * kGiB) return {.tier = Tier::Economy, .integrated = integrated};
  if (gpu.tensor_cores || gpu.wmma) {
    return {.tier = gpu.vram_bytes >= 12 * kGiB ? Tier::Fast : Tier::Normal, .integrated = integrated};
  }
  return {.tier = Tier::Light, .integrated = integrated};
}

}  // namespace ajp::app::play
