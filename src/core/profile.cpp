#include "core/profile.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <sstream>

#define TOML_EXCEPTIONS 0
#include <toml++/toml.hpp>

namespace ajp::core {
namespace {

using Result = std::expected<void, Refusal>;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Lit une clé obligatoire ou facultative d'une table, avec son chemin pour le refus.
template <class T>
std::expected<std::optional<T>, Refusal> optional_field(const toml::table& table, std::string_view key,
                                                        const std::string& where) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return std::optional<T>{};
  if constexpr (std::is_same_v<T, double>) {
    // Un fps écrit « 30 » est un entier TOML : on l'accepte comme réel.
    if (auto i = node->value_exact<std::int64_t>()) return std::optional<T>{static_cast<double>(*i)};
  }
  std::optional<T> value = node->value_exact<T>();
  if (!value) return refuse(RefusalCode::FieldType, std::format("{}.{}", where, key));
  return value;
}

template <class T>
std::expected<T, Refusal> required_field(const toml::table& table, std::string_view key,
                                         const std::string& where) {
  auto value = optional_field<T>(table, key, where);
  if (!value) return std::unexpected(value.error());
  if (!*value) return refuse(RefusalCode::FieldMissing, std::format("{}.{}", where, key));
  return **value;
}

template <class T>
Result check_range(T value, Range<T> range, const std::string& where) {
  if (!range.contains(value)) return refuse(RefusalCode::FieldOutOfRange, where);
  return {};
}

template <class E, std::size_t N>
std::expected<E, Refusal> parse_enum(std::string_view text,
                                     const std::array<std::pair<std::string_view, E>, N>& names,
                                     const std::string& where) {
  for (const auto& [name, value] : names) {
    if (name == text) return value;
  }
  return refuse(RefusalCode::FieldOutOfRange, std::format("{} = \"{}\"", where, text));
}

constexpr std::array<std::pair<std::string_view, StepKind>, 3> kStepKinds{{
    {"downscale", StepKind::Downscale},
    {"upscale", StepKind::Upscale},
    {"interpolate", StepKind::Interpolate},
}};
constexpr std::array<std::pair<std::string_view, Policy>, 3> kPolicies{
    {{"quality", Policy::Quality}, {"fast", Policy::Fast}, {"refine", Policy::Refine}}};
constexpr std::array<std::pair<std::string_view, Pacing>, 3> kPacings{
    {{"source", Pacing::Source}, {"smooth", Pacing::Smooth}, {"keep_poses", Pacing::KeepPoses}}};
constexpr std::array<std::pair<std::string_view, DownscaleFilter>, 2> kFilters{
    {{"area", DownscaleFilter::Area}, {"spline36", DownscaleFilter::Spline36}}};
constexpr std::array<std::pair<std::string_view, HdrMode>, 3> kHdrModes{
    {{"off", HdrMode::Off}, {"native", HdrMode::Native}, {"truehdr", HdrMode::TrueHdr}}};

// « 2 » ou « "5/2" ».
std::expected<Ratio, Refusal> parse_ratio(const toml::node& node, const std::string& where) {
  if (auto i = node.value_exact<std::int64_t>()) return Ratio{*i, 1};
  auto text = node.value_exact<std::string>();
  if (!text) return refuse(RefusalCode::FieldType, where);
  std::istringstream in(*text);
  Ratio ratio;
  char slash = 0;
  if (!(in >> ratio.num >> slash >> ratio.den) || slash != '/' || !in.eof()) {
    return refuse(RefusalCode::FieldType, std::format("{} = \"{}\"", where, *text));
  }
  if (!limits::kRatioTerm.contains(ratio.num) || !limits::kRatioTerm.contains(ratio.den)) {
    return refuse(RefusalCode::FieldOutOfRange, where);
  }
  return ratio;
}

std::expected<Step, Refusal> parse_step(const toml::table& table, const std::string& where) {
  Step step;
  auto kind_text = required_field<std::string>(table, "kind", where);
  if (!kind_text) return std::unexpected(kind_text.error());
  auto kind = parse_enum(*kind_text, kStepKinds, where + ".kind");
  if (!kind) return std::unexpected(kind.error());
  step.kind = *kind;

  if (step.kind == StepKind::Downscale) {
    auto filter = required_field<std::string>(table, "filter", where);
    if (!filter) return std::unexpected(filter.error());
    auto parsed = parse_enum(*filter, kFilters, where + ".filter");
    if (!parsed) return std::unexpected(parsed.error());
    step.filter = *parsed;
    auto height = required_field<std::int64_t>(table, "height", where);
    if (!height) return std::unexpected(height.error());
    if (auto r = check_range(*height, limits::kDownscaleHeight, where + ".height"); !r) {
      return std::unexpected(r.error());
    }
    step.height = static_cast<int>(*height);
    return step;
  }

  auto model = required_field<std::string>(table, "model", where);
  if (!model) return std::unexpected(model.error());
  step.model = *model;

  const toml::node* factor = table.get("factor");
  if (factor == nullptr) return refuse(RefusalCode::FieldMissing, where + ".factor");
  auto ratio = parse_ratio(*factor, where + ".factor");
  if (!ratio) return std::unexpected(ratio.error());
  step.factor = *ratio;

  if (step.kind == StepKind::Upscale) {
    if (step.factor.den != 1) return refuse(RefusalCode::FieldType, where + ".factor");
    if (auto r = check_range(step.factor.num, limits::kUpscaleFactor, where + ".factor"); !r) {
      return std::unexpected(r.error());
    }
    return step;
  }

  const double value = step.factor.value();
  if (value <= limits::kInterpFactor.lo || value > limits::kInterpFactor.hi) {
    return refuse(RefusalCode::FieldOutOfRange, where + ".factor");
  }
  auto policy = required_field<std::string>(table, "policy", where);
  if (!policy) return std::unexpected(policy.error());
  auto parsed_policy = parse_enum(*policy, kPolicies, where + ".policy");
  if (!parsed_policy) return std::unexpected(parsed_policy.error());
  step.policy = *parsed_policy;

  auto pacing = optional_field<std::string>(table, "pacing", where);
  if (!pacing) return std::unexpected(pacing.error());
  if (*pacing) {
    auto parsed_pacing = parse_enum(**pacing, kPacings, where + ".pacing");
    if (!parsed_pacing) return std::unexpected(parsed_pacing.error());
    step.pacing = *parsed_pacing;
  }
  return step;
}

std::expected<Rule, Refusal> parse_rule(const toml::table& table, const std::string& where) {
  Rule rule{.max_pixels = limits::kPixels.hi, .max_fps = limits::kFps.hi};

  for (auto [key, target] : {std::pair{"min_pixels", &rule.min_pixels}, {"max_pixels", &rule.max_pixels}}) {
    auto value = optional_field<std::int64_t>(table, key, where);
    if (!value) return std::unexpected(value.error());
    if (*value) *target = **value;
    if (auto r = check_range(*target, limits::kPixels, std::format("{}.{}", where, key)); !r) {
      return std::unexpected(r.error());
    }
  }
  for (auto [key, target] : {std::pair{"min_fps", &rule.min_fps}, {"max_fps", &rule.max_fps}}) {
    auto value = optional_field<double>(table, key, where);
    if (!value) return std::unexpected(value.error());
    if (*value) *target = **value;
    if (auto r = check_range(*target, limits::kFps, std::format("{}.{}", where, key)); !r) {
      return std::unexpected(r.error());
    }
  }

  const toml::array* steps = table["steps"].as_array();
  if (steps == nullptr) return refuse(RefusalCode::FieldMissing, where + ".steps");
  if (steps->size() > limits::kMaxSteps) return refuse(RefusalCode::TooManySteps, where + ".steps");

  double product = 1.0;
  for (std::size_t i = 0; i < steps->size(); ++i) {
    const std::string step_where = std::format("{}.steps[{}]", where, i);
    const toml::table* step_table = steps->get(i)->as_table();
    if (step_table == nullptr) return refuse(RefusalCode::FieldType, step_where);
    auto step = parse_step(*step_table, step_where);
    if (!step) return std::unexpected(step.error());
    if (step->kind != StepKind::Downscale) product *= step->factor.value();
    rule.steps.push_back(std::move(*step));
  }
  if (product > limits::kMaxFactorProduct) {
    return refuse(RefusalCode::FactorProductTooLarge, where + ".steps");
  }
  return rule;
}

std::expected<Profile, Refusal> parse_one_profile(const toml::table& table, const std::string& where) {
  Profile profile;
  auto name = optional_field<std::string>(table, "name", where);
  if (!name) return std::unexpected(name.error());
  profile.name = name->value_or("");

  auto hdr = optional_field<std::string>(table, "hdr", where);
  if (!hdr) return std::unexpected(hdr.error());
  if (*hdr) {
    auto mode = parse_enum(**hdr, kHdrModes, where + ".hdr");
    if (!mode) return std::unexpected(mode.error());
    profile.hdr = *mode;
  }

  if (const toml::node* rules_node = table.get("rule")) {
    const toml::array* rules = rules_node->as_array();
    if (rules == nullptr) return refuse(RefusalCode::FieldType, where + ".rule");
    for (std::size_t i = 0; i < rules->size(); ++i) {
      const std::string rule_where = std::format("{}.rule[{}]", where, i);
      const toml::table* rule_table = rules->get(i)->as_table();
      if (rule_table == nullptr) return refuse(RefusalCode::FieldType, rule_where);
      auto rule = parse_rule(*rule_table, rule_where);
      if (!rule) return std::unexpected(rule.error());
      profile.rules.push_back(std::move(*rule));
    }
  }
  return profile;
}

}  // namespace

std::expected<ProfileFile, Refusal> parse_profile(std::string_view toml_text) {
  toml::parse_result parsed = toml::parse(toml_text);
  if (!parsed) {
    const auto& error = parsed.error();
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("ligne {} : {}", error.source().begin.line, error.description()));
  }
  const toml::table& root = parsed.table();

  auto version = required_field<std::int64_t>(root, "version", "profil");
  if (!version) return std::unexpected(version.error());
  if (*version != kProfileVersion) {
    return refuse(RefusalCode::ProfileVersion, std::format("version = {}", *version));
  }

  ProfileFile file;
  auto start = optional_field<std::int64_t>(root, "start_profile", "profil");
  if (!start) return std::unexpected(start.error());
  const std::int64_t start_profile = start->value_or(1);
  if (auto r = check_range(start_profile, limits::kStartProfile, "start_profile"); !r) {
    return std::unexpected(r.error());
  }
  file.start_profile = static_cast<int>(start_profile);

  if (const toml::node* profiles_node = root.get("profile")) {
    const toml::array* profiles = profiles_node->as_array();
    if (profiles == nullptr) return refuse(RefusalCode::FieldType, "profile");
    for (std::size_t i = 0; i < profiles->size(); ++i) {
      const std::string where = std::format("profile[{}]", i);
      const toml::table* table = profiles->get(i)->as_table();
      if (table == nullptr) return refuse(RefusalCode::FieldType, where);

      auto slot = required_field<std::int64_t>(*table, "slot", where);
      if (!slot) return std::unexpected(slot.error());
      if (auto r = check_range(*slot, limits::kStartProfile, where + ".slot"); !r) {
        return std::unexpected(r.error());
      }
      auto& target = file.profiles[static_cast<std::size_t>(*slot - 1)];
      if (target) return refuse(RefusalCode::FieldOutOfRange, where + ".slot (déjà défini)");

      auto profile = parse_one_profile(*table, where);
      if (!profile) return std::unexpected(profile.error());
      target = std::move(*profile);
    }
  }
  const bool any_profile = std::ranges::any_of(file.profiles, [](const auto& p) { return p.has_value(); });
  if (any_profile && !file.profiles[static_cast<std::size_t>(file.start_profile - 1)]) {
    return refuse(RefusalCode::FieldOutOfRange, "start_profile (profil absent)");
  }
  return file;
}

std::expected<ProfileFile, Refusal> load_profile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return refuse(RefusalCode::ProfileUnreadable, path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parse_profile(text.str());
}

}  // namespace ajp::core
