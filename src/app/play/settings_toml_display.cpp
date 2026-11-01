#include "app/play/settings_toml_display.h"

#include <cstdint>
#include <format>
#include <string>
#include <utility>

#include "render/grade_params.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> out_of_range(std::string_view table, std::string_view display) {
  return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, std::format("hdr.{}.{}", table, display)});
}

bool within(double v, core::Range<double> r) {
  return v >= r.lo && v <= r.hi;
}

// Un grade calibré : toutes ses valeurs, dans leurs bornes ; sinon rien.
std::optional<PlaySettings::DisplayGrade> grade_of(std::string display, const toml::table& t) {
  using namespace render::grade_limits;
  PlaySettings::DisplayGrade g{.display = std::move(display)};
  const auto gray = t["gray"].value_exact<std::int64_t>();
  const auto shadows = t["shadows"].value<double>();
  const auto colors = t["colors"].value<double>();
  const auto gray_bright = t["gray_bright"].value<double>();
  const auto shadows_bright = t["shadows_bright"].value<double>();
  const auto colors_cut = t["colors_cut"].value<double>();
  if (!gray || !shadows || !colors || !gray_bright || !shadows_bright || !colors_cut) return std::nullopt;
  if (*gray < limits::kTrueHdrGray.lo || *gray > limits::kTrueHdrGray.hi || !within(*shadows, kShadows) ||
      !within(*colors, kVibrance) || !within(*gray_bright, kMiddleGray) ||
      !within(*shadows_bright, kShadows) || !within(*colors_cut, kAdaptiveCut))
    return std::nullopt;
  g.gray = static_cast<int>(*gray);
  g.shadows = *shadows;
  g.colors = *colors;
  g.gray_bright = *gray_bright;
  g.shadows_bright = *shadows_bright;
  g.colors_cut = *colors_cut;
  return g;
}

}  // namespace

void read_display_tables(const toml::table& hdr, PlaySettings& s,
                         std::vector<std::expected<void, Refusal>>& reads) {
  if (const toml::table* grades = hdr.get_as<toml::table>("display_grades")) {
    for (const auto& [display, node] : *grades) {
      const toml::table* t = node.as_table();
      auto grade = t != nullptr ? grade_of(std::string(display.str()), *t) : std::nullopt;
      if (!grade) {
        reads.emplace_back(out_of_range("display_grades", display.str()));
        continue;
      }
      s.display_grades.push_back(std::move(*grade));
    }
  }
}

void write_display_tables(const PlaySettings& s, toml::table& hdr) {
  // Au centième : 0.4 et non 0.40000000000000002.
  const auto r2 = [](double v) { return std::round(v * 100.0) / 100.0; };
  toml::table grades;
  for (const auto& g : s.display_grades)
    grades.insert_or_assign(g.display, toml::table{{"gray", g.gray},
                                                   {"shadows", r2(g.shadows)},
                                                   {"colors", r2(g.colors)},
                                                   {"gray_bright", r2(g.gray_bright)},
                                                   {"shadows_bright", r2(g.shadows_bright)},
                                                   {"colors_cut", r2(g.colors_cut)}});
  if (!grades.empty()) hdr.insert("display_grades", std::move(grades));
}

// Écarte de root la clé (sinon la section) dont le retrait fait accepter le fichier ; faux si aucune.
bool drop_offending(toml::table& root, const std::string& detail, std::vector<std::string>* dropped,
                    const TableReader& read) {
  for (auto& [section_key, section_node] : root) {
    toml::table* table = section_node.as_table();
    if (table == nullptr) continue;
    for (auto& [key, value] : *table) {
      const std::string where(section_key.str());
      const std::string name(key.str());
      toml::table trial = root;
      if (toml::table* in_trial = trial.get_as<toml::table>(where)) in_trial->erase(name);
      if (auto tried = read(trial); !tried && tried.error().detail == detail) continue;
      if (dropped != nullptr) dropped->push_back(std::format("{}.{} ({})", where, name, detail));
      table->erase(name);
      return true;
    }
  }
  for (auto& [section_key, section_node] : root) {
    if (section_node.as_table() == nullptr) continue;
    const std::string where(section_key.str());
    toml::table trial = root;
    trial.erase(where);
    if (auto tried = read(trial); !tried && tried.error().detail == detail) continue;
    if (dropped != nullptr) dropped->push_back(std::format("[{}] ({})", where, detail));
    root.erase(where);
    return true;
  }
  return false;
}

}  // namespace ajp::app::play
