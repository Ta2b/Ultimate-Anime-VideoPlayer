#include "app/engine_chooser.h"

#include <algorithm>
#include <format>
#include <utility>

#include "app/engine_text.h"

namespace ajp::app {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::optional<core::EnginePackage> package_for(const ChooserInput& input, core::EngineId engine) {
  const core::EnginePackage* package = core::find_package(input.packages, engine, input.os);
  if (package == nullptr) return std::nullopt;
  return *package;
}

bool bundled(const EngineRow& row) {
  return row.offer.delivery == core::Delivery::Bundled;
}

}  // namespace

EngineChooser::EngineChooser(ChooserInput input) : input_(std::move(input)) {
  const core::Settings& settings = input_.settings;
  for (const core::EngineOffer& offer : core::recommend_engines(input_.system)) {
    EngineRow row{.offer = offer,
                  .installed = std::ranges::contains(settings.installed_engines, offer.id),
                  .package = package_for(input_, offer.id)};
    const bool wanted = settings.first_run_done ? std::ranges::contains(settings.chosen_engines, offer.id)
                                                : offer.preselected || row.installed;
    row.checked = bundled(row) || (wanted && (row.installed || can_check(row)));
    rows_.push_back(std::move(row));
  }
}

const EngineRow* EngineChooser::row(core::EngineId engine) const {
  const auto found = std::ranges::find(rows_, engine, [](const EngineRow& r) { return r.offer.id; });
  return found == rows_.end() ? nullptr : &*found;
}

EngineRow* EngineChooser::find(core::EngineId engine) {
  const auto found = std::ranges::find(rows_, engine, [](const EngineRow& r) { return r.offer.id; });
  return found == rows_.end() ? nullptr : &*found;
}

bool EngineChooser::can_check(const EngineRow& row) const {
  if (!row.offer.relevant && !advanced_) return false;
  return row.offer.delivery != core::Delivery::Download || row.package.has_value();
}

RowLock EngineChooser::lock_of(core::EngineId engine) const {
  const EngineRow* found = row(engine);
  if (found == nullptr) return RowLock::NoPackage;
  if (bundled(*found)) return RowLock::Bundled;
  if (found->installed) return RowLock::Free;
  if (!found->offer.relevant && !advanced_) return RowLock::Advanced;
  if (found->offer.delivery == core::Delivery::Download && !found->package) return RowLock::NoPackage;
  return RowLock::Free;
}

void EngineChooser::set_advanced(bool on) {
  advanced_ = on;
  if (on) return;
  for (EngineRow& r : rows_) {
    if (!r.offer.relevant && !r.installed) r.checked = false;
  }
}

void EngineChooser::reset_to_recommended() {
  for (EngineRow& r : rows_) r.checked = bundled(r) || (r.offer.preselected && (r.installed || can_check(r)));
}

std::expected<void, Refusal> EngineChooser::set_checked(core::EngineId engine, bool checked) {
  EngineRow* found = find(engine);
  const std::string_view title = engine_title(engine);
  if (found == nullptr) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("{} n'existe pas sur {}", title, core::os_name(input_.os)));
  }
  if (checked == found->checked) return {};
  switch (lock_of(engine)) {
    case RowLock::Bundled:
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("{} est livré avec le lecteur et reste inclus", title));
    case RowLock::Advanced:
      if (checked) {
        return refuse(RefusalCode::FieldOutOfRange,
                      std::format("{} : {} ; cochez « Mode avancé » pour l'installer quand même", title,
                                  reason_text(found->offer.reason)));
      }
      break;
    case RowLock::NoPackage:
      if (checked) {
        return refuse(RefusalCode::FieldOutOfRange,
                      std::format("{} : aucun paquet pour {} dans l'index des moteurs", title,
                                  core::os_name(input_.os)));
      }
      break;
    case RowLock::Free:
      break;
  }
  found->checked = checked;
  return {};
}

std::expected<void, Refusal> EngineChooser::toggle(core::EngineId engine) {
  const EngineRow* found = row(engine);
  return set_checked(engine, found == nullptr || !found->checked);
}

std::uint64_t EngineChooser::download_bytes() const {
  std::uint64_t total = 0;
  for (const EngineRow& r : rows_) {
    if (r.checked && !r.installed && r.package) total += r.package->size;
  }
  for (const core::ModelEntry& model : input_.missing_models) total += model.size;
  return total;
}

std::expected<ChooserPlan, Refusal> EngineChooser::confirm() const {
  ChooserPlan plan{
      .settings = input_.settings, .models = input_.missing_models, .download_bytes = download_bytes()};
  if (!fits()) {
    return refuse(RefusalCode::DiskFull,
                  std::format("{} à télécharger, {} libres dans le dossier des données",
                              size_text(plan.download_bytes), size_text(free_bytes())));
  }
  core::Settings& settings = plan.settings;
  settings.first_run_done = true;
  settings.chosen_engines.clear();
  for (const EngineRow& r : rows_) {
    if (r.checked) settings.chosen_engines.push_back(r.offer.id);
    if (r.checked && !r.installed && r.package) plan.downloads.push_back(*r.package);
    if (r.checked && !r.installed && r.offer.delivery == core::Delivery::Import)
      plan.imports.push_back(r.offer.id);
    if (!r.checked && r.installed) plan.removals.push_back(r.offer.id);
  }
  std::erase_if(settings.installed_engines,
                [&](core::EngineId id) { return std::ranges::contains(plan.removals, id); });
  for (std::size_t usage = 0; usage < core::kUsageCount; ++usage) {
    std::optional<core::EngineId>& preferred = settings.preferred[usage];
    if (!preferred || !std::ranges::contains(plan.removals, *preferred)) continue;
    preferred = static_cast<core::Usage>(usage) == core::Usage::TrueHdr
                    ? std::nullopt
                    : std::optional(core::EngineId::Vulkan);
  }
  if (auto valid = core::validate_settings(settings); !valid) return std::unexpected(valid.error());
  return plan;
}

core::Settings record_installed(core::Settings settings, const core::SystemInfo& system,
                                core::EngineId engine) {
  if (!std::ranges::contains(settings.installed_engines, engine))
    settings.installed_engines.push_back(engine);
  if (!std::ranges::contains(settings.chosen_engines, engine)) settings.chosen_engines.push_back(engine);
  const std::vector<core::EngineOffer> offers = core::recommend_engines(system);
  const auto offer = std::ranges::find(offers, engine, &core::EngineOffer::id);
  if (offer == offers.end()) return settings;
  auto& preferred = settings.preferred;
  if (offer->reason == core::Reason::FastestForThisGpu) {
    for (const core::Usage usage : {core::Usage::Upscale, core::Usage::Interpolate}) {
      std::optional<core::EngineId>& slot = preferred[static_cast<std::size_t>(usage)];
      if (!slot || *slot == core::EngineId::Vulkan) slot = engine;
    }
  } else if (offer->reason == core::Reason::RtxHdrForThisGpu) {
    std::optional<core::EngineId>& slot = preferred[static_cast<std::size_t>(core::Usage::TrueHdr)];
    if (!slot) slot = engine;
  }
  return settings;
}

}  // namespace ajp::app
