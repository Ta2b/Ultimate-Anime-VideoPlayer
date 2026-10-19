// « ajp engines » et « ajp engines install » : l'écran « Moteurs » en mode texte.
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>

#include "app/commands.h"
#include "app/engine_text.h"
#include "app/system_report.h"
#include "core/model_index.h"

namespace ajp::app {
namespace {

namespace fs = std::filesystem;

// Texte d'un index livré ; std::nullopt s'il n'est pas là (construction sans index).
std::expected<std::optional<std::string>, Refusal> read_index(const fs::path& path) {
  std::error_code error;
  if (!fs::exists(path, error)) return std::nullopt;
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(Refusal{RefusalCode::FileUnreadable, utf8(path)});
  std::ostringstream text;
  text << in.rdbuf();
  if (!in && !in.eof()) return std::unexpected(Refusal{RefusalCode::FileUnreadable, utf8(path)});
  return text.str();
}

template <class Entry>
std::expected<std::vector<Entry>, Refusal> load_index(
    const fs::path& path, std::expected<std::vector<Entry>, Refusal> (*parse)(std::string_view)) {
  auto text = read_index(path);
  if (!text) return std::unexpected(text.error());
  if (!*text) return std::vector<Entry>{};
  auto entries = parse(**text);
  if (!entries) {
    return std::unexpected(
        Refusal{entries.error().code, std::format("{} : {}", utf8(path), entries.error().detail)});
  }
  return entries;
}

std::string badges_text(const core::EngineOffer& offer) {
  std::string text;
  for (const core::Badge badge : offer.badges)
    text += std::format("{}{}", text.empty() ? "" : ", ", badge_label(badge));
  return text;
}

// État d'une ligne, après son nom et ses badges.
std::string state_text(const EngineChooser& chooser, const EngineRow& row) {
  if (row.offer.delivery == core::Delivery::Bundled) return "livré avec le lecteur";
  if (row.installed) return "installé";
  if (row.offer.delivery == core::Delivery::Import) return "à importer : ajp import-sdk <dossier du SDK>";
  if (!row.package) return "aucun paquet dans l'index pour ce système";
  if (chooser.lock_of(row.offer.id) == RowLock::Advanced) {
    return std::format("{}, installable en mode avancé", size_text(row.package->size));
  }
  return std::format("à télécharger, {}", size_text(row.package->size));
}

void print_rows(const EngineChooser& chooser, std::ostream& out) {
  out << (chooser.first_run() ? "\nMoteurs (présélection pour ce GPU) :\n"
                              : "\nMoteurs (choix enregistré) :\n");
  for (const EngineRow& row : chooser.rows()) {
    out << std::format("  [{}] {:<22} {}\n", row.checked ? 'x' : ' ', engine_title(row.offer.id),
                       badges_text(row.offer));
    out << std::format("      {} ; {} ; {}\n", engine_summary(row.offer.id), reason_text(row.offer.reason),
                       state_text(chooser, row));
  }
  out << std::format("\nÀ télécharger (modèles compris) : {} ; place libre : {}\n",
                     size_text(chooser.download_bytes()), size_text(chooser.free_bytes()));
}

// Une ligne par élément commencé, puis tous les 10 % du total.
std::function<void(const InstallProgress&)> progress_printer(std::ostream& out) {
  return [&out, item = std::optional<std::size_t>{},
          tenth = std::uint64_t{0}](const InstallProgress& p) mutable {
    if (item != p.item) {
      item = p.item;
      out << std::format("[{}/{}] {}\n", p.item + 1, p.items, p.label);
    }
    const std::uint64_t now = p.total == 0 ? 10 : p.done * 10 / p.total;
    if (now > tenth) {
      tenth = now;
      out << std::format("      {} % ({} sur {})\n", now * 10, size_text(p.done), size_text(p.total));
    }
  };
}

}  // namespace

std::expected<EngineChooser, Refusal> open_chooser(const AppContext& context,
                                                   const core::SystemInfo& system) {
  auto settings = core::load_settings(context.paths.settings());
  if (!settings) return std::unexpected(settings.error());
  auto packages = load_index(context.paths.engines_lock(), &core::parse_engine_index);
  if (!packages) return std::unexpected(packages.error());
  auto models = load_index(context.paths.models_lock(), &core::parse_models_lock);
  if (!models) return std::unexpected(models.error());
  std::error_code error;
  const fs::space_info space = fs::space(context.paths.user_data, error);
  if (error) {
    return std::unexpected(Refusal{RefusalCode::FileUnreadable,
                                   std::format("{} : {}", utf8(context.paths.user_data), error.message())});
  }
  return EngineChooser(
      ChooserInput{.os = context.os,
                   .system = system,
                   .settings = std::move(*settings),
                   .packages = std::move(*packages),
                   .missing_models = missing_models(context.paths.layout(context.os), *models),
                   .free_bytes = space.available});
}

std::expected<core::Settings, Refusal> apply_plan(
    const ChooserPlan& plan, const core::SystemInfo& system, const AppContext& context,
    const Downloader& downloader, const std::function<void(const InstallProgress&)>& on_progress) {
  const core::InstallLayout layout = context.paths.layout(context.os);
  for (const core::EngineId engine : plan.removals) {
    if (auto r = uninstall_engine(layout, engine); !r) return std::unexpected(r.error());
  }
  std::vector<InstallItem> items;
  items.reserve(plan.downloads.size() + plan.models.size());
  for (const core::EnginePackage& package : plan.downloads) items.push_back(engine_item(layout, package));
  for (const core::ModelEntry& model : plan.models) items.push_back(model_item(layout, model));
  const auto installed = install_all(downloader, layout, items, context.download, on_progress, context.stop);

  // Ce qui est en place est noté, même après un refus : rien ne se retélécharge pour rien.
  core::Settings settings = plan.settings;
  for (const InstallItem& item : items) {
    if (item.engine && is_installed(item))
      settings = record_installed(std::move(settings), system, *item.engine);
  }
  if (auto saved = core::save_settings(context.paths.settings(), settings); !saved) {
    return std::unexpected(saved.error());
  }
  if (!installed) return std::unexpected(installed.error());
  return settings;
}

std::expected<void, Refusal> list_engines(const AppContext& context) {
  const SystemReport report = detect_system(*context.loader, context.os);
  for (const std::string& line : describe(report)) *context.out << line << '\n';
  auto chooser = open_chooser(context, to_system_info(report));
  if (!chooser) return std::unexpected(chooser.error());
  print_rows(*chooser, *context.out);
  *context.out << "Pour installer cette sélection : ajp engines install [moteur…]\n";
  return {};
}

std::expected<void, Refusal> install_engines(const EnginesArgs& args, const AppContext& context) {
  const core::SystemInfo system = to_system_info(detect_system(*context.loader, context.os));
  auto chooser = open_chooser(context, system);
  if (!chooser) return std::unexpected(chooser.error());
  chooser->set_advanced(!args.engines.empty());  // un moteur nommé s'installe même grisé
  for (const core::EngineId engine : args.engines) {
    if (auto r = chooser->set_checked(engine, true); !r) return r;
  }
  print_rows(*chooser, *context.out);
  auto plan = chooser->confirm();
  if (!plan) return std::unexpected(plan.error());
  if (plan->downloads.empty() && plan->models.empty() && plan->removals.empty()) {
    *context.out << "Rien à télécharger : tout est déjà installé.\n";
  }
  auto downloader = Downloader::create();
  if (!downloader) return std::unexpected(downloader.error());
  auto settings = apply_plan(*plan, system, context, *downloader, progress_printer(*context.out));
  if (!settings) return std::unexpected(settings.error());
  for (const core::EngineId engine : plan->imports) {
    *context.out << std::format("{} : téléchargez le SDK chez NVIDIA, puis « ajp import-sdk <dossier> ».\n",
                                engine_title(engine));
  }
  *context.out << "Réglages enregistrés dans " << utf8(context.paths.settings()) << ".\n";
  return {};
}

}  // namespace ajp::app
