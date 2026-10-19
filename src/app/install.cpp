#include "app/install.h"

#include <format>
#include <system_error>
#include <utility>

#include "app/data_paths.h"

namespace ajp::app {
namespace {

namespace fs = std::filesystem;

std::unexpected<Refusal> unwritable(const fs::path& path, const std::error_code& error) {
  return std::unexpected(
      Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", utf8(path), error.message())});
}

// Un modèle : un renommage du .part vers son nom final, qui remplace un ancien fichier.
std::expected<void, Refusal> place_file(const fs::path& part, const fs::path& target) {
  std::error_code error;
  fs::create_directories(target.parent_path(), error);
  if (error) return unwritable(target.parent_path(), error);
  fs::rename(part, target, error);
  if (error) return unwritable(target, error);
  return {};
}

// Un moteur : son dossier de version est préparé à côté, puis renommé d'un coup. Un dossier de
// version présent mais incomplet (is_installed faux) est remplacé.
std::expected<void, Refusal> place_engine(const fs::path& part, const fs::path& target) {
  const fs::path version_dir = target.parent_path();
  fs::path staging = version_dir;
  staging += ".staging";
  std::error_code error;
  fs::remove_all(staging, error);
  if (error) return unwritable(staging, error);
  if (auto r = place_file(part, staging / target.filename()); !r) return r;
  fs::remove_all(version_dir, error);
  if (error) return unwritable(version_dir, error);
  fs::rename(staging, version_dir, error);
  if (error) return unwritable(version_dir, error);
  return {};
}

}  // namespace

InstallItem model_item(const core::InstallLayout& layout, const core::ModelEntry& model) {
  return InstallItem{.label = std::format("modèle {}", model.id),
                     .url = model.url,
                     .sha256 = model.sha256,
                     .size = model.size,
                     .target = core::model_path(layout, model)};
}

InstallItem engine_item(const core::InstallLayout& layout, const core::EnginePackage& package) {
  return InstallItem{.label = std::format("moteur {} {}", core::engine_name(package.engine), package.version),
                     .url = package.url,
                     .sha256 = package.sha256,
                     .size = package.size,
                     .target = core::backend_library(layout, package),
                     .engine = package.engine};
}

bool is_installed(const InstallItem& item) {
  std::error_code error;
  const std::uintmax_t size = fs::file_size(item.target, error);
  return !error && size == item.size;
}

std::vector<core::ModelEntry> missing_models(const core::InstallLayout& layout,
                                             std::span<const core::ModelEntry> lock) {
  std::vector<core::ModelEntry> missing;
  for (const core::ModelEntry& model : lock) {
    if (!model.profiles.empty() && !is_installed(model_item(layout, model))) missing.push_back(model);
  }
  return missing;
}

std::expected<void, Refusal> install_downloaded(const core::InstallLayout& layout, const InstallItem& item) {
  const fs::path part = core::download_path(layout, item.sha256);
  if (auto verified = core::verify_file(part, item.size, item.sha256); !verified) {
    std::error_code ignored;
    fs::remove(part, ignored);  // un .part faux ne se reprend pas : le prochain essai repart de zéro
    return std::unexpected(
        Refusal{verified.error().code, std::format("{} : {}", item.label, verified.error().detail)});
  }
  return item.engine ? place_engine(part, item.target) : place_file(part, item.target);
}

std::expected<void, Refusal> uninstall_engine(const core::InstallLayout& layout, core::EngineId engine) {
  const fs::path folder = layout.engines / std::string(core::engine_name(engine));
  std::error_code error;
  fs::remove_all(folder, error);
  if (error) return unwritable(folder, error);
  return {};
}

std::expected<void, Refusal> install_all(const Downloader& downloader, const core::InstallLayout& layout,
                                         std::span<const InstallItem> items, DownloadOptions options,
                                         const std::function<void(const InstallProgress&)>& on_progress,
                                         const std::stop_token& stop) {
  std::uint64_t total = 0;
  for (const InstallItem& item : items) total += is_installed(item) ? 0 : item.size;

  std::uint64_t done = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const InstallItem& item = items[i];
    if (is_installed(item)) continue;
    if (on_progress) {
      options.on_progress = [&, i](const DownloadProgress& progress) {
        on_progress(InstallProgress{.item = i,
                                    .items = items.size(),
                                    .label = item.label,
                                    .done = done + progress.received,
                                    .total = total});
      };
    }
    const DownloadRequest request{
        .url = item.url, .part_file = core::download_path(layout, item.sha256), .size = item.size};
    if (auto r = downloader.fetch(request, options, stop); !r) {
      return std::unexpected(Refusal{r.error().code, std::format("{} : {}", item.label, r.error().detail)});
    }
    if (auto r = install_downloaded(layout, item); !r) return r;
    done += item.size;
  }
  return {};
}

}  // namespace ajp::app
