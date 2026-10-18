#include "vendor_hdr/sdk_store.h"

#include <chrono>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

#include "vendor_hdr/binary_probe.h"
#include "vendor_hdr/log.h"
#include "vendor_hdr/maxine_abi_check.h"
#include "vendor_hdr/refusal.h"
#include "vendor_hdr/sdk_copy.h"
#include "vendor_hdr/sdk_layout.h"
#include "vendor_hdr/sdk_manifest.h"

namespace ajp::vendor_hdr {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kManifestName = "current.toml";
constexpr std::uintmax_t kMaxManifestBytes = std::uintmax_t{1024} * 1024;

SdkStatus with_state(SdkKind kind, SdkState state, std::string reason) {
  return SdkStatus{.kind = kind, .state = state, .reason = std::move(reason), .installed = std::nullopt};
}

std::string reimport(SdkKind kind, std::string_view cause) {
  return std::format("{} : réimporter le {} (Paramètres > Moteurs).", cause, display_name(kind));
}

std::optional<std::string> read_manifest_text(const fs::path& file) {
  std::error_code error;
  const std::uintmax_t size = fs::file_size(file, error);
  if (error || size > kMaxManifestBytes) return std::nullopt;
  std::ifstream in(file, std::ios::binary);
  if (!in) return std::nullopt;
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

Done write_atomically(const fs::path& file, const std::string& text, const std::string& nonce) {
  const fs::path temporary = file.parent_path() / std::format(".{}-{}", file.filename().string(), nonce);
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out << text;
    if (!out.flush())
      return refuse(RefusalCode::IoFailed, std::format("écriture de {} impossible", file.string()));
  }
  std::error_code error;
  fs::rename(temporary, file, error);
  if (error) {
    fs::remove(temporary, error);
    return refuse(RefusalCode::IoFailed, std::format("{} n'a pas pu être remplacé", file.string()));
  }
  return {};
}

// Nom unique d'un import : l'heure en nanosecondes, augmentée tant qu'un dossier la porte.
std::string unique_nonce(const fs::path& root) {
  auto value = static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
  std::error_code error;
  while (fs::exists(root / std::format(".import-{:x}", value), error)) ++value;
  return std::format("{:x}", value);
}

// Les copies interrompues par un arrêt brutal ne servent plus à rien.
void remove_stale_imports(const fs::path& root) {
  std::error_code error;
  for (fs::directory_iterator it(root, error), end; !error && it != end; it.increment(error)) {
    if (it->path().filename().string().starts_with(".import-")) {
      std::error_code ignored;
      fs::remove_all(it->path(), ignored);
    }
  }
}

std::vector<ManifestFile> required_sizes(SdkKind kind, const fs::path& copy) {
  std::vector<ManifestFile> files;
  for (const RequiredFile& required : layout(kind).required) {
    const std::optional<fs::path> relative = locate(copy, required);
    if (!relative) continue;
    std::error_code error;
    const std::uintmax_t bytes = fs::file_size(copy / *relative, error);
    files.push_back(ManifestFile{.path = relative->generic_string(), .bytes = error ? 0 : bytes});
  }
  return files;
}

}  // namespace

std::expected<SdkVersion, Refusal> inspect_sdk(SdkKind kind, const fs::path& folder) {
  const SdkLayout& sdk = layout(kind);
  std::error_code error;
  if (!fs::is_directory(folder, error)) {
    return refuse(RefusalCode::SdkRejected, std::format("{} n'est pas un dossier", folder.string()));
  }
  for (const RequiredFile& required : sdk.required) {
    const std::optional<fs::path> relative = locate(folder, required);
    if (!relative) {
      return refuse(RefusalCode::SdkRejected, std::format("{} manque : {}", required.path, required.missing));
    }
    if (Done valid = check_binary(folder, *relative, required.format); !valid)
      return std::unexpected(valid.error());
  }
  auto version = read_sdk_version(kind, folder);
  if (!version) return std::unexpected(version.error());
  if (version->major() != sdk.major || version->minor() != sdk.minor) {
    return refuse(RefusalCode::SdkRejected,
                  std::format("version {} : ce lecteur prend en charge le {} ({}.{})", version->text(),
                              sdk.display_name, sdk.major, sdk.minor));
  }
  if (kind == SdkKind::MaxineVfx) {
    if (Done same = check_maxine_abi(folder); !same) return std::unexpected(same.error());
  }
  return version;
}

SdkStore::SdkStore(const fs::path& data_directory, SdkKind kind)
    : root_(data_directory / "optional-sdks" / layout(kind).directory), kind_(kind) {}

SdkStatus SdkStore::status() const {
  const fs::path manifest_file = root_ / kManifestName;
  std::error_code error;
  if (!fs::exists(manifest_file, error)) {
    return with_state(
        kind_, SdkState::NotInstalled,
        std::format("{} non importé : choisir son dossier dans Paramètres > Moteurs.", display_name(kind_)));
  }
  const std::optional<std::string> text = read_manifest_text(manifest_file);
  if (!text) return with_state(kind_, SdkState::Damaged, reimport(kind_, "manifeste illisible"));
  auto manifest = parse_manifest(*text, kind_);
  if (!manifest) {
    if (manifest.error().newer_format)
      return with_state(kind_, SdkState::WrongVersion, manifest.error().reason);
    return with_state(kind_, SdkState::Damaged, reimport(kind_, manifest.error().reason));
  }
  const SdkLayout& sdk = layout(kind_);
  if (manifest->version.major() != sdk.major || manifest->version.minor() != sdk.minor) {
    return with_state(kind_, SdkState::WrongVersion,
                      reimport(kind_, std::format("SDK {} importé, ce lecteur prend en charge {}.{}",
                                                  manifest->version.text(), sdk.major, sdk.minor)));
  }
  const fs::path copy = root_ / manifest->copy;
  for (const ManifestFile& file : manifest->files) {
    const std::uintmax_t bytes = fs::file_size(copy / file.path, error);
    if (error || bytes != file.bytes) {
      return with_state(kind_, SdkState::Damaged,
                        reimport(kind_, std::format("{} a changé depuis l'import", file.path)));
    }
  }
  auto version = inspect_sdk(kind_, copy);
  if (!version) return with_state(kind_, SdkState::Damaged, reimport(kind_, version.error().detail));
  if (*version != manifest->version) {
    return with_state(kind_, SdkState::Damaged, reimport(kind_, "version changée depuis l'import"));
  }
  return SdkStatus{.kind = kind_,
                   .state = SdkState::Ready,
                   .reason = {},
                   .installed = InstalledSdk(kind_, copy, *version)};
}

std::expected<SdkStatus, Refusal> SdkStore::import(const fs::path& folder, std::stop_token stop) {
  auto version = inspect_sdk(kind_, folder);
  if (!version) return std::unexpected(version.error());

  std::error_code error;
  fs::create_directories(root_ / "versions", error);
  if (error) return refuse(RefusalCode::IoFailed, std::format("{} : {}", root_.string(), error.message()));
  remove_stale_imports(root_);

  const std::string nonce = unique_nonce(root_);
  const fs::path staging = root_ / std::format(".import-{}", nonce);
  const Manifest manifest{.kind = kind_,
                          .version = *version,
                          .copy = std::format("versions/{}-{}", version->text(), nonce),
                          .files = {}};
  auto publish = [&]() -> Done {
    if (Done copied = copy_sdk_entries(folder, staging, layout(kind_).copied, stop); !copied) return copied;
    auto copied_version = inspect_sdk(kind_, staging);
    if (!copied_version) return std::unexpected(copied_version.error());
    if (*copied_version != *version)
      return refuse(RefusalCode::IoFailed, "la copie n'a pas la version du dossier");
    if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, "import annulé");
    Manifest chosen = manifest;
    chosen.files = required_sizes(kind_, staging);
    std::error_code rename_error;
    fs::rename(staging, root_ / chosen.copy, rename_error);
    if (rename_error)
      return refuse(RefusalCode::IoFailed, std::format("{} : {}", chosen.copy, rename_error.message()));
    return write_atomically(root_ / kManifestName, format_manifest(chosen), nonce);
  };
  if (Done published = publish(); !published) {
    fs::remove_all(staging, error);
    log().warn("import du {} refusé : {}", display_name(kind_), published.error().detail);
    return std::unexpected(published.error());
  }
  log().info("{} importé depuis {} (version {})", display_name(kind_), folder.string(), version->text());
  return status();
}

}  // namespace ajp::vendor_hdr
