#include "app/play/updater.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>  // mkdtemp (macOS : pas dans <cstdlib>)
#endif

#include "app/play/ed25519.h"
#include "app/play/engine_tools.h"
#include "app/play/i18n.h"
#include "core/dir_entries.h"
#include "core/log.h"
#include "core/model_index.h"
#include "core/user_overlay.h"
#include "prepare/process.h"
#include "prepare/utf8.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

#ifndef AJP_RELEASE_REPO
#define AJP_RELEASE_REPO "Ta2b/ajp-releases"
#endif
#ifndef AJP_UPDATE_PUBKEY
#define AJP_UPDATE_PUBKEY ""
#endif
#ifndef AJP_VERSION
#define AJP_VERSION "0.0.0"
#endif

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string read_all(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

std::string trimmed(std::string text) {
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
  while (!text.empty() && (text.front() == '\n' || text.front() == ' ')) text.erase(text.begin());
  return text;
}

// Un dossier neuf, à nous seuls, pour le manifeste : mkdtemp (0700, nom imprévisible) hors Windows ; sous
// Windows, le dossier temporaire est celui de l'utilisateur, et un nom tiré au hasard doit ne pas exister. Un
// dossier préparé d'avance (lien symbolique que curl -o suivrait) n'est jamais repris.
std::expected<fs::path, Refusal> private_temp_dir() {
  std::error_code error;
  const fs::path base = fs::temp_directory_path(error);
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("dossier temporaire : {}", error.message()));
#if defined(_WIN32)
  std::random_device random;
  for (int attempt = 0; attempt < 8; ++attempt) {
    const fs::path dir = base / std::format("ajp-update-{:08x}{:08x}", random(), random());
    if (fs::create_directory(dir, error)) return dir;
  }
  return refuse(RefusalCode::FileUnwritable, "dossier temporaire : aucun nom libre");
#else
  std::string pattern = prepare::utf8_path(base / "ajp-update-XXXXXX");
  if (mkdtemp(pattern.data()) == nullptr)
    return refuse(RefusalCode::FileUnwritable, std::format("mkdtemp : {}", std::strerror(errno)));
  return fs::path(pattern);
#endif
}

// curl du système : https seulement, délai borné. Un fichier absent du serveur (HTTP 404) est refusé
// TrackUnavailable, que check_for_update distingue d'une panne.
std::expected<void, Refusal> fetch(std::string_view url, const fs::path& file, int max_seconds,
                                   const std::stop_token& stop) {
  const auto curl = system_tool("curl");
  if (!curl)
    return refuse(RefusalCode::LibraryUnavailable, i18n::tr("curl not found: cannot check for updates"));
  prepare::ProcessSpec spec{
      .program = *curl,
      .args = {"-fsSL", "--proto", "=https", "--max-time", std::to_string(max_seconds), "--write-out",
               std::string(kCurlStatusFormat), "-o", prepare::utf8_path(file), std::string(url)},
      .timeout = std::chrono::seconds(max_seconds + 10)};
  int status = 0;
  auto exit = prepare::run_process(
      spec,
      [&](std::string_view line) {
        if (const auto read = curl_status_line(line)) status = *read;
      },
      stop);
  if (!exit) return std::unexpected(exit.error());
  if (exit->kind == prepare::ExitKind::Cancelled)
    return refuse(RefusalCode::Cancelled, i18n::tr("download cancelled"));
  if (exit->kind != prepare::ExitKind::Exited)
    return refuse(RefusalCode::DecodeFailed,
                  std::format("{} : {}", i18n::tr("download"), prepare::describe(*exit)));
  if (exit->code == 0) return {};
  return std::unexpected(curl_failure(static_cast<int>(exit->code), status));
}

// Les fichiers (et liens) d'un dossier, en chemins relatifs. Une erreur du parcours est une erreur : une
// liste incomplète supprimerait ou garderait les mauvais fichiers.
std::expected<std::vector<fs::path>, Refusal> tree_files(const fs::path& root) {
  std::vector<fs::path> files;
  std::error_code error;
  fs::recursive_directory_iterator it(root, fs::directory_options::none, error);
  const auto failed = [&](const fs::path& where) {
    return refuse(RefusalCode::FileUnreadable,
                  std::format("{} : {}", prepare::utf8_path(where), error.message()));
  };
  if (error) return failed(root);
  for (; it != fs::recursive_directory_iterator(); it.increment(error)) {
    if (error) return failed(root);
    if (it->is_directory(error) && !it->is_symlink(error)) continue;
    fs::path relative = fs::relative(it->path(), root, error);
    if (error || relative.empty()) return failed(it->path());
    files.push_back(std::move(relative));
  }
  if (error) return failed(root);
  std::ranges::sort(files);
  return files;
}

// Retire un fichier installé ; en service sous Windows, il est renommé en « .old » (guard_new_version le
// purge).
std::expected<void, Refusal> remove_installed(const fs::path& target) {
  std::error_code error;
  if (!fs::exists(target, error) && !fs::is_symlink(target, error)) return {};
  fs::remove(target, error);
  if (!error) return {};
  error.clear();
  fs::rename(target, fs::path(target.string() + ".old"), error);
  if (error)
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} : {}", prepare::utf8_path(target), error.message()));
  return {};
}

// Copie récursive qui remplace les fichiers même en service : le fichier est retiré (ou, sous Windows,
// renommé en « .old ») avant d'être recréé, ce que l'ouverture en écriture d'un exécutable chargé refuserait.
std::expected<void, Refusal> replace_tree(const fs::path& from, const fs::path& to) {
  const auto files = tree_files(from);
  if (!files) return std::unexpected(files.error());
  std::error_code error;
  for (const fs::path& relative : *files) {
    const fs::path source = from / relative;
    const fs::path target = to / relative;
    fs::create_directories(target.parent_path(), error);
    if (error)
      return refuse(RefusalCode::FileUnwritable,
                    std::format("{} : {}", prepare::utf8_path(target.parent_path()), error.message()));
    if (auto removed = remove_installed(target); !removed) return removed;
    if (fs::is_symlink(source, error)) {
      fs::copy_symlink(source, target, error);
    } else {
      fs::copy_file(source, target, fs::copy_options::none, error);
      if (!error) fs::permissions(target, fs::status(source, error).permissions(), error);
    }
    if (error)
      return refuse(RefusalCode::FileUnwritable,
                    std::format("{} : {}", prepare::utf8_path(target), error.message()));
  }
  return {};
}

// La liste des fichiers de la version installée, écrite à chaque mise à jour (une ligne par chemin relatif,
// en UTF-8) : ce qu'une version livrait et que la suivante ne livre plus se retire. Avant la première mise à
// jour qui l'écrit, il n'y en a pas, et rien n'est retiré.
constexpr std::string_view kFileList = ".ajp-files";

std::optional<std::vector<fs::path>> read_file_list(const fs::path& install) {
  std::ifstream in(install / kFileList, std::ios::binary);
  if (!in) return std::nullopt;
  std::vector<fs::path> files;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const fs::path relative = prepare::path_from_utf8(line);
    // Jamais hors de l'installation : une liste abîmée ne fait pas retirer « ../ ».
    if (!line.empty() && relative.is_relative() && !line.contains("..")) files.push_back(relative);
  }
  std::ranges::sort(files);
  return files;
}

std::expected<void, Refusal> write_file_list(const fs::path& install, const std::vector<fs::path>& files) {
  std::string text;
  for (const fs::path& file : files)
    if (file != fs::path(kFileList)) text += prepare::utf8_path(file) + "\n";
  return core::write_text_atomically(install / kFileList, text);
}

// Retire de l'installation ce que la liste d'avant tenait et que la nouvelle version n'a pas.
std::expected<void, Refusal> remove_dropped(const fs::path& install,
                                            const std::optional<std::vector<fs::path>>& before,
                                            const std::vector<fs::path>& now) {
  if (!before) return {};
  for (const fs::path& file : *before) {
    if (file == fs::path(kFileList) || std::ranges::binary_search(now, file)) continue;
    if (auto removed = remove_installed(install / file); !removed) return removed;
    log_channel("app")->info("mise à jour : {} retiré, absent de la nouvelle version",  // i18n-ok : journal
                             prepare::utf8_path(file));
  }
  return {};
}

// L'installation remise telle que previous la garde : ses fichiers copiés, ceux que la version remplacée
// avait en plus retirés, et sa liste réécrite.
std::expected<void, Refusal> restore_previous(const fs::path& previous, const fs::path& install,
                                              const std::optional<std::vector<fs::path>>& current) {
  const auto kept = tree_files(previous);
  if (!kept) return std::unexpected(kept.error());
  if (auto done = replace_tree(previous, install); !done) return done;
  if (auto done = remove_dropped(install, current, *kept); !done) return done;
  return write_file_list(install, *kept);
}

// La racine unique d'une archive extraite (« ajp-linux-x64/ »), ou le dossier lui-même.
fs::path archive_root(const fs::path& extracted) {
  std::error_code error;
  fs::path only;
  int count = 0;
  for (const fs::directory_entry& entry : core::dir_entries(extracted, error)) {
    only = entry.path();
    ++count;
  }
  return count == 1 && fs::is_directory(only, error) ? only : extracted;
}

fs::path updates_dir(const UpdatePaths& paths) {
  return paths.data / "updates";
}

}  // namespace

std::string_view release_repo() {
  return AJP_RELEASE_REPO;
}
std::string_view update_public_key() {
  return AJP_UPDATE_PUBKEY;
}

std::vector<std::string> update_public_keys(const std::vector<std::string>& revoked) {
  std::vector<std::string> keys;
  for (const std::string_view key :
       {std::string_view(AJP_UPDATE_PUBKEY), std::string_view(AJP_UPDATE_PUBKEY_BACKUP)})
    if (!key.empty() && std::ranges::find(revoked, key) == revoked.end()) keys.emplace_back(key);
  return keys;
}
std::string_view installed_version() {
  return AJP_VERSION;
}

std::string manifest_url(std::string_view repo, std::string_view channel) {
  if (channel == "preview")
    return std::format("https://github.com/{}/releases/download/preview/manifest.toml", repo);
  return std::format("https://github.com/{}/releases/latest/download/manifest.toml", repo);
}

std::string format_update_state(const UpdateState& state) {
  toml::table root;
  root.insert("last_check_unix", state.last_check_unix);
  root.insert("latest_version", state.latest_version);
  root.insert("notes_url", state.notes_url);
  root.insert("staged_version", state.staged_version);
  root.insert("staged_file", state.staged_file);
  root.insert("previous_version", state.previous_version);
  root.insert("last_error", state.last_error);
  root.insert("staged_sha256", state.staged_sha256);
  root.insert("staged_size", static_cast<std::int64_t>(state.staged_size));
  root.insert("unconfirmed_version", state.unconfirmed_version);
  root.insert("unconfirmed_launches", static_cast<std::int64_t>(state.unconfirmed_launches));
  root.insert("skipped_version", state.skipped_version);
  root.insert("launch_open", state.launch_open);
  toml::array revoked;
  for (const std::string& key : state.revoked_keys) revoked.push_back(key);
  root.insert("revoked_keys", std::move(revoked));
  std::ostringstream out;
  out << root << '\n';
  return out.str();
}

std::expected<UpdateState, Refusal> parse_update_state(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed)
    return refuse(RefusalCode::FileUnreadable, std::format("update.toml : {}", parsed.error().description()));
  const toml::table& root = parsed.table();
  UpdateState state;
  state.last_check_unix = root["last_check_unix"].value_or(std::int64_t{0});
  state.latest_version = root["latest_version"].value_or(std::string());
  state.notes_url = root["notes_url"].value_or(std::string());
  state.staged_version = root["staged_version"].value_or(std::string());
  state.staged_file = root["staged_file"].value_or(std::string());
  state.previous_version = root["previous_version"].value_or(std::string());
  state.last_error = root["last_error"].value_or(std::string());
  state.staged_sha256 = root["staged_sha256"].value_or(std::string());
  state.staged_size =
      static_cast<std::uint64_t>(std::max<std::int64_t>(root["staged_size"].value_or(std::int64_t{0}), 0));
  state.unconfirmed_version = root["unconfirmed_version"].value_or(std::string());
  state.unconfirmed_launches = static_cast<int>(
      std::clamp<std::int64_t>(root["unconfirmed_launches"].value_or(std::int64_t{0}), 0, 1000));
  state.skipped_version = root["skipped_version"].value_or(std::string());
  state.launch_open = root["launch_open"].value_or(false);
  if (const toml::array* revoked = root["revoked_keys"].as_array())
    for (const toml::node& key : *revoked)
      if (auto hex = key.value<std::string>()) state.revoked_keys.push_back(std::move(*hex));
  return state;
}

UpdateState load_update_state(const fs::path& file) {
  std::error_code error;
  if (!fs::exists(file, error)) return {};
  auto state = parse_update_state(read_all(file));
  return state ? *state : UpdateState{};
}

std::expected<void, Refusal> save_update_state(const fs::path& file, const UpdateState& state) {
  return core::write_text_atomically(file, format_update_state(state));
}

bool update_due(const UpdateState& state, std::string_view mode, std::string_view frequency,
                std::int64_t now_unix) {
  if (mode == "never") return false;
  if (frequency == "launch") return true;
  const std::int64_t interval = frequency == "weekly" ? std::int64_t{7} * 86400 : std::int64_t{86400};
  return state.last_check_unix == 0 || now_unix - state.last_check_unix >= interval ||
         now_unix < state.last_check_unix;
}

std::optional<int> curl_status_line(std::string_view line) {
  constexpr std::string_view kTag = "ajp-http ";
  if (!line.starts_with(kTag)) return std::nullopt;
  line.remove_prefix(kTag.size());
  int status = 0;
  const auto [end, error] = std::from_chars(line.data(), line.data() + line.size(), status);
  if (error != std::errc{} || end != line.data() + line.size() || status < 0 || status >= 600)
    return std::nullopt;
  return status;
}

Refusal curl_failure(int code, int http_status) {
  const std::string what = i18n::tr("download");
  const auto failed = [&](RefusalCode refusal, std::string text) {
    return Refusal{refusal, std::format("{} : {}", what, text)};
  };
  switch (code) {
    case 22: {
      const int status = http_status;
      if (status == 404)
        return failed(RefusalCode::TrackUnavailable, i18n::tr("file not found on the server"));
      if (status != 0)
        return failed(RefusalCode::DecodeFailed,
                      i18n::tr_format("the server refused the download (HTTP {})", status));
      return failed(RefusalCode::DecodeFailed, i18n::tr("the server refused the download"));
    }
    case 5:
    case 6:
      return failed(RefusalCode::DecodeFailed, i18n::tr("no Internet connection (server name not found)"));
    case 7:
      return failed(RefusalCode::DecodeFailed, i18n::tr("the update server cannot be reached"));
    case 28:
      return failed(RefusalCode::DecodeFailed, i18n::tr("the update server took too long to answer"));
    case 35:
    case 51:
    case 58:
    case 60:
    case 77:
      return failed(RefusalCode::DecodeFailed, i18n::tr("secure connection failed (certificate or proxy)"));
    case 23:
      return failed(RefusalCode::FileUnwritable, i18n::tr("cannot write the downloaded file"));
    default:
      return failed(RefusalCode::DecodeFailed, i18n::tr_format("download failed (curl code {})", code));
  }
}

std::expected<CheckResult, Refusal> check_for_update(std::string_view channel, const std::stop_token& stop) {
  return check_for_update(channel, stop, fetch, update_public_keys({}));
}

std::expected<CheckResult, Refusal> check_for_update(std::string_view channel, const std::stop_token& stop,
                                                     const UpdateFetch& fetch,
                                                     const std::vector<std::string>& keys) {
  if (keys.empty())
    return refuse(RefusalCode::LibraryUnavailable,
                  i18n::tr("this package has no signing key: updates cannot be verified"));
  const auto made = private_temp_dir();
  if (!made) return std::unexpected(made.error());
  const fs::path& dir = *made;
  struct Cleanup {
    fs::path dir;
    ~Cleanup() {
      std::error_code ignored;
      fs::remove_all(dir, ignored);
    }
  } cleanup{dir};
  const std::string url = manifest_url(release_repo(), channel);
  if (auto got = fetch(url, dir / "manifest.toml", 30, stop); !got) {
    // Rien de publié sur ce canal : pas une panne, il n'y a simplement rien de plus récent.
    if (got.error().code == RefusalCode::TrackUnavailable)
      return CheckResult{
          .newer = std::nullopt,
          .message = i18n::tr_format(
              "No version is published on the \"{}\" channel yet: the player is up to date", channel)};
    return std::unexpected(got.error());
  }
  if (auto got = fetch(url + ".sig", dir / "manifest.toml.sig", 30, stop); !got)
    return std::unexpected(got.error());
  const std::string manifest_text = read_all(dir / "manifest.toml");
  const std::string signature = trimmed(read_all(dir / "manifest.toml.sig"));
  const auto signer = std::ranges::find_if(
      keys, [&](const std::string& key) { return ed25519_verify_hex(key, manifest_text, signature); });
  if (signer == keys.end())
    return refuse(RefusalCode::HashMismatch, i18n::tr("the manifest signature is wrong: update ignored"));
  auto manifest = core::parse_update_manifest(manifest_text);
  if (!manifest) return std::unexpected(manifest.error());
  if (std::ranges::find(manifest->revoked_keys, *signer) != manifest->revoked_keys.end())
    return refuse(RefusalCode::HashMismatch, i18n::tr("the manifest signature is wrong: update ignored"));
  if (core::manifest_expired(*manifest,
                             std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())))
    return refuse(RefusalCode::FieldOutOfRange, i18n::tr("the update manifest has expired: update ignored"));
  if (manifest->channel != channel)
    return refuse(
        RefusalCode::FieldOutOfRange,
        i18n::tr_format("the manifest is for channel \"{}\", not \"{}\"", manifest->channel, channel));
  CheckResult result;
  result.revoked_keys = manifest->revoked_keys;
  const auto order = core::compare_versions(manifest->version, installed_version());
  if (order && *order > 0 && core::artifact_for(*manifest, core::kHostOs) != nullptr) {
    result.message = i18n::tr_format("{} is available", manifest->version);
    result.newer = std::move(*manifest);
  } else {
    result.message = i18n::tr("The player is up to date");
  }
  return result;
}

std::expected<fs::path, Refusal> download_update(const UpdatePaths& paths,
                                                 const core::UpdateManifest& manifest,
                                                 const std::stop_token& stop) {
  return download_update(paths, manifest, stop, fetch);
}

std::expected<fs::path, Refusal> download_update(const UpdatePaths& paths,
                                                 const core::UpdateManifest& manifest,
                                                 const std::stop_token& stop, const UpdateFetch& fetch) {
  const core::UpdateArtifact* artifact = core::artifact_for(manifest, core::kHostOs);
  if (artifact == nullptr)
    return refuse(RefusalCode::TrackUnavailable, i18n::tr("this release has no archive for this system"));
  const std::string name = artifact->url.substr(artifact->url.rfind('/') + 1);
  if (name.empty() || name.find("..") != std::string::npos)
    return refuse(RefusalCode::FieldOutOfRange, i18n::tr("unreadable archive name in the manifest"));
  std::error_code error;
  const fs::path dir = updates_dir(paths) / manifest.version;
  fs::create_directories(dir, error);
  fs::path file = dir / name;
  if (auto got = fetch(artifact->url, file, 1800, stop); !got) return std::unexpected(got.error());
  if (auto ok = core::verify_file(file, artifact->size, artifact->sha256, stop); !ok) {
    fs::remove(file, error);
    return std::unexpected(ok.error());
  }
  return file;
}

std::string update_managed_reason(const UpdatePaths& paths) {
#ifdef __APPLE__
  return i18n::tr("on macOS, download the new version from the releases page");
#endif
  // L'AppImage (AJP_PACKAGED, posée par son lanceur) est un fichier en lecture seule : on le remplace par le
  // nouveau. MSVC tient getenv pour dangereux (C4996) : seul l'AppImage, sous Linux, pose la variable.
#if defined(_MSC_VER)
#pragma warning(suppress : 4996)
#endif
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  const char* packaged = std::getenv("AJP_PACKAGED");  // env: lanceur
  if (packaged != nullptr && std::string_view(packaged) == "appimage")
    return i18n::tr("AppImage: download the new version from the releases page (replace the file)");
  std::error_code error;
  if (fs::exists("/.flatpak-info", error)) return i18n::tr("updated by your package manager (Flatpak)");
  const fs::path probe = paths.install / ".ajp-write-test";
  {
    std::ofstream out(probe, std::ios::binary);
    if (!out) return i18n::tr("installed where the player cannot write: updated by your package manager");
  }
  fs::remove(probe, error);
  return {};
}

std::expected<std::string, Refusal> run_update_check(const UpdatePaths& paths, std::string_view mode,
                                                     std::string_view channel, UpdateState& state,
                                                     const std::stop_token& stop) {
  state.last_check_unix =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  auto checked = check_for_update(channel, stop, fetch, update_public_keys(state.revoked_keys));
  if (checked)
    for (std::string& key : checked->revoked_keys)
      if (std::ranges::find(state.revoked_keys, key) == state.revoked_keys.end())
        state.revoked_keys.push_back(std::move(key));
  if (!checked) {
    state.last_error = checked.error().detail;
    return std::unexpected(checked.error());
  }
  state.last_error.clear();
  if (!checked->newer) {
    state.latest_version.clear();
    return checked->message;
  }
  state.latest_version = checked->newer->version;
  state.notes_url = checked->newer->notes_url;
  if (mode != "auto" || !update_managed_reason(paths).empty()) return checked->message;
  if (checked->newer->version == state.skipped_version)
    return checked->message + i18n::tr(" (not installed automatically: it did not start properly here)");
  if (state.staged_version == checked->newer->version)
    return checked->message + i18n::tr(" (already prepared)");
  auto file = download_update(paths, *checked->newer, stop);
  if (!file) {
    state.last_error = file.error().detail;
    return std::unexpected(file.error());
  }
  state.staged_version = checked->newer->version;
  state.staged_file = file->string();
  if (const core::UpdateArtifact* artifact = core::artifact_for(*checked->newer, core::kHostOs)) {
    state.staged_sha256 = artifact->sha256;
    state.staged_size = artifact->size;
  }
  return checked->message + i18n::tr(": prepared, installed at next launch");
}

std::expected<void, Refusal> extract_archive(const fs::path& archive, const fs::path& into) {
  const auto tar = system_tool("tar");
  if (!tar) return refuse(RefusalCode::LibraryUnavailable, i18n::tr("tar not found"));
  std::error_code error;
  fs::create_directories(into, error);
  prepare::ProcessSpec spec{.program = *tar,
                            .args = {"-xf", prepare::utf8_path(archive), "-C", prepare::utf8_path(into)},
                            .timeout = std::chrono::minutes(10)};
  return run_tool(spec, {}, {}, "extraction");
}

std::expected<std::string, Refusal> apply_staged_update(const UpdatePaths& paths, UpdateState& state) {
  if (state.staged_file.empty() || state.staged_version.empty())
    return refuse(RefusalCode::TrackUnavailable, i18n::tr("no update prepared"));
  std::error_code error;
  const fs::path archive = state.staged_file;
  if (!fs::exists(archive, error))
    return refuse(RefusalCode::FileUnreadable, i18n::tr("the prepared archive is gone"));
  // L'archive a pu changer depuis sa préparation (disque, autre programme) : revérifiée avant tout.
  if (!state.staged_sha256.empty()) {
    if (auto ok = core::verify_file(archive, state.staged_size, state.staged_sha256); !ok) {
      fs::remove(archive, error);
      return std::unexpected(ok.error());
    }
  }
  const fs::path extract = updates_dir(paths) / "extract";
  const fs::path previous = updates_dir(paths) / "previous";
  fs::remove_all(extract, error);
  if (auto done = extract_archive(archive, extract); !done) return std::unexpected(done.error());
  const fs::path root = archive_root(extract);
  // Une archive sans le lecteur ne remplace rien : l'installation resterait sans exécutable.
  if (!paths.executable.empty() && !fs::is_regular_file(root / paths.executable.filename(), error)) {
    fs::remove_all(extract, error);
    return refuse(
        RefusalCode::FileUnreadable,
        i18n::tr_format("the archive does not contain {}", prepare::utf8_path(paths.executable.filename())));
  }
  const auto delivered = tree_files(root);
  if (!delivered) {
    fs::remove_all(extract, error);
    return std::unexpected(delivered.error());
  }
  const std::optional<std::vector<fs::path>> before = read_file_list(paths.install);
  fs::remove_all(previous, error);
  fs::create_directories(previous, error);
  if (auto saved = replace_tree(paths.install, previous); !saved) {
    fs::remove_all(extract, error);
    return std::unexpected(saved.error());
  }
  auto done = replace_tree(root, paths.install);
  if (done) done = remove_dropped(paths.install, before, *delivered);
  if (done) done = write_file_list(paths.install, *delivered);
  if (!done) {
    // Une copie ratée à moitié : l'installation d'avant revient, sans les fichiers que la nouvelle a ajoutés.
    if (const auto restored = restore_previous(previous, paths.install, *delivered); !restored)
      log_channel("app")->warn(
          "mise à jour : l'ancienne version n'a pas pu être remise : {}",  // i18n-ok : journal
          restored.error().detail);
    fs::remove_all(extract, error);
    return std::unexpected(done.error());
  }
  fs::remove_all(extract, error);
  const std::string version = state.staged_version;
  state.previous_version = std::string(installed_version());
  state.staged_version.clear();
  state.staged_sha256.clear();
  state.staged_size = 0;
  state.latest_version.clear();
  state.unconfirmed_version = version;
  state.unconfirmed_launches = 0;
  fs::remove(archive, error);
  return i18n::tr_format("Updated to {}", version);
}

std::expected<std::string, Refusal> roll_back_update(const UpdatePaths& paths, UpdateState& state) {
  const fs::path previous = updates_dir(paths) / "previous";
  std::error_code error;
  if (!fs::exists(previous, error))
    return refuse(RefusalCode::TrackUnavailable, i18n::tr("no previous version kept"));
  // Ce que la version annulée a ajouté se retire avec elle.
  if (auto done = restore_previous(previous, paths.install, read_file_list(paths.install)); !done)
    return std::unexpected(done.error());
  const std::string back = state.previous_version;
  fs::remove_all(previous, error);
  state.previous_version.clear();
  state.latest_version.clear();
  state.unconfirmed_version.clear();
  state.unconfirmed_launches = 0;
  return i18n::tr_format("Previous version restored{}", back.empty() ? "" : " : " + back);
}

std::optional<std::string> guard_new_version(const UpdatePaths& paths, UpdateState& state,
                                             std::string_view running_version) {
  std::error_code error;
  // Les « .old » d'une application (fichiers en service renommés, Windows) : plus en service à présent.
  // Parcours à error_code (increment) : un dossier qui disparaît pendant le parcours ne lève pas au
  // lancement.
  if (!paths.install.empty() && fs::is_directory(paths.install, error)) {
    std::vector<fs::path> old;
    fs::recursive_directory_iterator it(paths.install, fs::directory_options::skip_permission_denied, error);
    for (; !error && it != fs::recursive_directory_iterator(); it.increment(error))
      if (it->path().extension() == ".old" && it->is_regular_file(error)) old.push_back(it->path());
    for (const fs::path& file : old) fs::remove(file, error);
  }
  if (state.unconfirmed_version.empty()) return std::nullopt;
  if (state.unconfirmed_version != running_version) {
    // Une autre version tourne (installée à la main, ou déjà revenue) : plus rien à garder.
    state.unconfirmed_version.clear();
    state.unconfirmed_launches = 0;
    return std::nullopt;
  }
  // Seul un lancement resté ouvert (le lecteur a planté ou a été tué) compte ; une fermeture voulue, non.
  if (state.launch_open) ++state.unconfirmed_launches;
  state.launch_open = true;
  if (state.unconfirmed_launches < kMaxUnconfirmedLaunches) return std::nullopt;
  const std::string bad = state.unconfirmed_version;
  auto back = roll_back_update(paths, state);
  if (!back) {
    state.last_error = back.error().detail;
    state.unconfirmed_version.clear();  // rien à remettre : ne pas réessayer à chaque lancement
    state.unconfirmed_launches = 0;
    return std::nullopt;
  }
  state.skipped_version = bad;
  state.last_error = i18n::tr_format("version {} did not start properly: previous version restored", bad);
  return *back;
}

bool close_launch(UpdateState& state) {
  return std::exchange(state.launch_open, false);
}

bool confirm_version(UpdateState& state, std::string_view running_version) {
  if (state.unconfirmed_version.empty() || state.unconfirmed_version != running_version) return false;
  state.unconfirmed_version.clear();
  state.unconfirmed_launches = 0;
  return true;
}

}  // namespace ajp::app::play
