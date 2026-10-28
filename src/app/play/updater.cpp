#include "app/play/updater.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>

#include "app/play/ed25519.h"
#include "app/play/engine_tools.h"
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

// curl du système : https seulement, délai borné.
std::expected<void, Refusal> fetch(std::string_view url, const fs::path& file, int max_seconds,
                                   const std::stop_token& stop) {
  const auto curl = system_tool("curl");
  if (!curl)
    return refuse(RefusalCode::LibraryUnavailable,
                  "curl introuvable : impossible de vérifier les mises à jour");
  prepare::ProcessSpec spec{.program = *curl,
                            .args = {"-fsSL", "--proto", "=https", "--max-time", std::to_string(max_seconds),
                                     "-o", prepare::utf8_path(file), std::string(url)},
                            .timeout = std::chrono::seconds(max_seconds + 10)};
  return run_tool(spec, {}, stop, "téléchargement");
}

// Copie récursive qui remplace les fichiers même en service : le fichier est retiré (ou, sous Windows,
// renommé en « .old ») avant d'être recréé, ce que l'ouverture en écriture d'un exécutable chargé refuserait.
std::expected<void, Refusal> replace_tree(const fs::path& from, const fs::path& to) {
  std::error_code error;
  fs::create_directories(to, error);
  for (const fs::directory_entry& entry :
       fs::recursive_directory_iterator(from, fs::directory_options::skip_permission_denied, error)) {
    const fs::path relative = fs::relative(entry.path(), from, error);
    const fs::path target = to / relative;
    if (entry.is_directory(error)) {
      fs::create_directories(target, error);
      continue;
    }
    fs::create_directories(target.parent_path(), error);
    if (fs::exists(target, error) || fs::is_symlink(target, error)) {
      fs::remove(target, error);
      if (error) {
        error.clear();
        fs::rename(target, fs::path(target.string() + ".old"), error);
        if (error)
          return refuse(RefusalCode::FileUnwritable,
                        std::format("{} : {}", prepare::utf8_path(target), error.message()));
      }
    }
    if (entry.is_symlink(error)) {
      fs::copy_symlink(entry.path(), target, error);
    } else {
      fs::copy_file(entry.path(), target, fs::copy_options::none, error);
      if (!error) fs::permissions(target, fs::status(entry.path(), error).permissions(), error);
    }
    if (error)
      return refuse(RefusalCode::FileUnwritable,
                    std::format("{} : {}", prepare::utf8_path(target), error.message()));
  }
  return {};
}

// La racine unique d'une archive extraite (« ajp-linux-x64/ »), ou le dossier lui-même.
fs::path archive_root(const fs::path& extracted) {
  std::error_code error;
  fs::path only;
  int count = 0;
  for (const fs::directory_entry& entry : fs::directory_iterator(extracted, error)) {
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

std::expected<CheckResult, Refusal> check_for_update(std::string_view channel, const std::stop_token& stop) {
  const std::string_view key = update_public_key();
  if (key.empty())
    return refuse(RefusalCode::LibraryUnavailable,
                  "ce paquet n'a pas de clé de signature : les mises à jour ne peuvent pas être vérifiées");
  std::error_code error;
  const fs::path dir =
      fs::temp_directory_path(error) /
      std::format("ajp-update-{}", std::chrono::steady_clock::now().time_since_epoch().count());
  fs::create_directories(dir, error);
  struct Cleanup {
    fs::path dir;
    ~Cleanup() {
      std::error_code ignored;
      fs::remove_all(dir, ignored);
    }
  } cleanup{dir};
  const std::string url = manifest_url(release_repo(), channel);
  if (auto got = fetch(url, dir / "manifest.toml", 30, stop); !got) return std::unexpected(got.error());
  if (auto got = fetch(url + ".sig", dir / "manifest.toml.sig", 30, stop); !got)
    return std::unexpected(got.error());
  const std::string manifest_text = read_all(dir / "manifest.toml");
  const std::string signature = trimmed(read_all(dir / "manifest.toml.sig"));
  if (!ed25519_verify_hex(key, manifest_text, signature))
    return refuse(RefusalCode::HashMismatch, "la signature du manifeste est fausse : mise à jour ignorée");
  auto manifest = core::parse_update_manifest(manifest_text);
  if (!manifest) return std::unexpected(manifest.error());
  if (manifest->channel != channel)
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("le manifeste est du canal « {} », pas « {} »", manifest->channel, channel));
  CheckResult result;
  const auto order = core::compare_versions(manifest->version, installed_version());
  if (order && *order > 0 && core::artifact_for(*manifest, core::kHostOs) != nullptr) {
    result.message = std::format("{} est disponible", manifest->version);
    result.newer = std::move(*manifest);
  } else {
    result.message = "Le lecteur est à jour";
  }
  return result;
}

std::expected<fs::path, Refusal> download_update(const UpdatePaths& paths,
                                                 const core::UpdateManifest& manifest,
                                                 const std::stop_token& stop) {
  const core::UpdateArtifact* artifact = core::artifact_for(manifest, core::kHostOs);
  if (artifact == nullptr)
    return refuse(RefusalCode::TrackUnavailable, "cette release n'a pas d'archive pour ce système");
  const std::string name = artifact->url.substr(artifact->url.rfind('/') + 1);
  if (name.empty() || name.find("..") != std::string::npos)
    return refuse(RefusalCode::FieldOutOfRange, "nom d'archive illisible dans le manifeste");
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
  return "sur macOS, la nouvelle version se télécharge sur la page des releases";
#endif
  // L'AppImage (AJP_PACKAGED, posée par son lanceur) est un fichier en lecture seule : on le remplace par le
  // nouveau. NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (const char* packaged = std::getenv("AJP_PACKAGED");
      packaged != nullptr && std::string_view(packaged) == "appimage")
    return "AppImage : la nouvelle version se télécharge sur la page des releases (le fichier se remplace)";
  std::error_code error;
  if (fs::exists("/.flatpak-info", error)) return "mis à jour par votre gestionnaire de paquets (Flatpak)";
  const fs::path probe = paths.install / ".ajp-write-test";
  {
    std::ofstream out(probe, std::ios::binary);
    if (!out)
      return "installé là où le lecteur ne peut pas s'écrire : mis à jour par votre gestionnaire de paquets";
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
  auto checked = check_for_update(channel, stop);
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
  if (state.staged_version == checked->newer->version) return checked->message + " (déjà préparée)";
  auto file = download_update(paths, *checked->newer, stop);
  if (!file) {
    state.last_error = file.error().detail;
    return std::unexpected(file.error());
  }
  state.staged_version = checked->newer->version;
  state.staged_file = file->string();
  return checked->message + " : préparée, installée au prochain lancement";
}

std::expected<void, Refusal> extract_archive(const fs::path& archive, const fs::path& into) {
  const auto tar = system_tool("tar");
  if (!tar) return refuse(RefusalCode::LibraryUnavailable, "tar introuvable");
  std::error_code error;
  fs::create_directories(into, error);
  prepare::ProcessSpec spec{.program = *tar,
                            .args = {"-xf", prepare::utf8_path(archive), "-C", prepare::utf8_path(into)},
                            .timeout = std::chrono::minutes(10)};
  return run_tool(spec, {}, {}, "extraction");
}

std::expected<std::string, Refusal> apply_staged_update(const UpdatePaths& paths, UpdateState& state) {
  if (state.staged_file.empty() || state.staged_version.empty())
    return refuse(RefusalCode::TrackUnavailable, "aucune mise à jour préparée");
  std::error_code error;
  const fs::path archive = state.staged_file;
  if (!fs::exists(archive, error)) return refuse(RefusalCode::FileUnreadable, "l'archive préparée a disparu");
  const fs::path extract = updates_dir(paths) / "extract";
  const fs::path previous = updates_dir(paths) / "previous";
  fs::remove_all(extract, error);
  if (auto done = extract_archive(archive, extract); !done) return std::unexpected(done.error());
  const fs::path root = archive_root(extract);
  fs::remove_all(previous, error);
  fs::create_directories(previous, error);
  if (auto saved = replace_tree(paths.install, previous); !saved) {
    fs::remove_all(extract, error);
    return std::unexpected(saved.error());
  }
  if (auto done = replace_tree(root, paths.install); !done) {
    // Une copie ratée à moitié : l'installation d'avant revient.
    if (const auto restored = replace_tree(previous, paths.install); !restored)
      log_channel("app")->warn("mise à jour : l'ancienne version n'a pas pu être remise : {}",
                               restored.error().detail);
    fs::remove_all(extract, error);
    return std::unexpected(done.error());
  }
  fs::remove_all(extract, error);
  const std::string version = state.staged_version;
  state.previous_version = std::string(installed_version());
  state.staged_version.clear();
  state.latest_version.clear();
  fs::remove(archive, error);
  return std::format("Mis à jour vers {}", version);
}

std::expected<std::string, Refusal> roll_back_update(const UpdatePaths& paths, UpdateState& state) {
  const fs::path previous = updates_dir(paths) / "previous";
  std::error_code error;
  if (!fs::exists(previous, error))
    return refuse(RefusalCode::TrackUnavailable, "aucune version précédente gardée");
  if (auto done = replace_tree(previous, paths.install); !done) return std::unexpected(done.error());
  const std::string back = state.previous_version;
  fs::remove_all(previous, error);
  state.previous_version.clear();
  state.latest_version.clear();
  return std::format("Version précédente remise{}", back.empty() ? "" : " : " + back);
}

}  // namespace ajp::app::play
