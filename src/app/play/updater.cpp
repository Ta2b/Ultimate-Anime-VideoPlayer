#include "app/play/updater.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
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
#include "core/file_sync.h"
#include "core/log.h"
#include "core/text.h"
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

std::string read_all(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
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

// Au-delà, le fichier lu n'est ni un manifeste ni un index de moteurs : curl l'abandonne.
constexpr std::int64_t kMaxFetchBytes = std::int64_t{1} << 20;

// curl du système : https seulement, délai et taille bornés. Un fichier absent du serveur (HTTP 404) est
// refusé TrackUnavailable, que check_for_update distingue d'une panne.
std::expected<void, Refusal> fetch(std::string_view url, const fs::path& file, int max_seconds,
                                   const std::stop_token& stop) {
  const auto curl = system_tool("curl");
  if (!curl)
    return refuse(RefusalCode::LibraryUnavailable, i18n::tr("curl not found: cannot check for updates"));
  // -q en premier : aucun .curlrc ne change ce que fait curl ; https seulement, redirections comprises ;
  // un manifeste ou un index tient en quelques Ko (SEC-13, A35-05). Sous Windows, le proxy et la révocation
  // vérifiée au mieux, comme les moteurs (A06-13).
  std::vector<std::string> args{"-q",
                                "-fsSL",
                                "--proto",
                                "=https",
                                "--proto-redir",
                                "=https",
                                "--max-filesize",
                                std::to_string(kMaxFetchBytes),
                                "--retry",
                                "2",
                                "--max-time",
                                std::to_string(max_seconds)};
  for (std::string& arg : curl_system_args()) args.push_back(std::move(arg));
  for (std::string arg : {std::string("--write-out"), std::string(kCurlStatusFormat), std::string("-o"),
                          prepare::ansi_safe_path(file), std::string(url)})
    args.push_back(std::move(arg));
  prepare::ProcessSpec spec{
      .program = *curl, .args = std::move(args), .timeout = std::chrono::seconds(max_seconds + 10)};
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
std::expected<void, Refusal> fetch_https(std::string_view url, const fs::path& file, int max_seconds,
                                         const std::stop_token& stop) {
  return fetch(url, file, max_seconds, stop);
}

std::string_view installed_version() {
  return AJP_VERSION;
}

std::string manifest_url(std::string_view repo) {
  return std::format("https://github.com/{}/releases/latest/download/manifest.toml", repo);
}

std::string releases_page(std::string_view repo) {
  return std::format("https://github.com/{}/releases/latest", repo);
}

std::string format_update_state(const UpdateState& state) {
  toml::table root;
  root.insert("last_check_unix", state.last_check_unix);
  root.insert("last_attempt_unix", state.last_attempt_unix);
  root.insert("latest_version", state.latest_version);
  root.insert("notes_url", state.notes_url);
  root.insert("last_error", state.last_error);
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
  state.last_attempt_unix = root["last_attempt_unix"].value_or(std::int64_t{0});
  state.latest_version = root["latest_version"].value_or(std::string());
  state.notes_url = root["notes_url"].value_or(std::string());
  state.last_error = root["last_error"].value_or(std::string());
  if (const toml::array* revoked = root["revoked_keys"].as_array())
    for (const toml::node& key : *revoked)
      if (auto hex = key.value<std::string>()) state.revoked_keys.push_back(std::move(*hex));
  return state;
}

void forget_installed_announce(UpdateState& state, std::string_view installed) {
  if (state.latest_version.empty()) return;
  const auto order = core::compare_versions(state.latest_version, installed);
  if (order && *order > 0) return;
  state.latest_version.clear();
  state.notes_url.clear();
}

UpdateState load_update_state(const fs::path& file) {
  std::error_code error;
  if (!fs::exists(file, error)) return {};
  auto parsed = parse_update_state(read_all(file));
  UpdateState state = parsed ? *parsed : UpdateState{};
  forget_installed_announce(state, installed_version());
  return state;
}

namespace {

// Les clés que cette version écrit, et celles des versions qui s'appliquaient elles-mêmes : retirées.
bool own_or_obsolete_key(std::string_view key) {
  static constexpr std::array<std::string_view, 6> kOwn{
      "last_check_unix", "latest_version", "notes_url", "last_error", "revoked_keys", "last_attempt_unix"};
  return std::ranges::contains(kOwn, key) || key.starts_with("staged_") || key.starts_with("unconfirmed_") ||
         key == "previous_version";
}

// Une à la fois dans le processus : le fil des mises à jour et le bouton « Check now » (A26-10).
std::mutex& update_state_mutex() {
  static std::mutex mutex;
  return mutex;
}

}  // namespace

std::string merge_update_state(std::string_view previous, const UpdateState& state) {
  std::string own = format_update_state(state);
  toml::parse_result before = toml::parse(previous);
  if (!before) return own;
  toml::table merged = toml::parse(own).table();
  // Les clés d'une version plus récente restent : un retour à celle-ci les retrouve (A36-05).
  for (auto&& [key, node] : before.table())
    if (!own_or_obsolete_key(key.str())) merged.insert(key, node);
  std::ostringstream out;
  out << merged << '\n';
  return out.str();
}

std::expected<void, Refusal> save_update_state(const fs::path& file, const UpdateState& state) {
  std::error_code error;
  const std::string previous = fs::exists(file, error) ? read_all(file) : std::string();
  return core::write_text_atomically(file, merge_update_state(previous, state));
}

std::expected<void, Refusal> change_update_state(const fs::path& file,
                                                 const std::function<void(UpdateState&)>& change) {
  const std::scoped_lock lock(update_state_mutex());
  UpdateState state = load_update_state(file);
  const UpdateState before = state;
  change(state);
  if (state == before) return {};
  return save_update_state(file, state);
}

void check_updates_if_due(const fs::path& file, UpdateMode mode, UpdateFrequency frequency, bool may_check,
                          std::int64_t now_unix, const std::stop_token& stop) {
  std::optional<std::expected<std::string, Refusal>> message;
  const auto kept = change_update_state(file, [&](UpdateState& state) {
    if (may_check && update_due(state, mode, frequency, now_unix)) message = run_update_check(state, stop);
  });
  if (!kept) log_channel("app")->warn("update.toml : {}", kept.error().detail);
  if (!message) return;
  log_channel("app")->info("mise à jour : {}",  // i18n-ok : journal
                           *message ? **message : message->error().detail);
}

std::expected<std::string, Refusal> check_updates_now(const fs::path& file, const std::stop_token& stop) {
  std::expected<std::string, Refusal> message = std::unexpected(Refusal{});
  const auto kept =
      change_update_state(file, [&](UpdateState& state) { message = run_update_check(state, stop); });
  if (!kept && message) return std::unexpected(kept.error());
  return message;
}

bool update_due(const UpdateState& state, UpdateMode mode, UpdateFrequency frequency, std::int64_t now_unix) {
  if (mode == UpdateMode::Never) return false;
  if (frequency == UpdateFrequency::Launch) return true;
  // Une horloge revenue en arrière : vérifier, plutôt que d'attendre qu'elle rattrape la date gardée.
  if (now_unix < state.last_check_unix || now_unix < state.last_attempt_unix) return true;
  // Une tentative récente restée sans réponse (pas de réseau) : un nouvel essai au bout d'une heure, pas de
  // la période entière (A26-07).
  if (state.last_attempt_unix > state.last_check_unix &&
      now_unix - state.last_attempt_unix < kUpdateRetrySeconds)
    return false;
  const std::int64_t interval =
      frequency == UpdateFrequency::Weekly ? std::int64_t{7} * 86400 : std::int64_t{86400};
  return state.last_check_unix == 0 || now_unix - state.last_check_unix >= interval;
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

std::expected<CheckResult, Refusal> check_for_update(const std::stop_token& stop, const UpdateFetch& fetch,
                                                     const std::vector<std::string>& keys,
                                                     std::string_view installed) {
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
  const std::string url = manifest_url(release_repo());
  if (auto got = fetch(url, dir / "manifest.toml", 30, stop); !got) {
    // La dernière release sans manifeste : la CI de release en joint toujours un, son absence n'est jamais
    // « à jour » (SEC-05, A28-02), elle se dit.
    if (got.error().code == RefusalCode::TrackUnavailable)
      return refuse(RefusalCode::TrackUnavailable,
                    i18n::tr("the latest release has no update manifest: check the releases page"));
    return std::unexpected(got.error());
  }
  if (auto got = fetch(url + ".sig", dir / "manifest.toml.sig", 30, stop); !got)
    return std::unexpected(got.error());
  const std::string manifest_text = read_all(dir / "manifest.toml");
  const std::string signature(core::trim(read_all(dir / "manifest.toml.sig")));
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
  if (manifest->channel != "stable")
    return refuse(
        RefusalCode::FieldOutOfRange,
        i18n::tr_format("the manifest is for channel \"{}\", not \"{}\"", manifest->channel, "stable"));
  CheckResult result;
  result.revoked_keys = manifest->revoked_keys;
  const auto order = core::compare_versions(manifest->version, installed);
  if (order && *order > 0) {
    result.message = i18n::tr_format("{} is available", manifest->version);
    result.newer = std::move(*manifest);
  } else {
    result.message = i18n::tr("The player is up to date");
  }
  return result;
}

std::expected<CheckResult, Refusal> check_for_update(const std::stop_token& stop) {
  return check_for_update(stop, fetch, update_public_keys({}), installed_version());
}

std::expected<std::string, Refusal> run_update_check(UpdateState& state, const std::stop_token& stop) {
  const std::int64_t now =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  return run_update_check(state, stop, fetch, now);
}

std::expected<std::string, Refusal> run_update_check(UpdateState& state, const std::stop_token& stop,
                                                     const UpdateFetch& fetch, std::int64_t now_unix) {
  state.last_attempt_unix = now_unix;
  // Le serveur a répondu (fichier reçu, ou absent : 404) : la vérification compte. Sans réponse (pas de
  // réseau, annulation à la fermeture), seule la tentative est notée et un nouvel essai vient au bout d'une
  // heure (A26-07).
  bool answered = false;
  const UpdateFetch noted = [&](std::string_view url, const fs::path& file, int max_seconds,
                                const std::stop_token& token) {
    auto got = fetch(url, file, max_seconds, token);
    if (got || got.error().code == RefusalCode::TrackUnavailable) answered = true;
    return got;
  };
  auto checked = check_for_update(stop, noted, update_public_keys(state.revoked_keys), installed_version());
  if (answered) state.last_check_unix = now_unix;
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
    state.notes_url.clear();
    return checked->message;
  }
  state.latest_version = checked->newer->version;
  state.notes_url = checked->newer->notes_url;
  return checked->message;
}

}  // namespace ajp::app::play
