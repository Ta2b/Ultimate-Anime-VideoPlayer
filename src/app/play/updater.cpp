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
#include "core/log.h"
#include "core/text.h"
#include "core/utf8_path.h"
#include "prepare/process.h"
#include "prepare/utf8.h"

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

std::int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

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
    return refuse(RefusalCode::FileUnwritable, i18n::tr_format("temporary folder: {}", error.message()));
#if defined(_WIN32)
  std::random_device random;
  for (int attempt = 0; attempt < 8; ++attempt) {
    const fs::path dir = base / std::format("ajp-update-{:08x}{:08x}", random(), random());
    if (fs::create_directory(dir, error)) return dir;
  }
  return refuse(RefusalCode::FileUnwritable, i18n::tr("no free name for a temporary folder"));
#else
  std::string pattern = prepare::utf8_path(base / "ajp-update-XXXXXX");
  if (mkdtemp(pattern.data()) == nullptr)
    return refuse(RefusalCode::FileUnwritable, i18n::tr_format("temporary folder: {}", std::strerror(errno)));
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
  // Le fichier s'écrit par son seul nom, dans le dossier de travail de curl : sous Windows, un dossier
  // d'utilisateur hors de la page de code ANSI n'est pas relu par curl, un nom ASCII l'est toujours (B26-06).
  // Un nom qui n'est pas ASCII garde le chemin complet.
  const std::string name = core::to_utf8(file.filename());
  const bool relative = !file.parent_path().empty() && std::ranges::all_of(name, [](char c) {
    return static_cast<unsigned char>(c) < 0x80;
  });
  for (std::string arg : {std::string("--write-out"), std::string(kCurlStatusFormat), std::string("-o"),
                          relative ? name : prepare::ansi_safe_path(file), std::string(url)})
    args.push_back(std::move(arg));
  prepare::ProcessSpec spec{.program = *curl,
                            .args = std::move(args),
                            .timeout = std::chrono::seconds(max_seconds + 10),
                            .working_dir = relative ? file.parent_path() : fs::path()};
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
  if (exit->code != 0) return std::unexpected(curl_failure(static_cast<int>(exit->code), status));
  // --max-filesize ne coupe pas un corps sans Content-Length (transfert découpé) : la taille se contrôle ici,
  // avant que read_all ne charge le fichier en mémoire (B26-05).
  std::error_code size_error;
  const auto written = fs::file_size(file, size_error);
  if (!size_error && written > static_cast<std::uintmax_t>(kMaxFetchBytes)) {
    std::error_code ignored;
    fs::remove(file, ignored);
    return refuse(RefusalCode::DecodeFailed,
                  std::format("{} : {}", i18n::tr("download"), i18n::tr("the file is larger than expected")));
  }
  return {};
}

}  // namespace

std::string_view release_repo() {
  return AJP_RELEASE_REPO;
}
std::string_view update_public_key() {
  return AJP_UPDATE_PUBKEY;
}

std::vector<std::string> update_public_keys(const std::vector<std::string>& revoked) {
  // En minuscules, comme les clés qu'un manifeste révoque : une clé compilée en majuscules serait sinon
  // introuvable dans la liste des révoquées, et sa révocation sans effet (B26-04).
  const auto is_revoked = [&](const std::string& key) {
    return std::ranges::any_of(revoked,
                               [&](const std::string& other) { return core::ascii_lower(other) == key; });
  };
  std::vector<std::string> keys;
  for (const std::string_view compiled :
       {std::string_view(AJP_UPDATE_PUBKEY), std::string_view(AJP_UPDATE_PUBKEY_BACKUP)}) {
    if (compiled.empty()) continue;
    std::string key = core::ascii_lower(compiled);
    if (!is_revoked(key)) keys.push_back(std::move(key));
  }
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

namespace {

// Les révocations qu'un manifeste signé par keys[signer] peut porter : une clé ne révoque que celles qui la
// précèdent dans la liste (l'active, puis la clé de secours). Une clé qui tient l'active ne peut donc pas
// révoquer la clé de secours, seule parade à son vol ; la clé de secours peut révoquer l'active. Une clé que
// le lecteur ne connaît pas (déjà révoquée, ou ancienne) est retenue telle quelle (B26-02).
std::vector<std::string> allowed_revocations(const std::vector<std::string>& keys, std::size_t signer,
                                             const std::vector<std::string>& wanted) {
  std::vector<std::string> allowed;
  for (const std::string& key : wanted) {
    const auto known = std::ranges::find(keys, key);
    if (known != keys.end() && static_cast<std::size_t>(known - keys.begin()) >= signer) continue;
    allowed.push_back(key);
  }
  return allowed;
}

}  // namespace

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
  // Le manifeste et sa signature se prennent en deux requêtes : si la release change entre les deux
  // (manifeste de l'ancienne, signature de la neuve), la signature est fausse. Une seconde tentative refait
  // les deux (B26-03). Une signature absente (404) se lit comme une signature fausse.
  std::string manifest_text;
  auto signer = keys.end();
  for (int attempt = 0; attempt < 2 && signer == keys.end(); ++attempt) {
    if (auto got = fetch(url, dir / "manifest.toml", 30, stop); !got) {
      // La dernière release sans manifeste : la CI de release en joint toujours un, son absence n'est jamais
      // « à jour » (SEC-05, A28-02), elle se dit.
      if (got.error().code == RefusalCode::TrackUnavailable)
        return refuse(RefusalCode::TrackUnavailable,
                      i18n::tr("the latest release has no update manifest: check the releases page"));
      return std::unexpected(got.error());
    }
    const auto sig = fetch(url + ".sig", dir / "manifest.toml.sig", 30, stop);
    if (!sig && sig.error().code != RefusalCode::TrackUnavailable) return std::unexpected(sig.error());
    if (!sig) continue;
    manifest_text = read_all(dir / "manifest.toml");
    const std::string signature(core::trim(read_all(dir / "manifest.toml.sig")));
    signer = std::ranges::find_if(
        keys, [&](const std::string& key) { return ed25519_verify_hex(key, manifest_text, signature); });
  }
  if (signer == keys.end())
    return refuse(RefusalCode::HashMismatch, i18n::tr("the manifest signature is wrong: update ignored"));
  auto manifest = core::parse_update_manifest(manifest_text);
  if (!manifest) {
    // Le détail de core est en français, pour le journal ; l'écran dit la même chose dans la langue choisie
    // (B26-08).
    log_channel("app")->info("manifeste des mises à jour : {}",  // i18n-ok : journal
                             manifest.error().detail);
    return refuse(manifest.error().code, i18n::tr("the update manifest is unreadable or incomplete"));
  }
  if (std::ranges::find(manifest->revoked_keys, *signer) != manifest->revoked_keys.end())
    return refuse(RefusalCode::HashMismatch, i18n::tr("the manifest signature is wrong: update ignored"));
  if (core::manifest_expired(*manifest,
                             std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())))
    return refuse(RefusalCode::FieldOutOfRange, i18n::tr("the update manifest has expired: update ignored"));
  CheckResult result;
  result.revoked_keys =
      allowed_revocations(keys, static_cast<std::size_t>(signer - keys.begin()), manifest->revoked_keys);
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
  return run_update_check(state, stop, fetch, unix_now());
}

std::expected<std::string, Refusal> run_update_check(UpdateState& state, const std::stop_token& stop,
                                                     const UpdateFetch& fetch, std::int64_t now_unix) {
  state.last_attempt_unix = now_unix;
  auto checked = check_for_update(stop, fetch, update_public_keys(state.revoked_keys), installed_version());
  // La vérification compte quand le serveur a répondu et qu'on l'a lue : un manifeste vérifié (même refusé
  // ensuite : expiré, illisible), ou une release sans manifeste (404). Sans réponse, ou coupée avant la
  // signature (pas de réseau, annulation à la fermeture, signature fausse), seule la tentative est notée et
  // un nouvel essai vient au bout d'une heure (A26-07, B26-03).
  const bool answered = checked || checked.error().code == RefusalCode::TrackUnavailable ||
                        checked.error().code == RefusalCode::FieldOutOfRange ||
                        checked.error().code == RefusalCode::FieldMissing ||
                        checked.error().code == RefusalCode::FileUnreadable;
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
