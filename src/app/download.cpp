#include "app/download.h"

#include <array>
#include <condition_variable>
#include <curl/curl.h>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <utility>

#include "app/data_paths.h"
#include "core/model_index.h"

namespace ajp::app {
namespace {

namespace fs = std::filesystem;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Ce qu'un essai laisse à la boucle des essais, quand il ne se termine pas par un refus définitif.
enum class Verdict : std::uint8_t {
  Complete,  // le .part a la taille attendue
  Retry,     // coupure, délai, erreur passagère du serveur : on reprend le .part
  Restart,   // le serveur ne sait pas reprendre à cet endroit : on repart de zéro
};

struct Outcome {
  Verdict verdict = Verdict::Retry;
  std::string error;  // cause de l'essai manqué, citée si tous les essais échouent
};

// État d'un essai, partagé avec les rappels de libcurl (même thread).
struct Transfer {
  std::ofstream& out;
  const DownloadOptions& options;
  std::stop_token stop;
  std::uint64_t received = 0;  // octets dans le .part
  std::uint64_t size = 0;
  std::uint64_t next_report = 0;
  bool oversize = false;
  bool write_failed = false;
};

std::size_t on_data(char* data, std::size_t /*one*/, std::size_t count, void* user) {
  auto& transfer = *static_cast<Transfer*>(user);
  if (count > transfer.size - transfer.received) {
    transfer.oversize = true;
    return 0;  // tout autre nombre que count interrompt le transfert
  }
  transfer.out.write(data, static_cast<std::streamsize>(count));
  if (!transfer.out) {
    transfer.write_failed = true;
    return 0;
  }
  transfer.received += count;
  return count;
}

void report(Transfer& transfer) {
  if (!transfer.options.on_progress) return;
  transfer.options.on_progress(DownloadProgress{.received = transfer.received, .total = transfer.size});
  transfer.next_report = transfer.received + transfer.options.progress_step;
}

// libcurl l'appelle pendant la connexion et le transfert, et au moins une fois par seconde à l'arrêt.
int on_tick(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
  auto& transfer = *static_cast<Transfer*>(user);
  if (transfer.stop.stop_requested()) return 1;
  if (transfer.received >= transfer.next_report) report(transfer);
  return 0;
}

std::unexpected<Refusal> cancelled(const DownloadRequest& request) {
  return refuse(RefusalCode::Cancelled, std::format("téléchargement de {} annulé", request.url));
}

std::expected<void, Refusal> remove_part(const fs::path& part) {
  std::error_code error;
  fs::remove(part, error);
  if (error) return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", utf8(part), error.message()));
  return {};
}

// Retrait sans refus : le refus en cours dit déjà pourquoi le .part ne sert plus.
void discard(const fs::path& part) {
  std::error_code ignored;
  fs::remove(part, ignored);
}

// Octets déjà reçus dans le .part ; un .part plus long que l'attendu vient d'un autre fichier et
// ne peut plus servir : il est supprimé.
std::expected<std::uint64_t, Refusal> resumable_size(const fs::path& part, std::uint64_t size) {
  std::error_code error;
  const std::uintmax_t existing = fs::file_size(part, error);
  if (error) return 0;
  if (existing <= size) return static_cast<std::uint64_t>(existing);
  if (auto r = remove_part(part); !r) return std::unexpected(r.error());
  return 0;
}

std::expected<void, Refusal> check_options(const DownloadRequest& request, const DownloadOptions& options) {
  if (!core::limits::kDownloadSize.contains(static_cast<std::int64_t>(request.size))) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("{} : taille {} hors de l'index", request.url, request.size));
  }
  if (options.attempts < 1 || options.progress_step == 0 || options.connect_timeout.count() <= 0 ||
      options.stall_time.count() <= 0 || options.retry_delay.count() < 0 || options.protocols.empty()) {
    return refuse(RefusalCode::FieldOutOfRange, "options de téléchargement incohérentes");
  }
  if (request.part_file.empty() || request.url.empty()) {
    return refuse(RefusalCode::FieldMissing, "téléchargement sans URL ou sans fichier .part");
  }
  return {};
}

std::expected<void, Refusal> check_space(const fs::path& part, std::uint64_t remaining) {
  std::error_code error;
  const fs::space_info space = fs::space(part.parent_path(), error);
  if (error)
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} : {}", utf8(part.parent_path()), error.message()));
  if (space.available < remaining) {
    return refuse(RefusalCode::DiskFull, std::format("{} : {} octets à recevoir, {} libres",
                                                     utf8(part.parent_path()), remaining, space.available));
  }
  return {};
}

// Les réponses où réessayer a un sens : délai, trop de demandes, panne passagère du serveur.
bool transient_http(long status) {
  return status == 408 || status == 429 || status >= 500;
}

// Traduit la fin d'un essai : verdict, ou refus définitif.
std::expected<Outcome, Refusal> judge(CURLcode code, CURL* easy, const Transfer& transfer,
                                      const char* error_text, const DownloadRequest& request) {
  const std::string cause = error_text[0] == '\0'
                                ? std::string(curl_easy_strerror(code))
                                : std::format("{} ({})", curl_easy_strerror(code), error_text);
  switch (code) {
    case CURLE_OK:
      if (transfer.received == transfer.size) return Outcome{.verdict = Verdict::Complete};
      return Outcome{.error =
                         std::format("connexion close à {} octets sur {}", transfer.received, transfer.size)};
    case CURLE_ABORTED_BY_CALLBACK:
      return cancelled(request);
    case CURLE_WRITE_ERROR:  // on_data : .part impossible à écrire, ou plus d'octets que l'index
      if (transfer.write_failed) return refuse(RefusalCode::FileUnwritable, utf8(request.part_file));
      return refuse(
          RefusalCode::HashMismatch,
          std::format("{} : le serveur envoie plus que les {} octets de l'index", request.url, request.size));
    case CURLE_RANGE_ERROR:
      return Outcome{.verdict = Verdict::Restart, .error = "reprise refusée par le serveur"};
    case CURLE_HTTP_RETURNED_ERROR: {
      long status = 0;
      curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
      if (status == 416) return Outcome{.verdict = Verdict::Restart, .error = "HTTP 416"};
      if (transient_http(status)) return Outcome{.error = std::format("HTTP {}", status)};
      return refuse(RefusalCode::DownloadFailed, std::format("{} : HTTP {}", request.url, status));
    }
    case CURLE_UNSUPPORTED_PROTOCOL:
    case CURLE_URL_MALFORMAT:
    case CURLE_PEER_FAILED_VERIFICATION:
    case CURLE_SSL_CACERT_BADFILE:
    case CURLE_TOO_MANY_REDIRECTS:
      return refuse(RefusalCode::DownloadFailed, std::format("{} : {}", request.url, cause));
    default:
      return Outcome{.error = cause};
  }
}

using Easy = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>;

// Chaque option doit être acceptée : une liste de protocoles refusée (libcurl sans TLS, par
// exemple) laisserait sinon tous les protocoles permis, sans rien dire.
//
// Pas de CURLOPT_MAXFILESIZE : libcurl 8.5 (Ubuntu 24.04, celui de la CI) l'applique à un 200
// complet (serveur sans reprise) avant de rendre CURLE_RANGE_ERROR, qui fait repartir de zéro.
// on_data borne déjà le .part à la taille de l'index.
std::expected<void, Refusal> configure(CURL* easy, const DownloadRequest& request,
                                       const DownloadOptions& options, Transfer& transfer,
                                       std::uint64_t offset, char* error_text) {
  CURLcode code = CURLE_OK;
  const char* failed = "";
  const auto set = [&](CURLoption option, const char* name, auto value) {
    if (code != CURLE_OK) return;
    code = curl_easy_setopt(easy, option, value);
    failed = name;
  };
  set(CURLOPT_URL, "CURLOPT_URL", request.url.c_str());
  set(CURLOPT_PROTOCOLS_STR, "CURLOPT_PROTOCOLS_STR", options.protocols.c_str());
  set(CURLOPT_REDIR_PROTOCOLS_STR, "CURLOPT_REDIR_PROTOCOLS_STR", options.protocols.c_str());
  set(CURLOPT_FOLLOWLOCATION, "CURLOPT_FOLLOWLOCATION", 1L);
  set(CURLOPT_MAXREDIRS, "CURLOPT_MAXREDIRS", 5L);
  set(CURLOPT_FAILONERROR, "CURLOPT_FAILONERROR", 1L);
  set(CURLOPT_NOSIGNAL, "CURLOPT_NOSIGNAL", 1L);  // aucun signal : les délais restent sûrs entre threads
  set(CURLOPT_ERRORBUFFER, "CURLOPT_ERRORBUFFER", error_text);
  set(CURLOPT_CONNECTTIMEOUT_MS, "CURLOPT_CONNECTTIMEOUT_MS",
      static_cast<long>(options.connect_timeout.count()));
  set(CURLOPT_LOW_SPEED_LIMIT, "CURLOPT_LOW_SPEED_LIMIT", static_cast<long>(options.stall_rate));
  set(CURLOPT_LOW_SPEED_TIME, "CURLOPT_LOW_SPEED_TIME", static_cast<long>(options.stall_time.count()));
  set(CURLOPT_RESUME_FROM_LARGE, "CURLOPT_RESUME_FROM_LARGE", static_cast<curl_off_t>(offset));
  set(CURLOPT_WRITEFUNCTION, "CURLOPT_WRITEFUNCTION", &on_data);
  set(CURLOPT_WRITEDATA, "CURLOPT_WRITEDATA", &transfer);
  set(CURLOPT_NOPROGRESS, "CURLOPT_NOPROGRESS", 0L);
  set(CURLOPT_XFERINFOFUNCTION, "CURLOPT_XFERINFOFUNCTION", &on_tick);
  set(CURLOPT_XFERINFODATA, "CURLOPT_XFERINFODATA", &transfer);
  if (code != CURLE_OK) {
    return refuse(RefusalCode::DownloadFailed, std::format("{} : curl_easy_setopt({}) : {}", request.url,
                                                           failed, curl_easy_strerror(code)));
  }
  return {};
}

// Un essai : reprend le .part à sa taille actuelle et le complète autant que possible.
std::expected<Outcome, Refusal> attempt(const DownloadRequest& request, const DownloadOptions& options,
                                        std::stop_token stop, std::uint64_t offset) {
  std::ofstream out(request.part_file, std::ios::binary | std::ios::app);
  if (!out) return refuse(RefusalCode::FileUnwritable, utf8(request.part_file));
  const Easy easy(curl_easy_init(), &curl_easy_cleanup);
  if (!easy) return refuse(RefusalCode::DownloadFailed, "curl_easy_init");

  Transfer transfer{.out = out,
                    .options = options,
                    .stop = std::move(stop),
                    .received = offset,
                    .size = request.size,
                    .next_report = offset};
  std::array<char, CURL_ERROR_SIZE> error_text{};
  if (auto r = configure(easy.get(), request, options, transfer, offset, error_text.data()); !r) {
    return std::unexpected(r.error());
  }
  const CURLcode code = curl_easy_perform(easy.get());
  out.close();
  if (!out && !transfer.write_failed) return refuse(RefusalCode::FileUnwritable, utf8(request.part_file));
  auto outcome = judge(code, easy.get(), transfer, error_text.data(), request);
  if (outcome && outcome->verdict == Verdict::Complete) report(transfer);
  return outcome;
}

// Attend delay, ou moins si stop est demandé entre-temps. Rend false si stop est demandé.
bool pause(std::chrono::milliseconds delay, const std::stop_token& stop) {
  std::mutex mutex;
  std::condition_variable_any wake;
  std::unique_lock lock(mutex);
  return !wake.wait_for(lock, stop, delay, [] { return false; }) && !stop.stop_requested();
}

}  // namespace

std::expected<Downloader, Refusal> Downloader::create() {
  if (const CURLcode code = curl_global_init(CURL_GLOBAL_DEFAULT); code != CURLE_OK) {
    return refuse(RefusalCode::DownloadFailed,
                  std::format("curl_global_init : {}", curl_easy_strerror(code)));
  }
  Downloader downloader;
  downloader.owner_ = true;
  return downloader;
}

Downloader::Downloader(Downloader&& other) noexcept : owner_(std::exchange(other.owner_, false)) {}

Downloader& Downloader::operator=(Downloader&& other) noexcept {
  if (this != &other) {
    if (owner_) curl_global_cleanup();
    owner_ = std::exchange(other.owner_, false);
  }
  return *this;
}

Downloader::~Downloader() {
  if (owner_) curl_global_cleanup();
}

std::expected<void, Refusal> Downloader::fetch(const DownloadRequest& request, const DownloadOptions& options,
                                               const std::stop_token& stop) const {
  if (auto r = check_options(request, options); !r) return r;
  std::error_code error;
  fs::create_directories(request.part_file.parent_path(), error);
  if (error)
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} : {}", utf8(request.part_file), error.message()));

  for (int round = 1;; ++round) {
    if (stop.stop_requested()) return cancelled(request);
    auto offset = resumable_size(request.part_file, request.size);
    if (!offset) return std::unexpected(offset.error());
    if (*offset == request.size) {  // déjà complet : un lancement précédent l'a fini
      if (options.on_progress)
        options.on_progress(DownloadProgress{.received = *offset, .total = request.size});
      return {};
    }
    if (auto r = check_space(request.part_file, request.size - *offset); !r) return r;
    auto outcome = attempt(request, options, stop, *offset);
    if (!outcome) {
      if (outcome.error().code == RefusalCode::HashMismatch)
        discard(request.part_file);  // ne peut plus servir
      return std::unexpected(outcome.error());
    }
    if (outcome->verdict == Verdict::Complete) return {};
    if (outcome->verdict == Verdict::Restart) {
      if (auto r = remove_part(request.part_file); !r) return r;
    }
    if (round >= options.attempts) {
      return refuse(RefusalCode::DownloadFailed, std::format("{} : {} essais sans succès ; dernier : {}",
                                                             request.url, options.attempts, outcome->error));
    }
    if (!pause(options.retry_delay * round, stop)) return cancelled(request);
  }
}

}  // namespace ajp::app
