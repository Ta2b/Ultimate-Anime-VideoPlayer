#include "app/play/resource_usage.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stop_token>
#include <string_view>
#include <system_error>

#include "app/play/placebo_screen.h"
#include "core/log.h"
#include "core/thread.h"

#ifdef _WIN32
#include <windows.h>
// windows.h d'abord : psapi.h en dépend.
#include <psapi.h>
#include <tlhelp32.h>
#endif

namespace ajp::app::play {

namespace {

#ifdef __linux__
// « VmRSS:   812345 kB » : la valeur en octets ; rien si le champ manque.
std::optional<std::uint64_t> status_kib(std::string_view text, std::string_view field) {
  const auto at = text.find(field);
  if (at == std::string_view::npos) return std::nullopt;
  std::uint64_t value = 0;
  bool digits = false;
  for (std::size_t i = at + field.size(); i < text.size() && text[i] != '\n'; ++i) {
    if (text[i] < '0' || text[i] > '9') {
      if (digits) break;
      continue;
    }
    digits = true;
    value = value * 10 + static_cast<std::uint64_t>(text[i] - '0');
  }
  if (!digits) return std::nullopt;
  return value;
}
#endif

}  // namespace

ProcessResources process_resources() {
  ProcessResources out;
#ifdef __linux__
  std::ifstream in("/proc/self/status");
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (const auto rss = status_kib(text, "VmRSS:")) out.resident_bytes = *rss * 1024;
  if (const auto threads = status_kib(text, "Threads:")) out.threads = static_cast<int>(*threads);
  std::error_code error;
  int fds = 0;
  for (std::filesystem::directory_iterator it("/proc/self/fd", error), end; !error && it != end;
       it.increment(error))
    ++fds;
  if (!error) out.handles = fds;
#elif defined(_WIN32)
  // C34-14 : la charge privée, pas l'ensemble de travail, que Windows rogne (fenêtre réduite, pression
  // mémoire) : une fuite lente s'y cachait entre deux relevés de la règle « en hausse de 50 % ».
  PROCESS_MEMORY_COUNTERS_EX counters{};
  auto* plain = reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters);  // NOLINT(*-reinterpret-cast)
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), plain, sizeof(counters)))
    out.resident_bytes = counters.PrivateUsage;
  DWORD handles = 0;
  if (GetProcessHandleCount(GetCurrentProcess(), &handles)) out.handles = static_cast<int>(handles);
  const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snapshot != INVALID_HANDLE_VALUE) {
    THREADENTRY32 entry{.dwSize = sizeof(THREADENTRY32)};
    int threads = 0;
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry))
      if (entry.th32OwnerProcessID == GetCurrentProcessId()) ++threads;
    CloseHandle(snapshot);
    out.threads = threads;
  }
#endif
  return out;
}

std::string describe(const ProcessResources& resources) {
  std::string out;
  const auto add = [&](const std::string& part) { out += out.empty() ? part : ", " + part; };
  constexpr double kMib = 1024.0 * 1024.0;
  if (resources.resident_bytes)
    add(std::format("mémoire {:.0f} Mio", static_cast<double>(*resources.resident_bytes) / kMib));  // i18n-ok
  if (resources.threads) add(std::format("{} threads", *resources.threads));
  if (resources.handles) add(std::format("{} descripteurs", *resources.handles));  // i18n-ok : journal
  return out.empty() ? std::string("inconnues") : out;                             // i18n-ok : journal
}

namespace {

std::mutex& reader_mutex() {
  static std::mutex mutex;
  return mutex;
}

DeviceMemoryReader& reader_slot() {
  static DeviceMemoryReader reader;
  return reader;
}

std::string device_memory_text() {
  const std::scoped_lock lock(reader_mutex());  // la chaîne ne se ferme pas pendant la lecture
  if (!reader_slot()) return {};
  const auto memory = reader_slot()();
  if (!memory) return {};
  constexpr double kMib = 1024.0 * 1024.0;
  return std::format(", mémoire vidéo {:.0f} / {:.0f} Mio",  // i18n-ok : journal
                     static_cast<double>(memory->first) / kMib, static_cast<double>(memory->second) / kMib);
}

}  // namespace

namespace {
// Chaque chaîne ouverte : le relevé suivant devient la référence de la dérive (C33-02). L'accueil, relevé
// avant la première chaîne, ne fait plus crier à la fuite pendant toute la séance.
std::atomic<std::uint64_t>& chains_opened() {
  static std::atomic<std::uint64_t> opened{0};
  return opened;
}
}  // namespace

void set_device_memory_reader(DeviceMemoryReader reader) {
  if (reader) chains_opened().fetch_add(1, std::memory_order_relaxed);
  const std::scoped_lock lock(reader_mutex());
  reader_slot() = std::move(reader);
}

std::string resource_drift(const ProcessResources& first, const ProcessResources& now) {
  constexpr double kDrift = 1.5;
  std::string out;
  if (first.resident_bytes && now.resident_bytes && *first.resident_bytes > 0 &&
      static_cast<double>(*now.resident_bytes) > kDrift * static_cast<double>(*first.resident_bytes))
    out += std::format("mémoire résidente {} Mio contre {} au premier relevé",  // i18n-ok : journal
                       *now.resident_bytes >> 20U, *first.resident_bytes >> 20U);
  if (first.handles && now.handles && *first.handles > 0 &&
      static_cast<double>(*now.handles) > kDrift * static_cast<double>(*first.handles))
    out += std::format("{}{} descripteurs contre {} au premier relevé",  // i18n-ok : journal
                       out.empty() ? "" : ", ", *now.handles, *first.handles);
  // Les fils : le seul signal d'un fil abandonné ou oublié (C33-02).
  if (first.threads && now.threads && *first.threads > 0 &&
      static_cast<double>(*now.threads) > kDrift * static_cast<double>(*first.threads))
    out += std::format("{}{} threads contre {} au premier relevé",  // i18n-ok : journal
                       out.empty() ? "" : ", ", *now.threads, *first.threads);
  return out;
}

ResourcesAfterTeardown::~ResourcesAfterTeardown() {
  screen_.release_ring();
  log_channel("app")->info("ressources après la fermeture de la chaîne : {}",  // i18n-ok : journal
                           describe(process_resources()));
}

std::jthread start_resource_log() {
  auto started = start_thread(
      "resources", ThreadReport([](const Refusal& refused) {
        log_channel("app")->warn("ressources : {}", refused.detail);  // i18n-ok : journal
      }),
      [](const std::stop_token& stop, const ThreadReport&) {
        std::mutex mutex;
        std::condition_variable_any wake;
        std::optional<ProcessResources> first;
        std::uint64_t chain = chains_opened().load(std::memory_order_relaxed);
        while (true) {
          std::unique_lock lock(mutex);
          if (wake.wait_for(lock, stop, std::chrono::minutes(5), [] { return false; }) ||
              stop.stop_requested())
            return;
          const ProcessResources now = process_resources();
          if (const std::uint64_t opened = chains_opened().load(std::memory_order_relaxed); opened != chain) {
            chain = opened;
            first.reset();
          }
          if (!first) first = now;
          log_channel("app")->info("ressources : {}{}", describe(now), device_memory_text());  // i18n-ok
          if (const std::string drift = resource_drift(*first, now); !drift.empty())
            log_channel("app")->warn("ressources en hausse : {}", drift);  // i18n-ok : journal
        }
      });
  if (started) return std::move(*started);
  return {};
}

}  // namespace ajp::app::play
