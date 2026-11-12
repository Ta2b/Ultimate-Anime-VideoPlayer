// L'état des mises à jour (<données>/update.toml) : lecture, réécriture sans perdre les clés d'une version
// plus récente, et les vérifications du fil de fond et du bouton « Check now », qui ne tiennent le verrou que
// pour lire puis réécrire le fichier (B26-09, B30-02).
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include "app/play/updater.h"
#include "core/file_sync.h"
#include "core/log.h"
#include "core/text.h"
#include "core/update_manifest.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

}  // namespace

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

namespace {

// Les clés révoquées qu'un update.toml abîmé (tronqué, édité à la main) laisse encore lire : toute suite de
// 64 chiffres hexadécimaux entre guillemets, dans la liste après « revoked_keys ». Une clé révoquée ne
// redevient pas crue parce que le fichier est illisible (B26-10).
std::vector<std::string> salvage_revoked_keys(std::string_view text) {
  std::vector<std::string> keys;
  const std::size_t name = text.find("revoked_keys");
  if (name == std::string_view::npos) return keys;
  text.remove_prefix(name);
  const std::size_t open = text.find('[');
  if (open == std::string_view::npos) return keys;
  text.remove_prefix(open + 1);
  if (const std::size_t close = text.find(']'); close != std::string_view::npos) text = text.substr(0, close);
  constexpr std::size_t kQuoted = 66;  // 64 chiffres et deux guillemets
  for (std::size_t i = 0; i + kQuoted <= text.size();) {
    if (text[i] != '"' || text[i + kQuoted - 1] != '"') {
      ++i;
      continue;
    }
    std::string key = core::ascii_lower(text.substr(i + 1, kQuoted - 2));
    const bool hex =
        std::ranges::all_of(key, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
    if (hex && std::ranges::find(keys, key) == keys.end()) keys.push_back(std::move(key));
    i += hex ? kQuoted : 1;
  }
  return keys;
}

}  // namespace

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
  const std::string text = read_all(file);
  auto state = parse_update_state(text);
  if (state) {
    forget_installed_announce(*state, installed_version());
    return *state;
  }
  UpdateState salvaged;
  salvaged.revoked_keys = salvage_revoked_keys(text);
  return salvaged;
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
  if (!previous.empty() && !toml::parse(previous)) {
    // Un fichier illisible va être remplacé : sa copie reste à côté, pour qui voudrait y regarder (B26-10).
    fs::path backup = file;
    backup += ".bak";
    if (const auto kept = core::write_text_atomically(backup, previous); !kept)
      log_channel("app")->warn("update.toml.bak : {}", kept.error().detail);
  }
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

namespace {

// L'état du fichier, lu sous le verrou ; le réseau, lui, ne le tient jamais (B26-09, B30-02) : « Check now »
// pendant une vérification de fond, ou la fermeture du lecteur, n'attendent pas quarante secondes de curl.
UpdateState load_locked(const fs::path& file) {
  const std::scoped_lock lock(update_state_mutex());
  return load_update_state(file);
}

// Ce qu'une vérification a changé (after par rapport à before) se reporte sur l'état du fichier tel qu'il est
// maintenant : une autre vérification, ou une clé révoquée entre-temps, n'est pas écrasée.
void adopt_check(UpdateState& current, const UpdateState& before, const UpdateState& after) {
  if (after.last_check_unix != before.last_check_unix) current.last_check_unix = after.last_check_unix;
  if (after.last_attempt_unix != before.last_attempt_unix)
    current.last_attempt_unix = after.last_attempt_unix;
  if (after.latest_version != before.latest_version) current.latest_version = after.latest_version;
  if (after.notes_url != before.notes_url) current.notes_url = after.notes_url;
  if (after.last_error != before.last_error) current.last_error = after.last_error;
  for (const std::string& key : after.revoked_keys)
    if (std::ranges::find(current.revoked_keys, key) == current.revoked_keys.end())
      current.revoked_keys.push_back(key);
}

std::expected<void, Refusal> keep_check(const fs::path& file, const UpdateState& before,
                                        const UpdateState& after) {
  return change_update_state(file, [&](UpdateState& current) { adopt_check(current, before, after); });
}

}  // namespace

void check_updates_if_due(const fs::path& file, UpdateMode mode, UpdateFrequency frequency, bool may_check,
                          std::int64_t now_unix, const std::stop_token& stop) {
  check_updates_if_due(file, mode, frequency, may_check, now_unix, stop, UpdateFetch(fetch_https));
}

void check_updates_if_due(const fs::path& file, UpdateMode mode, UpdateFrequency frequency, bool may_check,
                          std::int64_t now_unix, const std::stop_token& stop, const UpdateFetch& get) {
  if (!may_check) return;
  const UpdateState before = load_locked(file);
  if (!update_due(before, mode, frequency, now_unix)) return;
  UpdateState after = before;
  const auto message = run_update_check(after, stop, get, now_unix);
  if (const auto kept = keep_check(file, before, after); !kept)
    log_channel("app")->warn("update.toml : {}", kept.error().detail);
  log_channel("app")->info("mise à jour : {}",  // i18n-ok : journal
                           message ? *message : message.error().detail);
}

std::expected<std::string, Refusal> check_updates_now(const fs::path& file, const std::stop_token& stop) {
  return check_updates_now(file, stop, UpdateFetch(fetch_https));
}

std::expected<std::string, Refusal> check_updates_now(const fs::path& file, const std::stop_token& stop,
                                                      const UpdateFetch& get) {
  const UpdateState before = load_locked(file);
  UpdateState after = before;
  auto message = run_update_check(after, stop, get, unix_now());
  const auto kept = keep_check(file, before, after);
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

}  // namespace ajp::app::play
