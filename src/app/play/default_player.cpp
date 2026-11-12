#include "app/play/default_player.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <stop_token>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>

#include "app/play/i18n.h"
#include "app/play/pumped_task.h"
#include "core/file_sync.h"
#include "core/text.h"
#include "core/text_file.h"
#include "core/thread.h"
#include "core/utf8_path.h"
#include "prepare/process.h"
#include "prepare/utf8.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
// Le shell après windows.h.
#include <shlobj.h>
#include <shlwapi.h>

#include <SDL3/SDL_misc.h>
#endif

namespace ajp::app::play {
namespace {

// Les conteneurs vidéo courants, tous lus par FFmpeg. AVI a deux noms dans shared-mime-info.
constexpr std::array<std::string_view, 16> kMimeTypes{
    "video/x-matroska", "video/webm",      "video/mp4",          "video/x-m4v",
    "video/quicktime",  "video/x-msvideo", "video/vnd.avi",      "video/mpeg",
    "video/mp2t",       "video/x-flv",     "video/x-ms-wmv",     "video/ogg",
    "video/3gpp",       "video/x-ogm+ogg", "video/x-theora+ogg", "video/x-ms-asf"};
// La seule table des extensions vidéo (P3.85) : inscription, épisodes, fichiers glissés, boîte « Ouvrir ».
// .ogm (type x-ogm+ogg déclaré), .vob, .asf et .divx : d'anciens fichiers d'anime que FFmpeg lit et que la
// boîte « Ouvrir une vidéo » et « Ouvrir avec » refusaient (B19-10).
constexpr std::array<std::string_view, 19> kExtensions{
    ".mkv", ".webm", ".mp4", ".m4v", ".mov", ".avi", ".mpg", ".mpeg", ".ts",  ".m2ts",
    ".mts", ".flv",  ".wmv", ".ogv", ".3gp", ".ogm", ".vob", ".asf",  ".divx"};

constexpr std::string_view kClasses = "Software\\Classes\\";
constexpr std::string_view kCapabilities = "Software\\AJP\\Capabilities";
constexpr std::string_view kRegisteredApplications = "Software\\RegisteredApplications";

// Une valeur de chaîne d'un .desktop : « \ » s'écrit « \\ ». Les autres échappements (\n, \t, \r,
// \s) ne servent pas : les caractères de contrôle sont refusés avant.
std::string desktop_string(std::string_view text) {
  std::string escaped;
  for (const char c : text) {
    if (c == '\\') escaped += '\\';
    escaped += c;
  }
  return escaped;
}

// L'argument de Exec, entre guillemets : « " », « ` », « $ » et « \ » y prennent une barre oblique
// inverse et « % » se double (code de champ), puis la règle des chaînes s'applique à tout.
std::string exec_argument(std::string_view text) {
  std::string quoted = "\"";
  for (const char c : text) {
    if (c == '"' || c == '`' || c == '$' || c == '\\') quoted += '\\';
    if (c == '%') quoted += '%';
    quoted += c;
  }
  quoted += '"';
  return desktop_string(quoted);
}

std::string_view file_name(std::string_view windows_path) {
  const std::size_t slash = windows_path.find_last_of("\\/");
  return slash == std::string_view::npos ? windows_path : windows_path.substr(slash + 1);
}

[[maybe_unused]] std::string mime_list() {
  std::string text;
  for (const std::string_view mime : kMimeTypes) text += std::format(" {}", mime);
  return text;
}

#if defined(_WIN32)

std::wstring wide(std::string_view utf8) {
  if (utf8.empty()) return {};
  const auto size = static_cast<int>(utf8.size());
  const int count = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), size, nullptr, 0);
  std::wstring text(static_cast<std::size_t>(count), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), size, text.data(), count);
  return text;
}

std::expected<void, Refusal> write_registry(std::span<const RegistryValue> values) {
  for (const RegistryValue& value : values) {
    const std::wstring key = wide(value.key);
    const std::wstring name = wide(value.name);
    const std::wstring data = wide(value.data);
    const LSTATUS status =
        RegSetKeyValueW(HKEY_CURRENT_USER, key.c_str(), value.name.empty() ? nullptr : name.c_str(), REG_SZ,
                        data.c_str(), static_cast<DWORD>((data.size() + 1) * sizeof(wchar_t)));
    if (status != ERROR_SUCCESS)
      return refuse(RefusalCode::FileUnwritable,
                    i18n::tr_format("registry: HKCU\\{} (error {})", value.key, status));
  }
  SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
  return {};
}

// Une clé ou une valeur déjà absente n'est pas une erreur.
bool removed(LSTATUS status) {
  return status == ERROR_SUCCESS || status == ERROR_FILE_NOT_FOUND || status == ERROR_PATH_NOT_FOUND;
}

std::expected<void, Refusal> erase_registry(std::string_view executable) {
  for (const std::string& tree : registry_trees(executable)) {
    const LSTATUS status = RegDeleteTreeW(HKEY_CURRENT_USER, wide(tree).c_str());  // la clé et ses sous-clés
    if (!removed(status))
      return refuse(RefusalCode::FileUnwritable,
                    i18n::tr_format("registry: HKCU\\{} (error {})", tree, status));
  }
  RegDeleteKeyW(HKEY_CURRENT_USER, L"Software\\AJP");  // seulement vide : rien d'autre n'y est à nous
  for (const RegistryValue& value : registry_shared_values()) {
    const LSTATUS status =
        RegDeleteKeyValueW(HKEY_CURRENT_USER, wide(value.key).c_str(), wide(value.name).c_str());
    if (!removed(status))
      return refuse(RefusalCode::FileUnwritable,
                    i18n::tr_format("registry: HKCU\\{} : {} (error {})", value.key, value.name, status));
  }
  SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
  return {};
}

#elif !defined(__APPLE__)

// update-desktop-database recalcule le cache des types du dossier, que certains bureaux lisent
// plutôt que les .desktop ; sans lui, ceux-là attendent leur propre passage. Au mieux : rien ne
// dépend de sa réussite.
void refresh_desktop_database(const std::filesystem::path& dir) {
  for (const char* tool : {"/usr/bin/update-desktop-database", "/usr/local/bin/update-desktop-database"}) {
    std::error_code error;
    if (!std::filesystem::exists(tool, error)) continue;
    const prepare::ProcessSpec spec{
        .program = tool, .args = {prepare::utf8_path(dir)}, .timeout = std::chrono::seconds(15)};
    const bool refreshed = prepare::run_process(spec, [](std::string_view) {}, std::stop_token()).has_value();
    static_cast<void>(refreshed);  // au mieux, voir plus haut
    return;
  }
}

#endif

}  // namespace

std::span<const std::string_view> video_mime_types() {
  return kMimeTypes;
}

std::span<const std::string_view> video_extensions() {
  return kExtensions;
}

const char* video_dialog_pattern() {
  static const std::string pattern = [] {
    std::string joined;
    for (const std::string_view extension : kExtensions) {
      if (!joined.empty()) joined += ';';
      joined += extension.substr(1);
    }
    return joined;
  }();
  return pattern.c_str();
}

std::expected<std::string, Refusal> desktop_entry(const std::filesystem::path& executable,
                                                  bool icon_installed) {
  const std::string path = prepare::utf8_path(executable);
  const bool control = std::ranges::any_of(path, [](char c) {
    const auto byte = static_cast<unsigned char>(c);
    return byte < 0x20 || byte == 0x7F;
  });
  if (path.empty() || control || !prepare::is_utf8(path)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  "lecteur par défaut : le chemin du lecteur ne tient pas dans un fichier .desktop");
  }
  std::string mimes;
  for (const std::string_view mime : kMimeTypes) mimes += std::format("{};", mime);
  return std::format(
      "[Desktop Entry]\n"
      "Type=Application\n"
      "Version=1.5\n"
      "Name={}\n"
      "GenericName=Video Player\n"
      "GenericName[fr]=Lecteur vidéo\n"
      "Comment=Upscaled and interpolated anime in real time\n"
      "Comment[fr]=Anime agrandi et interpolé en temps réel\n"
      "Icon={}\n"
      "StartupWMClass={}\n"
      "TryExec={}\n"
      "Exec={} %f\n"
      "Terminal=false\n"
      "Categories=AudioVideo;Video;Player;\n"
      "MimeType={}\n",
      kAppName, icon_installed ? kAppId : std::string_view("video-x-generic"), kAppId, desktop_string(path),
      exec_argument(path), mimes);
}

std::string without_desktop_id(std::string_view mimeapps, std::string_view id) {
  std::string out;
  out.reserve(mimeapps.size());
  while (!mimeapps.empty()) {
    const std::size_t end = mimeapps.find('\n');
    std::string_view line = mimeapps.substr(0, end);
    mimeapps.remove_prefix(end == std::string_view::npos ? mimeapps.size() : end + 1);
    const std::size_t equal = line.find('=');
    if (equal == std::string_view::npos || line.starts_with('[') || line.starts_with('#')) {
      out.append(line);
      if (end != std::string_view::npos) out += '\n';
      continue;
    }
    // « video/x-matroska=ajp.desktop;vlc.desktop; » : la liste sans la fiche du lecteur.
    std::string kept;
    std::string_view values = line.substr(equal + 1);
    bool removed = false;
    while (!values.empty()) {
      const std::size_t semi = values.find(';');
      const std::string_view value = core::trim(values.substr(0, semi));
      values.remove_prefix(semi == std::string_view::npos ? values.size() : semi + 1);
      if (value == id) {
        removed = true;
      } else if (!value.empty()) {
        kept.append(value);
        kept += ';';
      }
    }
    if (!removed) {
      out.append(line);
    } else if (!kept.empty()) {
      out.append(line.substr(0, equal + 1));
      out += kept;
    } else {
      continue;  // plus rien pour ce type : la ligne part
    }
    if (end != std::string_view::npos) out += '\n';
  }
  return out;
}

std::filesystem::path desktop_file(const PlayerPlace& place) {
  const std::filesystem::path data = place.data.empty() ? place.home / ".local" / "share" : place.data;
  return data / "applications" / std::string(kDesktopId);
}

#if !defined(_WIN32) && !defined(__APPLE__)
// B19-01 : l'icône du lecteur sous <données>/icons/hicolor, où le bureau la cherche par son nom. Elle vient
// de ajp.png à côté de l'exécutable (le paquet Linux la livre) ; sans elle, la fiche garde l'icône générique.
std::filesystem::path icon_file(const PlayerPlace& place) {
  const std::filesystem::path data = place.data.empty() ? place.home / ".local" / "share" : place.data;
  return data / "icons" / "hicolor" / "256x256" / "apps" / (std::string(kAppId) + ".png");
}

bool install_icon(const PlayerPlace& place) {
  namespace fs = std::filesystem;
  std::error_code error;
  const fs::path source = place.executable.parent_path() / "ajp.png";
  if (!fs::is_regular_file(source, error)) return false;
  const fs::path target = icon_file(place);
  if (fs::is_regular_file(target, error) && fs::file_size(target, error) == fs::file_size(source, error))
    return true;
  fs::create_directories(target.parent_path(), error);
  return !error && fs::copy_file(source, target, fs::copy_options::overwrite_existing, error) && !error;
}

// B19-07 : une fiche qui nomme encore un autre exécutable présent n'est pas reprise (deux installations
// qui alternent la réécriraient, avec update-desktop-database, à chaque lancement) ; seule celle dont
// l'exécutable a disparu suit celui-ci.
bool names_other_live_launcher(const std::string& current, const std::string& wanted) {
  const auto try_exec = [](const std::string& text) {
    constexpr std::string_view kKey = "\nTryExec=";
    const std::size_t at = text.find(kKey);
    if (at == std::string::npos) return std::string();
    const std::size_t start = at + kKey.size();
    return text.substr(start, text.find('\n', start) - start);
  };
  const std::string old_launcher = try_exec(current);
  if (old_launcher.empty() || old_launcher == try_exec(wanted)) return false;
  std::string path = old_launcher;
  for (std::size_t at = path.find("\\\\"); at != std::string::npos; at = path.find("\\\\", at + 1))
    path.erase(at, 1);  // « \\ » du fichier : une barre inverse
  std::error_code error;
  return std::filesystem::is_regular_file(core::path_from_utf8(path), error);
}
#endif

#if defined(_WIN32)
// L'exécutable que nomme une commande du registre (« "C:\\x\\ajp.exe" "%1" » ou sans guillemets) existe-t-il
// ?
bool names_other_live_executable(const std::string& command) {
  if (command.empty()) return false;
  std::string executable;
  if (command.front() == '"') {
    const std::size_t end = command.find('"', 1);
    executable = command.substr(1, end == std::string::npos ? std::string::npos : end - 1);
  } else {
    executable = command.substr(0, command.find(' '));
  }
  if (executable.empty()) return false;
  std::error_code error;
  return std::filesystem::is_regular_file(core::path_from_utf8(executable), error);
}
#endif

std::filesystem::path user_data(std::filesystem::path pref) {
  if (pref.empty()) return {};
  if (!pref.has_filename()) pref = pref.parent_path();  // la barre finale de SDL
  return pref.parent_path().parent_path();
}

std::vector<RegistryValue> registry_values(std::string_view executable) {
  const std::string command = std::format("\"{}\" \"%1\"", executable);
  const std::string prog = std::format("{}{}", kClasses, kProgId);
  const std::string app = std::format("{}Applications\\{}", kClasses, file_name(executable));
  std::vector<RegistryValue> values{
      {prog, "", i18n::tr_format("Video ({})", kAppName)},
      {prog + "\\DefaultIcon", "", std::format("\"{}\",0", executable)},
      {prog + "\\shell\\open", "FriendlyAppName", std::string(kAppName)},
      {prog + "\\shell\\open\\command", "", command},
      {app, "FriendlyAppName", std::string(kAppName)},
      {app + "\\shell\\open\\command", "", command},
      {std::string(kCapabilities), "ApplicationName", std::string(kAppName)},
      {std::string(kCapabilities), "ApplicationDescription", "Upscaled and interpolated anime in real time"},
  };
  for (const std::string_view extension : kExtensions) {
    values.push_back({app + "\\SupportedTypes", std::string(extension), ""});
    values.push_back(
        {std::format("{}\\FileAssociations", kCapabilities), std::string(extension), std::string(kProgId)});
    values.push_back({std::format("{}{}\\OpenWithProgids", kClasses, extension), std::string(kProgId), ""});
  }
  values.push_back(
      {std::string(kRegisteredApplications), std::string(kRegisteredName), std::string(kCapabilities)});
  return values;
}

std::vector<std::string> registry_trees(std::string_view executable) {
  return {std::format("{}{}", kClasses, kProgId),
          std::format("{}Applications\\{}", kClasses, file_name(executable)), std::string(kCapabilities)};
}

std::vector<RegistryValue> registry_shared_values() {
  std::vector<RegistryValue> values;
  values.reserve(kExtensions.size() + 1);
  for (const std::string_view extension : kExtensions)
    values.push_back({std::format("{}{}\\OpenWithProgids", kClasses, extension), std::string(kProgId), ""});
  values.push_back({std::string(kRegisteredApplications), std::string(kRegisteredName), ""});
  return values;
}

namespace {
// P4.11 : xdg-mime ne se lance qu'une fois par processus (et après une inscription ou un retrait), plus à
// chaque ouverture de chaîne, « Appliquer » ou épisode.
// A19-04 : la question part sur un fil à elle (xdg-mime peut prendre 10 s, deux fois) ; tant qu'elle n'a
// pas de réponse, le lecteur n'est pas dit « par défaut ». generation écarte la réponse d'une question
// posée avant une inscription ou un retrait.
std::mutex g_default_mutex;           // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::optional<bool> g_default_known;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// Quand elle a été posée : la réponse se repose après ce délai (B19-09). Sous Windows, le réglage se change
// dans la page des paramètres du système, après le clic ; la valeur gardée toute la séance restait fausse.
std::chrono::steady_clock::time_point
    g_default_known_at;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
constexpr std::chrono::seconds kDefaultFresh{20};
bool g_default_asking = false;           // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::uint64_t g_default_generation = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void forget_default_player() {
  const std::scoped_lock lock(g_default_mutex);
  g_default_known.reset();
  g_default_asking = false;
  ++g_default_generation;
}
}  // namespace

#if !defined(_WIN32) && !defined(__APPLE__)
namespace {
// Dans un AppImage l'exécutable est monté à un endroit provisoire : la fiche pointe sur le fichier
// .AppImage lui-même (variable APPIMAGE posée par son lanceur).
std::filesystem::path launcher_of(const PlayerPlace& place) {
  std::filesystem::path launcher = place.executable;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  const char* image = std::getenv("APPIMAGE");  // env: lanceur
  if (image != nullptr && *image != '\0') launcher = image;
  return launcher;
}
}  // namespace
#endif

std::expected<std::string, Refusal> register_player(const PlayerPlace& place) {
  forget_default_player();
#if defined(_WIN32)
  if (auto written = write_registry(registry_values(prepare::utf8_path(place.executable))); !written)
    return std::unexpected(std::move(written.error()));
  return i18n::tr_format(
      "AJP is in \"Open with\". To make it the default: \"Make this the default player\" in Settings > "
      "General > Application, or Settings > Apps > Default apps > {}.",
      kAppName);
#elif defined(__APPLE__)
  static_cast<void>(place);
  return std::string(
      i18n::tr("AJP.app already declares the videos it opens. To make it the default: right-click a video > "
               "Get Info > Open with: AJP > Change All."));
#else
  if (place.home.empty() && place.data.empty())
    return refuse(RefusalCode::FieldMissing, i18n::tr("default player: home folder unknown"));
  auto text = desktop_entry(launcher_of(place), install_icon(place));
  if (!text) return std::unexpected(std::move(text.error()));
  const std::filesystem::path file = desktop_file(place);
  if (auto written = core::write_text_atomically(file, *text); !written)
    return std::unexpected(written.error());
  refresh_desktop_database(file.parent_path());
  return i18n::tr_format(
      "AJP is in \"Open with\" ({}). To make it the default: \"Make this the default player\" in Settings > "
      "General > Application, or xdg-mime default {}{}",
      prepare::utf8_path(file), kDesktopId, mime_list());
#endif
}

bool refresh_registration(const PlayerPlace& place) {
#if defined(_WIN32)
  // Inscrit si le ProgId a sa commande ; à refaire si elle nomme un autre exécutable.
  const std::vector<RegistryValue> values = registry_values(prepare::utf8_path(place.executable));
  const auto command = std::ranges::find_if(values, [](const RegistryValue& value) {
    return value.key.ends_with("AJP.Video\\shell\\open\\command");
  });
  if (command == values.end()) return false;
  const std::wstring key = wide(command->key);
  std::array<wchar_t, 2048> data{};
  DWORD size = static_cast<DWORD>(data.size() * sizeof(wchar_t));
  if (RegGetValueW(HKEY_CURRENT_USER, key.c_str(), nullptr, RRF_RT_REG_SZ, nullptr, data.data(), &size) !=
      ERROR_SUCCESS)
    return false;
  const std::string current = core::to_utf8(std::filesystem::path(std::wstring(data.data())));
  if (current == command->data) return false;
  // Une commande qui nomme encore un autre ajp.exe présent (une version publiée et un dossier de
  // développement) n'est pas reprise : deux installations qui alternent la réécriraient, et l'Explorateur
  // rechargerait ses icônes et ses associations (SHCNE_ASSOCCHANGED), à chaque lancement. Seule une commande
  // dont l'exécutable a disparu suit celui-ci, comme la fiche de Linux.
  if (names_other_live_executable(current)) return false;
  return write_registry(values).has_value();
#elif defined(__APPLE__)
  static_cast<void>(place);
  return false;  // l'Info.plist suit AJP.app où qu'il soit
#else
  // Inscrit si la fiche existe ; à refaire si elle n'est plus celle de cet exécutable.
  const std::filesystem::path file = desktop_file(place);
  std::error_code error;
  if ((place.home.empty() && place.data.empty()) || !std::filesystem::is_regular_file(file, error))
    return false;
  const auto wanted = desktop_entry(launcher_of(place), install_icon(place));
  std::ifstream in(file, std::ios::binary);
  const std::string current{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (!wanted || current == *wanted || names_other_live_launcher(current, *wanted)) return false;
  if (!core::write_text_atomically(file, *wanted)) return false;
  refresh_desktop_database(file.parent_path());
  return true;
#endif
}

std::expected<std::string, Refusal> unregister_player(const PlayerPlace& place) {
  forget_default_player();
#if defined(_WIN32)
  if (auto erased = erase_registry(prepare::utf8_path(place.executable)); !erased)
    return std::unexpected(std::move(erased.error()));
  return std::string(i18n::tr("AJP is no longer in \"Open with\"."));
#elif defined(__APPLE__)
  static_cast<void>(place);
  return std::string(i18n::tr("On macOS the registration follows AJP.app: remove it by trashing the app."));
#else
  if (place.home.empty() && place.data.empty())
    return refuse(RefusalCode::FieldMissing, i18n::tr("default player: home folder unknown"));
  const std::filesystem::path file = desktop_file(place);
  std::error_code error;
  if (!std::filesystem::remove(file, error)) {
    if (error) {
      return refuse(RefusalCode::FileUnwritable,
                    std::format("{} : {}", prepare::utf8_path(file), error.message()));
    }
    return std::string(i18n::tr("AJP was not registered: nothing to remove."));
  }
  std::filesystem::remove(icon_file(place), error);  // l'icône copiée à l'inscription
  // C36-09 : les lignes que xdg-mime default a écrites ne pointent plus vers rien ; retirées.
  if (!place.home.empty()) {
    const std::filesystem::path mimeapps = place.home / ".config" / "mimeapps.list";
    if (auto text = core::read_text_file_if_present(mimeapps); text && *text) {
      if (const std::string cleaned = without_desktop_id(**text, kDesktopId); cleaned != **text)
        std::ignore = core::write_text_atomically(mimeapps, cleaned);
    }
  }
  refresh_desktop_database(file.parent_path());
  return std::string(i18n::tr("AJP is no longer in \"Open with\"."));
#endif
}

std::expected<std::string, Refusal> choose_as_default(const PlayerPlace& place) {
  auto registered = register_player(place);
  if (!registered) return registered;
  return default_once_registered(*registered);
}

std::expected<std::string, Refusal> default_once_registered([[maybe_unused]] const std::string& registered) {
#if defined(_WIN32)
  // Windows 11 ouvre la page du lecteur, avec « Définir par défaut » pour tous ses types ; Windows 10
  // ouvre la liste des applications par défaut.
  if (!SDL_OpenURL(std::format("ms-settings:defaultapps?registeredAppUser={}", kRegisteredName).c_str()))
    return registered;
  return i18n::tr_format(
      "Settings opened on {}: \"Set default\" at the top of the page, or one type at a time.", kAppName);
#elif defined(__APPLE__)
  return registered;
#else
  // Aucune recherche dans PATH (prepare/process.h) : les emplacements des paquets xdg-utils.
  for (const char* tool : {"/usr/bin/xdg-mime", "/usr/local/bin/xdg-mime", "/bin/xdg-mime"}) {
    std::error_code error;
    if (!std::filesystem::exists(tool, error)) continue;
    prepare::ProcessSpec spec{
        .program = tool, .args = {"default", std::string(kDesktopId)}, .timeout = std::chrono::seconds(30)};
    for (const std::string_view mime : kMimeTypes) spec.args.emplace_back(mime);
    std::string last;
    auto run = prepare::run_process(spec, [&](std::string_view line) { last = line; }, std::stop_token());
    if (!run) return std::unexpected(std::move(run.error()));
    if (run->kind != prepare::ExitKind::Exited || run->code != 0) {
      return refuse(RefusalCode::FileUnwritable, std::format("xdg-mime : {}{}{}", prepare::describe(*run),
                                                             last.empty() ? "" : " : ", last));
    }
    return std::string(i18n::tr("AJP is now the video player (mkv, mp4, webm, avi, mov…)."));
  }
  return refuse(RefusalCode::LibraryUnavailable,
                i18n::tr("xdg-mime not found (xdg-utils package): right-click a video > Open with > AJP and "
                         "tick \"always\""));
#endif
}

bool default_handler_is_ajp(std::string_view query_output) {
  while (!query_output.empty() &&
         (query_output.back() == '\n' || query_output.back() == '\r' || query_output.back() == ' '))
    query_output.remove_suffix(1);
  return query_output == kDesktopId;
}

namespace {

// Les deux conteneurs que l'on ouvre le plus : AJP est le lecteur par défaut s'il l'est pour les deux. La
// seconde question ne part que si la première dit oui (A19-05).
#if defined(_WIN32)
constexpr std::array<std::string_view, 2> kAskedExtensions{".mkv", ".mp4"};

// Le choix de l'utilisateur, que Windows garde sous FileExts\<extension>\UserChoice (ProgId).
std::string user_choice(std::string_view extension) {
  const std::wstring key = wide(std::format(
      "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\{}\\UserChoice", extension));
  std::array<wchar_t, 256> data{};
  DWORD size = static_cast<DWORD>(data.size() * sizeof(wchar_t));
  if (RegGetValueW(HKEY_CURRENT_USER, key.c_str(), L"ProgId", RRF_RT_REG_SZ, nullptr, data.data(), &size) !=
      ERROR_SUCCESS)
    return {};
  return core::to_utf8(std::filesystem::path(std::wstring(data.data())));
}

// B34-09 : l'exécutable que Windows associe à l'extension pour « open », par la règle du système (UserChoice,
// UserChoiceLatest des Windows 11 récents, règles de la machine) et non par la seule clé UserChoice.
std::string associated_executable(std::string_view extension) {
  const std::wstring name = wide(std::string(extension));
  std::array<wchar_t, 1024> data{};
  DWORD size = static_cast<DWORD>(data.size());
  if (FAILED(AssocQueryStringW(ASSOCF_NOTRUNCATE, ASSOCSTR_EXECUTABLE, name.c_str(), L"open", data.data(),
                               &size)))
    return {};
  return core::to_utf8(std::filesystem::path(std::wstring(data.data())));
}
#elif !defined(__APPLE__)
constexpr std::array<std::string_view, 2> kAskedMimes{"video/x-matroska", "video/mp4"};

bool xdg_default_is_ajp(const char* tool, std::string_view mime) {
  const prepare::ProcessSpec spec{
      .program = tool, .args = {"query", "default", std::string(mime)}, .timeout = std::chrono::seconds(10)};
  // Toutes les lignes (stdout et stderr mêlés) : un avertissement de Qt ou de KDE écrit après l'identifiant
  // ne doit pas faire comparer autre chose que la fiche .desktop.
  bool found = false;
  auto run = prepare::run_process(
      spec, [&](std::string_view line) { found = found || default_handler_is_ajp(line); }, std::stop_token());
  return run && run->kind == prepare::ExitKind::Exited && run->code == 0 && found;
}
#endif

bool query_default_player() {
#if defined(_WIN32)
  return std::ranges::all_of(kAskedExtensions, [](std::string_view extension) {
    return user_choice_is_ajp(user_choice(extension)) || executable_is_ajp(associated_executable(extension));
  });
#elif defined(__APPLE__)
  return false;  // le Finder ne le dit pas sans LaunchServices : la case suffit
#else
  for (const char* tool : {"/usr/bin/xdg-mime", "/usr/local/bin/xdg-mime", "/bin/xdg-mime"}) {
    std::error_code error;
    if (!std::filesystem::exists(tool, error)) continue;
    return std::ranges::all_of(kAskedMimes,
                               [tool](std::string_view mime) { return xdg_default_is_ajp(tool, mime); });
  }
  return false;
#endif
}
}  // namespace

bool user_choice_is_ajp(std::string_view prog_id) {
  // « Ouvrir avec > Toujours » écrit « Applications\ajp.exe » et non notre ProgId (B19-02).
  constexpr std::string_view kApplication = "Applications\\ajp.exe";
  if (prog_id == kProgId) return true;
  return prog_id.size() == kApplication.size() &&
         std::ranges::equal(prog_id, kApplication,
                            [](char a, char b) { return core::ascii_lower(a) == core::ascii_lower(b); });
}

bool executable_is_ajp(std::string_view executable) {
  const std::size_t cut = executable.find_last_of("/\\");
  const std::string_view name = cut == std::string_view::npos ? executable : executable.substr(cut + 1);
  return core::ascii_lower(name) == "ajp.exe";
}

bool is_default_player(const PlayerPlace& place) {
  static_cast<void>(place);
  const std::scoped_lock lock(g_default_mutex);
  const bool fresh = g_default_known && std::chrono::steady_clock::now() - g_default_known_at < kDefaultFresh;
  if (fresh) return *g_default_known;
  const bool stale_answer =
      g_default_known.value_or(false);  // celle d'avant, le temps que la nouvelle vienne
  if (!g_default_asking) {
    g_default_asking = true;
    const std::uint64_t generation = g_default_generation;
    // Par start_thread, gardé par abandon_thread : la question (xdg-mime, 10 s au plus, deux fois) ne retient
    // pas la sortie et n'est plus un thread détaché hors de core/thread.h (B30-03, B33-08).
    const auto finished = std::make_shared<std::atomic<bool>>(false);
    auto started =
        start_thread("default-player", ThreadReport([](const Refusal&) {}),
                     [generation, finished](const std::stop_token&, const ThreadReport&) {
                       bool answer = false;
                       try {
                         answer = query_default_player();
                       } catch (...) {  // NOLINT(bugprone-empty-catch) : sans réponse, pas par défaut
                       }
                       {
                         const std::scoped_lock done(g_default_mutex);
                         if (generation == g_default_generation) {
                           g_default_known = answer;
                           g_default_known_at = std::chrono::steady_clock::now();
                           g_default_asking = false;
                         }
                       }
                       finished->store(true);
                     });
    if (started) {
      abandon_thread(std::move(*started), [finished] { return finished->load(); });
    } else {  // pas de fil : la question n'est pas posée, le lecteur n'est pas dit par défaut
      g_default_known = false;
      g_default_known_at = std::chrono::steady_clock::now();
      g_default_asking = false;
    }
  }
  return stale_answer;
}

void leave_lone_console() {
#ifdef _WIN32
  // ajp.exe est une application fenêtrée : un double clic n'ouvre plus de console (elle apparaissait un
  // instant puis se refermait, le lecteur semblait s'ouvrir deux fois). Lancé d'un terminal, le lecteur
  // écrit dans la console de ce terminal ; une sortie redirigée (tube, fichier) reste la sienne.
  const auto missing = [](DWORD which) {
    const HANDLE handle = GetStdHandle(which);
    return handle == nullptr || handle == INVALID_HANDLE_VALUE;
  };
  const bool out = missing(STD_OUTPUT_HANDLE);
  const bool err = missing(STD_ERROR_HANDLE);
  if ((out || err) && AttachConsole(ATTACH_PARENT_PROCESS) != 0) {
    FILE* reopened = nullptr;
    if (out && freopen_s(&reopened, "CONOUT$", "w", stdout) == 0)
      SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stdout))));
    if (err && freopen_s(&reopened, "CONOUT$", "w", stderr) == 0)
      SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stderr))));
  }
  // Le journal est en UTF-8. La page de code est celle de la console, que le terminal parent garde après
  // nous : elle lui est rendue à la sortie (A19-11).
  if (GetConsoleWindow() != nullptr) {
    static UINT previous = GetConsoleOutputCP();
    if (previous != CP_UTF8 && SetConsoleOutputCP(CP_UTF8) != 0)
      std::atexit([] { SetConsoleOutputCP(previous); });
  }
#endif
}

}  // namespace ajp::app::play
