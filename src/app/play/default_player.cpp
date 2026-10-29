#include "app/play/default_player.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <format>
#include <fstream>
#include <stop_token>
#include <system_error>
#include <utility>

#include "prepare/process.h"
#include "prepare/utf8.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// Le shell après windows.h.
#include <shlobj.h>

#include <SDL3/SDL_misc.h>
#endif

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Les conteneurs vidéo courants, tous lus par FFmpeg. AVI a deux noms dans shared-mime-info.
constexpr std::array<std::string_view, 15> kMimeTypes{
    "video/x-matroska", "video/webm",    "video/mp4",  "video/x-m4v",     "video/quicktime",
    "video/x-msvideo",  "video/vnd.avi", "video/mpeg", "video/mp2t",      "video/x-flv",
    "video/x-ms-wmv",   "video/ogg",     "video/3gpp", "video/x-ogm+ogg", "video/x-theora+ogg"};
constexpr std::array<std::string_view, 14> kExtensions{".mkv", ".webm", ".mp4",  ".m4v", ".mov",
                                                       ".avi", ".mpg",  ".mpeg", ".ts",  ".m2ts",
                                                       ".flv", ".wmv",  ".ogv",  ".3gp"};

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
                    std::format("registre : HKCU\\{} (erreur {})", value.key, status));
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
                    std::format("registre : HKCU\\{} (erreur {})", tree, status));
  }
  RegDeleteKeyW(HKEY_CURRENT_USER, L"Software\\AJP");  // seulement vide : rien d'autre n'y est à nous
  for (const RegistryValue& value : registry_shared_values()) {
    const LSTATUS status =
        RegDeleteKeyValueW(HKEY_CURRENT_USER, wide(value.key).c_str(), wide(value.name).c_str());
    if (!removed(status))
      return refuse(RefusalCode::FileUnwritable,
                    std::format("registre : HKCU\\{} : {} (erreur {})", value.key, value.name, status));
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

std::expected<std::string, Refusal> desktop_entry(const std::filesystem::path& executable) {
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
      "GenericName=Lecteur vidéo\n"
      "Comment=Anime agrandi et interpolé en temps réel\n"
      "Icon=video-x-generic\n"
      "TryExec={}\n"
      "Exec={} %f\n"
      "Terminal=false\n"
      "Categories=AudioVideo;Video;Player;\n"
      "MimeType={}\n",
      kAppName, desktop_string(path), exec_argument(path), mimes);
}

std::filesystem::path desktop_file(const PlayerPlace& place) {
  const std::filesystem::path data = place.data.empty() ? place.home / ".local" / "share" : place.data;
  return data / "applications" / std::string(kDesktopId);
}

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
      {prog, "", std::format("Vidéo ({})", kAppName)},
      {prog + "\\DefaultIcon", "", std::format("\"{}\",0", executable)},
      {prog + "\\shell\\open", "FriendlyAppName", std::string(kAppName)},
      {prog + "\\shell\\open\\command", "", command},
      {app, "FriendlyAppName", std::string(kAppName)},
      {app + "\\shell\\open\\command", "", command},
      {std::string(kCapabilities), "ApplicationName", std::string(kAppName)},
      {std::string(kCapabilities), "ApplicationDescription", "Anime agrandi et interpolé en temps réel"},
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

std::expected<std::string, Refusal> register_player(const PlayerPlace& place) {
#if defined(_WIN32)
  if (auto written = write_registry(registry_values(prepare::utf8_path(place.executable))); !written)
    return std::unexpected(std::move(written.error()));
  return std::format(
      "AJP figure dans « Ouvrir avec ». Pour en faire le lecteur des vidéos : « Faire de ce lecteur le "
      "lecteur par défaut » "
      "dans Paramètres > Général > Application, ou Paramètres > Applications > Applications par défaut > {}.",
      kAppName);
#elif defined(__APPLE__)
  static_cast<void>(place);
  return std::string(
      "AJP.app déclare déjà les vidéos qu'il ouvre. Pour en faire le lecteur par défaut : clic droit sur "
      "une vidéo > Lire les informations > Ouvrir avec : AJP > Tout modifier.");
#else
  if (place.home.empty() && place.data.empty())
    return refuse(RefusalCode::FieldMissing, "lecteur par défaut : dossier personnel inconnu");
  // Dans un AppImage l'exécutable est monté à un endroit provisoire : la fiche pointe sur le fichier
  // .AppImage lui-même (variable APPIMAGE posée par son lanceur).
  std::filesystem::path launcher = place.executable;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  const char* image = std::getenv("APPIMAGE");  // env: lanceur
  if (image != nullptr && *image != '\0') launcher = image;
  auto text = desktop_entry(launcher);
  if (!text) return std::unexpected(std::move(text.error()));
  const std::filesystem::path file = desktop_file(place);
  std::error_code error;
  std::filesystem::create_directories(file.parent_path(), error);
  std::filesystem::path temp = file;
  temp += ".tmp";
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out << *text;
    if (!out.flush()) return refuse(RefusalCode::FileUnwritable, prepare::utf8_path(temp));
  }
  std::filesystem::rename(temp, file, error);
  if (error) {
    return refuse(RefusalCode::FileUnwritable,
                  std::format("{} : {}", prepare::utf8_path(file), error.message()));
  }
  refresh_desktop_database(file.parent_path());
  return std::format(
      "AJP figure dans « Ouvrir avec » ({}). Pour en faire le lecteur des vidéos : « Faire de ce lecteur le "
      "lecteur par défaut » "
      "dans Paramètres > Général > Application, ou xdg-mime default {}{}",
      prepare::utf8_path(file), kDesktopId, mime_list());
#endif
}

std::expected<std::string, Refusal> unregister_player(const PlayerPlace& place) {
#if defined(_WIN32)
  if (auto erased = erase_registry(prepare::utf8_path(place.executable)); !erased)
    return std::unexpected(std::move(erased.error()));
  return std::string("AJP ne figure plus dans « Ouvrir avec ».");
#elif defined(__APPLE__)
  static_cast<void>(place);
  return std::string(
      "Sous macOS, l'inscription suit AJP.app : la retirer, c'est mettre l'app à la corbeille.");
#else
  if (place.home.empty() && place.data.empty())
    return refuse(RefusalCode::FieldMissing, "lecteur par défaut : dossier personnel inconnu");
  const std::filesystem::path file = desktop_file(place);
  std::error_code error;
  if (!std::filesystem::remove(file, error)) {
    if (error) {
      return refuse(RefusalCode::FileUnwritable,
                    std::format("{} : {}", prepare::utf8_path(file), error.message()));
    }
    return std::string("AJP n'était pas inscrit : rien à retirer.");
  }
  refresh_desktop_database(file.parent_path());
  return std::string("AJP ne figure plus dans « Ouvrir avec ».");
#endif
}

std::expected<std::string, Refusal> choose_as_default(const PlayerPlace& place) {
  auto registered = register_player(place);
  if (!registered) return registered;
#if defined(_WIN32)
  // Windows 11 ouvre la page du lecteur, avec « Définir par défaut » pour tous ses types ; Windows 10
  // ouvre la liste des applications par défaut.
  if (!SDL_OpenURL(std::format("ms-settings:defaultapps?registeredAppUser={}", kRegisteredName).c_str()))
    return registered;
  return std::format(
      "Paramètres ouverts sur {} : « Définir par défaut » en haut de la page, ou un type à la fois.",
      kAppName);
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
    return std::string("AJP est maintenant le lecteur des vidéos (mkv, mp4, webm, avi, mov…).");
  }
  return refuse(RefusalCode::LibraryUnavailable,
                "xdg-mime introuvable (paquet xdg-utils) : clic droit sur une vidéo > Ouvrir avec > AJP, "
                "en cochant « toujours »");
#endif
}

bool default_handler_is_ajp(std::string_view query_output) {
  while (!query_output.empty() &&
         (query_output.back() == '\n' || query_output.back() == '\r' || query_output.back() == ' '))
    query_output.remove_suffix(1);
  return query_output == kDesktopId;
}

bool is_default_player(const PlayerPlace& place) {
  static_cast<void>(place);
#if defined(_WIN32) || defined(__APPLE__)
  return false;  // Windows ne le dit qu'au registre de l'utilisateur (UserChoice) : la case suffit
#else
  for (const char* tool : {"/usr/bin/xdg-mime", "/usr/local/bin/xdg-mime", "/bin/xdg-mime"}) {
    std::error_code error;
    if (!std::filesystem::exists(tool, error)) continue;
    const prepare::ProcessSpec spec{.program = tool,
                                    .args = {"query", "default", "video/x-matroska"},
                                    .timeout = std::chrono::seconds(10)};
    std::string out;
    auto run = prepare::run_process(spec, [&](std::string_view line) { out = line; }, std::stop_token());
    return run && run->kind == prepare::ExitKind::Exited && run->code == 0 && default_handler_is_ajp(out);
  }
  return false;
#endif
}

void leave_lone_console() {
#ifdef _WIN32
  std::array<DWORD, 2> processes{};
  if (GetConsoleProcessList(processes.data(), static_cast<DWORD>(processes.size())) == 1) {
    FreeConsole();
  } else {
    SetConsoleOutputCP(CP_UTF8);  // le journal est en UTF-8 : « échoué », pas « ├®chou├® »
  }
#endif
}

}  // namespace ajp::app::play
