#include "app/play/engine_tools.h"

#include <array>
#include <cctype>
#include <charconv>
#include <format>
#include <string>
#include <system_error>

#include "core/charconv.h"

#if defined(_WIN32)
#include <windows.h>
#include <winhttp.h>
#endif

namespace ajp::app::play {

std::optional<std::filesystem::path> system_tool(std::string_view name) {
#if defined(_WIN32)
  // Le dossier système réel (Windows ailleurs que sur C:, ou dans un autre dossier), jamais un chemin en dur.
  std::array<wchar_t, MAX_PATH> system{};
  const UINT length = GetSystemDirectoryW(system.data(), static_cast<UINT>(system.size()));
  if (length == 0 || length >= system.size()) return std::nullopt;
  const std::array<std::filesystem::path, 1> dirs{std::filesystem::path(std::wstring(system.data(), length))};
  const std::string file = std::string(name) + ".exe";
#else
  const std::array<std::filesystem::path, 4> dirs{"/usr/bin", "/bin", "/usr/local/bin", "/opt/homebrew/bin"};
  const std::string file(name);
#endif
  for (const std::filesystem::path& dir : dirs) {
    std::error_code error;
    if (std::filesystem::is_regular_file(dir / file, error)) return dir / file;
  }
  return std::nullopt;
}

std::optional<std::string> proxy_from_setting(std::string_view setting) {
  // « hôte:port » pour tous, ou « http=h1:p;https=h2:p » par protocole.
  if (setting.empty()) return std::nullopt;
  if (setting.find('=') == std::string_view::npos) {
    const std::size_t end = setting.find_first_of("; ");
    return std::string(setting.substr(0, end));
  }
  for (std::string_view rest = setting; !rest.empty();) {
    const std::size_t end = rest.find_first_of("; ");
    const std::string_view item = rest.substr(0, end);
    if (item.starts_with("https=")) return std::string(item.substr(6));
    rest = end == std::string_view::npos ? std::string_view() : rest.substr(end + 1);
  }
  return std::nullopt;
}

namespace {

std::optional<std::string> windows_proxy() {
#if defined(_WIN32)
  WINHTTP_CURRENT_USER_IE_PROXY_CONFIG config{};
  if (WinHttpGetIEProxyConfigForCurrentUser(&config) == FALSE) return std::nullopt;
  std::string text;
  if (config.lpszProxy != nullptr) {
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, config.lpszProxy, -1, nullptr, 0, nullptr, nullptr);
    if (bytes > 1) {
      text.resize(static_cast<std::size_t>(bytes - 1));
      WideCharToMultiByte(CP_UTF8, 0, config.lpszProxy, -1, text.data(), bytes, nullptr, nullptr);
    }
  }
  for (LPWSTR owned : {config.lpszProxy, config.lpszProxyBypass, config.lpszAutoConfigUrl}) {
    if (owned != nullptr) GlobalFree(owned);
  }
  return proxy_from_setting(text);
#else
  return std::nullopt;
#endif
}

}  // namespace

std::vector<std::string> curl_system_args() {
  std::vector<std::string> args;
#ifdef _WIN32
  args.emplace_back("--ssl-revoke-best-effort");
#endif
  if (const auto proxy = windows_proxy()) {
    args.emplace_back("--proxy");
    args.push_back(*proxy);
  }
  return args;
}

std::vector<std::string> curl_download_args() {
  std::vector<std::string> args{"-q",
                                "--proto",
                                "=https,file",
                                "--proto-redir",
                                "=https",
                                "--fail",
                                "--location",
                                "--retry",
                                "5",
                                "--retry-delay",
                                "3",
                                "--retry-all-errors",
                                "--speed-limit",
                                "1024",
                                "--speed-time",
                                "60",
                                "--connect-timeout",
                                "30"};
  for (std::string& arg : curl_system_args()) args.push_back(std::move(arg));
  return args;
}

std::optional<float> curl_percent(std::string_view line) {
  const std::size_t percent = line.rfind('%');
  if (percent == std::string_view::npos) return std::nullopt;
  std::size_t start = percent;
  while (start > 0 &&
         (std::isdigit(static_cast<unsigned char>(line[start - 1])) != 0 || line[start - 1] == '.'))
    --start;
  float value = 0;
  const auto [end, error] = core::from_chars(line.data() + start, line.data() + percent, value);
  if (error != std::errc{} || end != line.data() + percent) return std::nullopt;
  return value / 100.0F;
}

std::expected<void, Refusal> run_tool(const prepare::ProcessSpec& spec, const prepare::LineSink& on_line,
                                      const std::stop_token& stop, std::string_view what) {
  std::string last;
  const auto sink = [&](std::string_view line) {
    last = line;
    if (on_line) on_line(line);
  };
  auto exit = prepare::run_process(spec, sink, stop);
  if (!exit) return std::unexpected(exit.error());
  if (exit->kind == prepare::ExitKind::Cancelled)
    return std::unexpected(Refusal{RefusalCode::Cancelled, std::format("{} annulé", what)});
  if (exit->kind != prepare::ExitKind::Exited || exit->code != 0) {
    return std::unexpected(Refusal{RefusalCode::DecodeFailed,
                                   std::format("{} : {} ({})", what, prepare::describe(*exit), last)});
  }
  return {};
}

}  // namespace ajp::app::play
