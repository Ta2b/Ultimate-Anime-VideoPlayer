#include "app/play/engine_tools.h"

#include <array>
#include <cctype>
#include <charconv>
#include <format>
#include <string>
#include <system_error>

namespace ajp::app::play {

std::optional<std::filesystem::path> system_tool(std::string_view name) {
#if defined(_WIN32)
  const std::array<std::filesystem::path, 1> dirs{"C:/Windows/System32"};
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

std::optional<float> curl_percent(std::string_view line) {
  const std::size_t percent = line.rfind('%');
  if (percent == std::string_view::npos) return std::nullopt;
  std::size_t start = percent;
  while (start > 0 &&
         (std::isdigit(static_cast<unsigned char>(line[start - 1])) != 0 || line[start - 1] == '.'))
    --start;
  float value = 0;
  const auto [end, error] = std::from_chars(line.data() + start, line.data() + percent, value);
  if (error != std::errc{} || end != line.data() + percent) return std::nullopt;
  return value / 100.0F;
}

std::expected<void, Refusal> run_tool(prepare::ProcessSpec spec, const prepare::LineSink& on_line,
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
