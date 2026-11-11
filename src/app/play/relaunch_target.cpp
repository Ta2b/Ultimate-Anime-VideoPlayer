#include "app/play/relaunch_target.h"

#include <charconv>
#include <chrono>
#include <format>
#include <fstream>
#include <string_view>
#include <system_error>

#include "app/play/frame_stats.h"
#include "app/play/player.h"
#include "core/file_sync.h"
#include "core/log.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

bool parse_int(std::string_view text, std::int64_t& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

std::string_view reason_name(RelaunchReason reason) {
  return reason == RelaunchReason::GpuLost ? "gpu-lost" : "engine-updated";
}

}  // namespace

std::filesystem::path relaunch_target_file(const std::filesystem::path& data) {
  return data / "relaunch.txt";
}

std::int64_t relaunch_now_s() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::expected<void, Refusal> write_relaunch_target(const std::filesystem::path& store,
                                                   const RelaunchTarget& target) {
  const std::string file = prepare::utf8_path(target.file);
  if (file.find_first_of("\n\r") != std::string::npos)
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "relance : chemin à saut de ligne"});
  return core::write_text_atomically(
      store, std::format("file={}\nstart_us={}\npaused={}\nreason={}\nwritten_s={}\n", file, target.start_us,
                         target.paused ? 1 : 0, reason_name(target.reason), target.written_s));
}

std::optional<RelaunchTarget> take_relaunch_target(const std::filesystem::path& store, std::int64_t now_s) {
  std::error_code error;
  if (!std::filesystem::exists(store, error)) return std::nullopt;
  RelaunchTarget target;
  bool has_file = false;
  bool has_time = false;
  {
    std::ifstream in(store, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
      const std::size_t equal = line.find('=');
      if (equal == std::string::npos) continue;
      const std::string_view key = std::string_view(line).substr(0, equal);
      const std::string_view value = std::string_view(line).substr(equal + 1);
      std::int64_t number = 0;
      if (key == "file" && !value.empty()) {
        target.file = prepare::path_from_utf8(value);
        has_file = true;
      } else if (key == "start_us" && parse_int(value, number)) {
        target.start_us = number < 0 ? 0 : number;
      } else if (key == "paused") {
        target.paused = value == "1";
      } else if (key == "reason") {
        target.reason = value == "gpu-lost" ? RelaunchReason::GpuLost : RelaunchReason::EngineUpdated;
      } else if (key == "written_s" && parse_int(value, number)) {
        target.written_s = number;
        has_time = true;
      }
    }
  }
  std::filesystem::remove(store, error);  // lu une fois : une relance suivante ne le rejoue pas
  if (!has_file || !has_time || now_s - target.written_s > kRelaunchFreshS || target.written_s > now_s + 5)
    return std::nullopt;
  return target;
}

bool ask_relaunch(const PlayerPaths& paths, const std::filesystem::path& file, std::int64_t start_us,
                  bool paused, RelaunchReason reason) {
  if (paths.relaunch == nullptr) return false;
  if (!file.empty()) {
    std::error_code absolute_error;
    const std::filesystem::path absolute = std::filesystem::absolute(file, absolute_error);
    if (auto written = write_relaunch_target(relaunch_target_file(paths.data),
                                             RelaunchTarget{.file = absolute_error ? file : absolute,
                                                            .start_us = start_us,
                                                            .paused = paused,
                                                            .reason = reason,
                                                            .written_s = relaunch_now_s()});
        !written)
      log_channel("app")->warn("relance : {} ; le lancement est rejoué tel quel, la position est perdue",
                               written.error().detail);
  }
  *paths.relaunch = true;
  if (paths.relaunch_reason != nullptr)
    *paths.relaunch_reason =
        reason == RelaunchReason::GpuLost ? RelaunchedFor::GpuLost : RelaunchedFor::EngineUpdated;
  return true;
}

std::optional<RelaunchTarget> take_relaunch(PlayOptions& options, const PlayerPaths& paths,
                                            std::int64_t now_s) {
  if (options.bench_out || options.data_dir) return std::nullopt;
  const std::filesystem::path store = relaunch_target_file(paths.data);
  if (options.relaunched == RelaunchedFor::None) {
    std::error_code error;
    if (std::filesystem::remove(store, error)) log_channel("app")->info("relance : fichier resté retiré");
    return std::nullopt;
  }
  std::optional<RelaunchTarget> target = take_relaunch_target(store, now_s);
  if (!target) return std::nullopt;
  log_channel("app")->info("relance : {} à {}", prepare::utf8_path(target->file.filename()),
                           format_position(target->start_us));
  options.file = target->file;
  return target;
}

}  // namespace ajp::app::play
