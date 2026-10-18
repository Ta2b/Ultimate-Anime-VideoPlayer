#include "subs/ass_library.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <format>
#include <string>

#include <ass/ass.h>
#include <spdlog/logger.h>

#include "subs/log.h"

namespace ajp::subs {
namespace {

// Niveaux de libass : 0 fatal, 1 erreur, 2 avertissement, 4 info, 6 détail, 7 débogage.
spdlog::level::level_enum level_of(int libass_level) {
  if (libass_level <= 1) return spdlog::level::err;
  if (libass_level <= 3) return spdlog::level::warn;
  if (libass_level <= 5) return spdlog::level::debug;
  return spdlog::level::trace;
}

void forward_message(int level, const char* format, va_list args, void* data) {
  auto* log = static_cast<spdlog::logger*>(data);
  if (!log->should_log(level_of(level))) return;
  std::array<char, 1024> text{};
  if (std::vsnprintf(text.data(), text.size(), format, args) < 0) return;
  log->log(level_of(level), "libass : {}", text.data());
}

}  // namespace

void AssLibrary::Free::operator()(ass_library* library) const {
  ass_library_done(library);
}

AssLibrary::AssLibrary(std::shared_ptr<spdlog::logger> log, std::unique_ptr<ass_library, Free> library)
    : log_(std::move(log)), library_(std::move(library)) {}

std::expected<AssLibrary, Refusal> AssLibrary::create() {
  std::shared_ptr<spdlog::logger> log = make_logger();
  std::unique_ptr<ass_library, Free> library(ass_library_init());
  if (library == nullptr) return std::unexpected(Refusal{RefusalCode::LibassFailed, "ass_library_init"});
  ass_set_message_cb(library.get(), forward_message, log.get());
  // Polices des sections [Fonts] des scripts : jointes à la lecture, comme mpv
  // (embeddedfonts=yes par défaut). Sans cela libass les décode puis les jette.
  ass_set_extract_fonts(library.get(), 1);
  return AssLibrary(std::move(log), std::move(library));
}

std::expected<void, Refusal> AssLibrary::add_font(std::string_view name, std::span<const std::byte> data) {
  if (name.empty() || data.empty() || data.size() > kMaxFontBytes) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("police jointe « {} » : {} octets", name, data.size())});
  }
  const std::string terminated(name);  // libass attend une chaîne terminée par zéro
  ass_add_font(library_.get(), terminated.c_str(), reinterpret_cast<const char*>(data.data()),
               static_cast<int>(data.size()));
  log_->debug("police jointe « {} », {} octets", name, data.size());
  return {};
}

}  // namespace ajp::subs
