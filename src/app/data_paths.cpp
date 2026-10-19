#include "app/data_paths.h"

#include <format>
#include <string>
#include <system_error>

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_stdinc.h>

namespace ajp::app {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

std::string utf8(const std::filesystem::path& path) {
  const std::u8string text = path.u8string();
  return {text.begin(), text.end()};
}

std::filesystem::path path_from_utf8(std::string_view text) {
  return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

std::filesystem::path resources_from_base(core::Os os, const std::filesystem::path& base) {
  if (os == core::Os::Linux) return (base / ".." / "share" / kAppFolder).lexically_normal();
  return base.lexically_normal();
}

std::expected<DataPaths, Refusal> data_paths_in(const std::filesystem::path& user_data,
                                                const std::filesystem::path& resources) {
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(user_data, error).lexically_normal();
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", utf8(user_data), error.message()));
  if (!absolute.has_filename() && absolute.has_relative_path())
    absolute = absolute.parent_path();  // « …/ajp/ »
  std::filesystem::create_directories(absolute, error);
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", utf8(absolute), error.message()));
  if (!std::filesystem::is_directory(absolute, error)) {
    return refuse(RefusalCode::FileUnwritable, std::format("{} n'est pas un dossier", utf8(absolute)));
  }
  const std::filesystem::path shipped = std::filesystem::absolute(resources, error).lexically_normal();
  if (error)
    return refuse(RefusalCode::FileUnreadable, std::format("{} : {}", utf8(resources), error.message()));
  return DataPaths{.user_data = absolute, .resources = shipped};
}

std::filesystem::path executable_dir() {
  const char* base = SDL_GetBasePath();  // possédé par SDL
  return base == nullptr ? std::filesystem::path{} : path_from_utf8(base);
}

std::expected<DataPaths, Refusal> system_data_paths(core::Os os) {
  const std::string app(kAppFolder);
  char* pref = SDL_GetPrefPath("", app.c_str());
  if (pref == nullptr) {
    return refuse(RefusalCode::FileUnwritable, std::format("SDL_GetPrefPath : {}", SDL_GetError()));
  }
  const std::filesystem::path user_data = path_from_utf8(pref);
  SDL_free(pref);
  const std::filesystem::path base = executable_dir();
  if (base.empty())
    return refuse(RefusalCode::FileUnreadable, std::format("SDL_GetBasePath : {}", SDL_GetError()));
  return data_paths_in(user_data, resources_from_base(os, base));
}

}  // namespace ajp::app
