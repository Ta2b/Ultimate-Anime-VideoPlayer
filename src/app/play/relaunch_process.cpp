#include "app/play/relaunch_process.h"

#include <cerrno>
#include <cstring>
#include <format>

#ifdef _WIN32
#include <process.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#endif

#ifndef _WIN32
extern char** environ;  // NOLINT(readability-redundant-declaration) env: hérité, POSIX ne le déclare pas
#endif

namespace ajp::app::play {

std::string windows_quoted(const std::string& argument) {
  if (!argument.empty() && argument.find_first_of(" \t\"") == std::string::npos) return argument;
  std::string out = "\"";
  std::size_t backslashes = 0;
  for (const char c : argument) {
    if (c == '\\') {
      ++backslashes;
      out += c;
      continue;
    }
    if (c == '"') out.append(backslashes + 1, '\\');
    out += c;
    backslashes = 0;
  }
  out.append(backslashes, '\\');  // avant le guillemet fermant
  out += '"';
  return out;
}

std::string start_relaunched(const std::filesystem::path& executable,
                             const std::vector<std::string>& arguments, [[maybe_unused]] bool replace) {
#ifdef _WIN32
  // Chaque argument cité, et l'exécutable gardé dans une chaîne qui vit jusqu'au lancement (c_str() d'un
  // temporaire pendrait).
  const std::string program = executable.string();
  std::vector<std::string> quoted{windows_quoted(program)};
  for (const std::string& argument : arguments) quoted.push_back(windows_quoted(argument));
  std::vector<const char*> pointers;
  for (const std::string& argument : quoted) pointers.push_back(argument.c_str());
  pointers.push_back(nullptr);
  if (_spawnv(_P_NOWAIT, program.c_str(), pointers.data()) != -1) return {};
  const int error = errno;
  return std::format("_spawnv : {} (errno {})", std::strerror(error), error);
#else
  const long open_max = sysconf(_SC_OPEN_MAX);
  for (int fd = 3; fd < (open_max > 0 && open_max < 65536 ? open_max : 65536); ++fd) {
    const int flags = fcntl(fd, F_GETFD);
    if (flags >= 0) fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
  std::vector<char*> pointers{const_cast<char*>(executable.c_str())};
  for (const std::string& argument : arguments) pointers.push_back(const_cast<char*>(argument.c_str()));
  pointers.push_back(nullptr);
  if (replace) {
    execv(executable.c_str(), pointers.data());
    const int error = errno;  // execv ne revient que sur un échec
    return std::format("execv : {} (errno {})", std::strerror(error), error);
  }
  pid_t child = 0;
  // posix_spawn (clone et exec dans l'enfant) : rien n'attend les threads de ce processus.
  const int error =
      posix_spawn(&child, executable.c_str(), nullptr, nullptr, pointers.data(), environ);  // env: hérité
  if (error == 0) return {};
  return std::format("posix_spawn : {} (errno {})", std::strerror(error), error);
#endif
}

}  // namespace ajp::app::play
