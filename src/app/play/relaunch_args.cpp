#include "app/play/relaunch_args.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/close_range.h>
#include <sys/syscall.h>
#endif
#endif

namespace ajp::app::play {

std::vector<std::string> relaunch_arguments(std::span<const std::string_view> launched,
                                            RelaunchedFor reason) {
  std::vector<std::string> arguments;
  for (const std::string_view argument : launched)
    if (!argument.starts_with("--relaunched=")) arguments.emplace_back(argument);
  arguments.push_back("--relaunched=" + std::string(relaunched_name(reason)));
  return arguments;
}

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

void close_on_exec_all() {
#ifndef _WIN32
#if defined(__linux__) && defined(SYS_close_range) && defined(CLOSE_RANGE_CLOEXEC)
  // Linux 5.11 et plus : un seul appel, sans borne sur le nombre de descripteurs.
  if (::syscall(SYS_close_range, 3U, ~0U, CLOSE_RANGE_CLOEXEC) == 0) return;
#endif
  const long open_max = ::sysconf(_SC_OPEN_MAX);
  const long last = open_max > 0 && open_max < 65536 ? open_max : 65536;
  for (int fd = 3; fd < last; ++fd) {
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags >= 0) ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
#endif
}

}  // namespace ajp::app::play
