// Lanceur POSIX (Linux, macOS) : posix_spawn, tube non bloquant lu par poll.
//
// L'enfant prend son propre groupe de processus : l'annulation et le délai le
// tuent avec tout ce qu'il aurait lancé. Les descripteurs du lecteur sont tous
// ouverts en O_CLOEXEC ; l'enfant ne reçoit que stdin (nul), stdout et stderr.
#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <optional>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

#include "prepare/process.h"

#ifdef __APPLE__
#include <crt_externs.h>
#endif

namespace ajp::prepare {
namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

constexpr milliseconds kPollSlice{20};

std::unexpected<Refusal> spawn_failure(const std::filesystem::path& program, int error) {
  return std::unexpected(Refusal{
      RefusalCode::BuilderFailed,
      "lancement impossible de " + program.string() + " : " + std::generic_category().message(error)});
}

// L'environnement du lecteur, transmis sans être lu.
char** inherited_environment() {
#ifdef __APPLE__
  return *_NSGetEnviron();
#else
  return environ;
#endif
}

// L'environnement hérité, chaque nom de extra remplacé par sa valeur, en tableau pour posix_spawn.
struct Environment {
  std::vector<std::string> storage;
  std::vector<char*> pointers;
};
Environment environment_with(const std::vector<std::string>& extra) {
  Environment env;
  for (char** e = inherited_environment(); *e != nullptr; ++e) {
    const std::string_view entry(*e);
    const std::string_view name = entry.substr(0, entry.find('='));
    const bool replaced = std::ranges::any_of(extra, [&](const std::string& x) {
      return x.size() > name.size() && x.starts_with(name) && x[name.size()] == '=';
    });
    if (!replaced) env.storage.emplace_back(entry);
  }
  env.storage.insert(env.storage.end(), extra.begin(), extra.end());
  for (std::string& s : env.storage) env.pointers.push_back(s.data());
  env.pointers.push_back(nullptr);
  return env;
}

class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Fd& operator=(Fd&&) = delete;
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() { reset(); }
  int get() const { return fd_; }
  void reset() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }

 private:
  int fd_ = -1;
};

struct Pipe {
  Fd read;
  Fd write;
};

std::expected<Pipe, int> make_pipe() {
  std::array<int, 2> fds{};
#ifdef __linux__
  if (::pipe2(fds.data(), O_CLOEXEC) != 0) return std::unexpected(errno);
#else
  if (::pipe(fds.data()) != 0) return std::unexpected(errno);
  ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
#endif
  Pipe pipe{.read = Fd(fds[0]), .write = Fd(fds[1])};
  ::fcntl(pipe.read.get(), F_SETFL, ::fcntl(pipe.read.get(), F_GETFL) | O_NONBLOCK);
  return pipe;
}

// Actions et attributs de posix_spawn, détruits à la sortie de portée.
class SpawnSetup {
 public:
  SpawnSetup(int output, const std::filesystem::path& working_dir) {
    ::posix_spawn_file_actions_init(&actions_);
    if (!working_dir.empty()) ::posix_spawn_file_actions_addchdir_np(&actions_, working_dir.c_str());
    ::posix_spawn_file_actions_addopen(&actions_, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    ::posix_spawn_file_actions_adddup2(&actions_, output, STDOUT_FILENO);
    ::posix_spawn_file_actions_adddup2(&actions_, output, STDERR_FILENO);
    ::posix_spawnattr_init(&attr_);
    sigset_t none;
    sigemptyset(&none);
    sigset_t defaults;
    sigemptyset(&defaults);
    for (const int sig : {SIGPIPE, SIGINT, SIGTERM, SIGQUIT, SIGHUP, SIGCHLD}) sigaddset(&defaults, sig);
    ::posix_spawnattr_setsigmask(&attr_, &none);
    ::posix_spawnattr_setsigdefault(&attr_, &defaults);
    ::posix_spawnattr_setpgroup(&attr_, 0);
    ::posix_spawnattr_setflags(&attr_,
                               POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
  }
  SpawnSetup(const SpawnSetup&) = delete;
  SpawnSetup& operator=(const SpawnSetup&) = delete;
  ~SpawnSetup() {
    ::posix_spawnattr_destroy(&attr_);
    ::posix_spawn_file_actions_destroy(&actions_);
  }
  const posix_spawn_file_actions_t* actions() const { return &actions_; }
  const posix_spawnattr_t* attributes() const { return &attr_; }

 private:
  posix_spawn_file_actions_t actions_{};
  posix_spawnattr_t attr_{};
};

// Lit tout ce qui est disponible ; true quand tous les écrivains ont fermé.
bool drain(int fd, LineSplitter& lines) {
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t n = ::read(fd, buffer.data(), buffer.size());
    if (n > 0) {
      lines.feed(std::string_view(buffer.data(), static_cast<std::size_t>(n)));
    } else if (n == 0) {
      return true;
    } else if (errno != EINTR) {
      return errno != EAGAIN && errno != EWOULDBLOCK;
    }
  }
}

}  // namespace

std::expected<ProcessExit, Refusal> run_process(const ProcessSpec& spec, const LineSink& on_line,
                                                const std::stop_token& stop) {
  if (spec.timeout <= milliseconds::zero()) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "délai du constructeur nul ou négatif"});
  }
  auto pipe = make_pipe();
  if (!pipe) return spawn_failure(spec.program, pipe.error());

  std::vector<std::string> storage{spec.program.string()};
  storage.insert(storage.end(), spec.args.begin(), spec.args.end());
  std::vector<char*> argv;
  argv.reserve(storage.size() + 1);
  for (std::string& arg : storage) argv.push_back(arg.data());
  argv.push_back(nullptr);

  const auto start = Clock::now();
  pid_t pid = 0;
  {
    const SpawnSetup setup(pipe->write.get(), spec.working_dir);
    Environment env;
    if (!spec.extra_env.empty()) env = environment_with(spec.extra_env);
    const int error =
        ::posix_spawn(&pid, spec.program.c_str(), setup.actions(), setup.attributes(), argv.data(),
                      spec.extra_env.empty() ? inherited_environment() : env.pointers.data());
    if (error != 0) return spawn_failure(spec.program, error);
  }
  pipe->write.reset();  // seul l'enfant écrit : sa fin ferme le tube

  LineSplitter lines(on_line);
  std::optional<ExitKind> killed;
  bool eof = false;
  int status = 0;
  for (;;) {
    const auto now = Clock::now();
    if (!killed && (stop.stop_requested() || now >= start + spec.timeout)) {
      killed = stop.stop_requested() ? ExitKind::Cancelled : ExitKind::TimedOut;
      ::kill(-pid, SIGKILL);
    }
    const auto left = std::chrono::duration_cast<milliseconds>(start + spec.timeout - now);
    const int slice =
        killed ? 0 : static_cast<int>(std::clamp(left, milliseconds::zero(), kPollSlice).count());
    pollfd ready{.fd = pipe->read.get(), .events = POLLIN, .revents = 0};
    if (!eof && ::poll(&ready, 1, slice) > 0) eof = drain(pipe->read.get(), lines);
    if (eof && !killed) ::poll(nullptr, 0, slice);  // l'enfant vit sans sortie : simple attente
    const pid_t waited = ::waitpid(pid, &status, killed ? 0 : WNOHANG);
    if (waited == pid) break;
    if (waited < 0 && errno != EINTR) {
      status = -1;  // SIGCHLD ignoré par l'hôte : l'enfant a été récolté sans nous
      break;
    }
  }
  if (!eof) drain(pipe->read.get(), lines);
  lines.finish();

  ProcessExit exit{.elapsed = std::chrono::duration_cast<milliseconds>(Clock::now() - start)};
  if (killed) {
    exit.kind = *killed;
  } else if (status >= 0 && WIFEXITED(status)) {
    exit.code = WEXITSTATUS(status);
  } else {
    exit.kind = ExitKind::Crashed;
    exit.code = status >= 0 && WIFSIGNALED(status) ? WTERMSIG(status) : -1;
  }
  return exit;
}

}  // namespace ajp::prepare
