// Lanceur POSIX (Linux, macOS) : fork et exec sous Linux, posix_spawn sous macOS, tube non
// bloquant lu par poll.
//
// L'enfant prend son propre groupe de processus : l'annulation et le délai le
// tuent avec tout ce qu'il aurait lancé. Les descripteurs du lecteur sont tous
// ouverts en O_CLOEXEC ; l'enfant ne reçoit que stdin (nul), stdout et stderr.
// Sous Linux, l'enfant meurt avec le lecteur (PR_SET_PDEATHSIG) : un trtexec orphelin
// fausserait la construction suivante (A05-01), comme l'objet Job de Windows l'empêche.
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
#include "prepare/utf8.h"

#ifdef __APPLE__
#include <crt_externs.h>
#endif
#ifdef __linux__
#include <sys/prctl.h>
#include <sys/resource.h>
#endif

namespace ajp::prepare {
namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

constexpr milliseconds kPollSlice{20};

// Les signaux que l'enfant reprend à leur action par défaut, quoi qu'en fasse le lecteur.
constexpr std::array kDefaultSignals{SIGPIPE, SIGINT, SIGTERM, SIGQUIT, SIGHUP, SIGCHLD};

std::unexpected<Refusal> spawn_failure(const std::filesystem::path& program, int error) {
  return std::unexpected(Refusal{
      RefusalCode::BuilderFailed,
      "lancement impossible de " + utf8_path(program) + " : " + std::generic_category().message(error)});
}

// L'environnement du lecteur, transmis sans être lu.
char** inherited_environment() {
#ifdef __APPLE__
  return *_NSGetEnviron();  // env: hérité
#else
  return environ;  // env: hérité
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

// Un descripteur au-dessus de 0, 1 et 2 : un lecteur lancé sans eux recevrait son tube en 1, que
// dup2(1, 1) laisserait en O_CLOEXEC, et l'enfant perdrait sa sortie.
int above_stdio(int fd) {
  if (fd > STDERR_FILENO) return fd;
  const int moved = ::fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
  if (moved < 0) return fd;
  ::close(fd);
  return moved;
}

std::expected<Pipe, int> make_pipe() {
  std::array<int, 2> fds{};
#ifdef __linux__
  if (::pipe2(fds.data(), O_CLOEXEC) != 0) return std::unexpected(errno);
#else
  if (::pipe(fds.data()) != 0) return std::unexpected(errno);
  ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
#endif
  Pipe pipe{.read = Fd(above_stdio(fds[0])), .write = Fd(above_stdio(fds[1]))};
  ::fcntl(pipe.read.get(), F_SETFL, ::fcntl(pipe.read.get(), F_GETFL) | O_NONBLOCK);
  return pipe;
}

#ifndef __linux__
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
    for (const int sig : kDefaultSignals) sigaddset(&defaults, sig);
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
#endif

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

#ifdef __linux__
// L'enfant entre vfork et exec : rien que des appels sûrs dans un processus à plusieurs threads (ni
// allocation, ni verrou). Un échec part par `report` (errno), puis _exit.
[[noreturn]] void child_failed(int report) {
  const int error = errno;
  while (::write(report, &error, sizeof error) < 0 && errno == EINTR) {
  }
  ::_exit(127);
}

[[noreturn]] void exec_child(const char* program, char* const* argv, char* const* envp,
                             const char* working_dir, int output, int report, pid_t parent) {
  const auto fail = [report] { child_failed(report); };
  // Le lecteur meurt (plantage, kill -9, session fermée) : l'enfant reçoit SIGKILL (A05-01). Le
  // signal suit le thread qui a forké ; run_process l'y garde jusqu'à la fin de l'enfant. Un
  // lecteur déjà mort avant prctl : l'enfant a changé de parent et s'arrête de lui-même.
  if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) fail();
  if (::getppid() != parent) ::_exit(127);
  if (::setpgid(0, 0) != 0) fail();
  // C34-01 : la hausse de priorité du thread qui lance (nice négatif, celle du thread principal) ne passe pas
  // à trtexec ni à curl, qui travaillent pour le fond. Comme sous Windows, où un enfant part en priorité
  // normale : jamais au-dessus du nice 0. getpriority ne peut échouer sur soi-même ; -1 est un nice valable.
  if (::getpriority(PRIO_PROCESS, 0) < 0) ::setpriority(PRIO_PROCESS, 0, 0);
  struct sigaction fallback{};
  fallback.sa_handler = SIG_DFL;
  sigemptyset(&fallback.sa_mask);
  for (const int sig : kDefaultSignals) ::sigaction(sig, &fallback, nullptr);
  sigset_t none;
  sigemptyset(&none);
  ::sigprocmask(SIG_SETMASK, &none, nullptr);
  if (working_dir != nullptr && ::chdir(working_dir) != 0) fail();
  const int null = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  if (null < 0 || (null != STDIN_FILENO && ::dup2(null, STDIN_FILENO) < 0)) fail();
  if (null == STDIN_FILENO) ::fcntl(STDIN_FILENO, F_SETFD, 0);  // gardé à l'exec
  if (::dup2(output, STDOUT_FILENO) < 0 || ::dup2(output, STDERR_FILENO) < 0) fail();
  ::execve(program, argv, envp);
  child_failed(report);
}

// Lance l'enfant ; rend son pid, ou l'errno de fork ou d'exec (comme posix_spawn).
std::expected<pid_t, int> spawn_child(const std::filesystem::path& program, char* const* argv,
                                      char* const* envp, const std::filesystem::path& working_dir,
                                      int output) {
  std::array<int, 2> report{};
  if (::pipe2(report.data(), O_CLOEXEC) != 0) return std::unexpected(errno);
  const Fd report_read(above_stdio(report[0]));
  Fd report_write(above_stdio(report[1]));
  const char* dir = working_dir.empty() ? nullptr : working_dir.c_str();
  const pid_t parent = ::getpid();
  // vfork et non fork : le lecteur porte un contexte CUDA, Vulkan et plusieurs Gio d'adresses, dont fork
  // copierait les tables de pages (fautes copie-sur-écriture du thread de rendu jusqu'à l'exec, ENOMEM sous
  // vm.overcommit_memory=2). L'enfant ne fait que des appels système sûrs puis execve ou _exit (exec_child ne
  // rend jamais la main), ce qui est ce que vfork permet ; le parent reprend à l'exec (B05-08).
  const pid_t pid = ::vfork();  // NOLINT(bugprone-unsafe-functions) : choix expliqué ci-dessus
  if (pid < 0) return std::unexpected(errno);
  if (pid == 0) exec_child(program.c_str(), argv, envp, dir, output, report_write.get(), parent);
  report_write.reset();
  // exec réussi : le tube se ferme (O_CLOEXEC) et read rend 0.
  int error = 0;
  ssize_t got = 0;
  do {
    got = ::read(report_read.get(), &error, sizeof error);
  } while (got < 0 && errno == EINTR);
  if (got == static_cast<ssize_t>(sizeof error)) {
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    return std::unexpected(error);
  }
  return pid;
}
#endif

// L'enfant de spec, sa sortie sur output : son pid, ou l'errno du lancement.
std::expected<pid_t, int> spawn(const ProcessSpec& spec, char* const* argv, int output) {
  Environment env;
  if (!spec.extra_env.empty()) env = environment_with(spec.extra_env);
  char** const envp = spec.extra_env.empty() ? inherited_environment() : env.pointers.data();
#ifdef __linux__
  return spawn_child(spec.program, argv, envp, spec.working_dir, output);
#else
  const SpawnSetup setup(output, spec.working_dir);
  pid_t pid = 0;
  const int error =
      ::posix_spawn(&pid, spec.program.c_str(), setup.actions(), setup.attributes(), argv, envp);
  if (error != 0) return std::unexpected(error);
  return pid;
#endif
}

}  // namespace

std::expected<ProcessExit, Refusal> run_process(const ProcessSpec& spec, const LineSink& on_line,
                                                const std::stop_token& stop) {
  if (spec.timeout <= milliseconds::zero()) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "délai du constructeur nul ou négatif"});
  }
  auto pipe = make_pipe();
  if (!pipe) return spawn_failure(spec.program, pipe.error());

  std::vector<std::string> storage{utf8_path(spec.program)};
  storage.insert(storage.end(), spec.args.begin(), spec.args.end());
  std::vector<char*> argv;
  argv.reserve(storage.size() + 1);
  for (std::string& arg : storage) argv.push_back(arg.data());
  argv.push_back(nullptr);

  const auto start = Clock::now();
  const auto spawned = spawn(spec, argv.data(), pipe->write.get());
  if (!spawned) return spawn_failure(spec.program, spawned.error());
  const pid_t pid = *spawned;
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
