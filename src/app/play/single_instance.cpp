#include "app/play/single_instance.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <format>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>

#include "core/log.h"
#include "core/thread.h"
#include "core/utf8_path.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

constexpr std::size_t kMaxMessage = std::size_t{64} * 1024;
constexpr std::string_view kAck = "ok\n";
constexpr auto kAckWait = std::chrono::seconds(3);
constexpr auto kSlice = std::chrono::milliseconds(100);

bool message_complete(std::string_view message) {
  return message == "\n" || message.ends_with("\n\n") || message.size() >= kMaxMessage;
}

}  // namespace

// Les fichiers reçus vont dans la file de SDL comme un glisser-déposer ; SDL garde le pointeur jusqu'à la
// lecture de l'événement, les textes restent donc ici (quelques chemins par session).
bool push_forwarded_files(const std::vector<std::string>& files) {
  static std::mutex mutex;
  static std::deque<std::string> held;
  const std::scoped_lock lock(mutex);
  // Sans fichier (l'icône relancée), un dépôt vide : la fenêtre passe devant, rien ne s'ouvre.
  const std::vector<std::string> dropped = files.empty() ? std::vector<std::string>{std::string()} : files;
  bool all = true;
  for (const std::string& file : dropped) {
    held.push_back(file);
    SDL_Event drop{};
    drop.type = SDL_EVENT_DROP_FILE;
    drop.drop.source = kForwardedDrop.data();
    drop.drop.data = held.back().c_str();
    if (SDL_PushEvent(&drop)) continue;
    all = false;
    const auto log = log_channel("app");
    log->warn("instance : fichier reçu perdu ({})", SDL_GetError());
  }
  return all;
}

namespace {

// Remet les fichiers à SDL puis répond au lancement qui les envoie (A19-02) : « ok » seulement une fois
// les fichiers dans la file, sinon il ouvre lui-même sa fenêtre. Un lecteur qui démarre encore (SDL pas
// prêt avant la fin de l'attente de l'autre côté) répond tout de suite et remet les fichiers ensuite.
template <class Reply>
void hand_over(const std::vector<std::string>& files, const std::stop_token& stop, Reply reply) {
  const auto until = std::chrono::steady_clock::now() + kAckWait / 2;
  while (!stop.stop_requested() && SDL_WasInit(SDL_INIT_EVENTS) == 0 &&
         std::chrono::steady_clock::now() < until)
    std::this_thread::sleep_for(kSlice);
  if (SDL_WasInit(SDL_INIT_EVENTS) != 0) {
    if (push_forwarded_files(files)) reply();
    return;
  }
  reply();
  while (!stop.stop_requested() && SDL_WasInit(SDL_INIT_EVENTS) == 0) std::this_thread::sleep_for(kSlice);
  if (!stop.stop_requested()) push_forwarded_files(files);
}

}  // namespace

bool shares_instance(std::span<const std::string_view> args) {
  for (const std::string_view arg : args)
    if (arg.starts_with("--") && !arg.starts_with("--relaunched=")) return false;  // une relance partage
  return true;
}

std::string instance_key(const std::filesystem::path& settings_dir) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const char c : prepare::utf8_path(settings_dir)) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ULL;
  }
  return std::format("ajp-{:016x}", hash);
}

std::string instance_message(const std::vector<std::string>& files) {
  std::string message;
  for (const std::string& file : files) {
    if (file.empty() || file.find('\n') != std::string::npos) continue;
    message += file;
    message += '\n';
  }
  message += '\n';
  return message;
}

std::vector<std::string> instance_message_files(std::string_view message) {
  std::vector<std::string> files;
  while (!message.empty()) {
    const std::size_t end = message.find('\n');
    if (end == std::string_view::npos || end == 0) break;  // ligne coupée ou ligne vide : la fin
    files.emplace_back(message.substr(0, end));
    message.remove_prefix(end + 1);
  }
  return files;
}

#ifdef _WIN32

namespace {

constexpr auto kReadLimit = std::chrono::seconds(2);  // le message d'un client, comme sous Linux

std::wstring pipe_name(std::string_view key) {
  std::wstring name = L"\\\\.\\pipe\\";
  for (const char c : key) name += static_cast<wchar_t>(c);  // ASCII : « ajp- » et des chiffres
  return name;
}

}  // namespace

class InstanceListener {
 public:
  InstanceListener(HANDLE pipe, std::wstring name) : pipe_(pipe), name_(std::move(name)) {}
  InstanceListener(const InstanceListener&) = delete;
  InstanceListener& operator=(const InstanceListener&) = delete;
  ~InstanceListener() {
    if (thread_.joinable()) {
      thread_.request_stop();
      // ConnectNamedPipe attend un client : ce lancement s'y connecte lui-même pour la libérer.
      HANDLE self =
          CreateFileW(name_.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
      if (self != INVALID_HANDLE_VALUE) CloseHandle(self);
      thread_.join();
    }
    CloseHandle(pipe_);
  }

  void start() {
    auto started = start_thread("instance", ThreadReport([](const Refusal& refusal) {
                                  const auto log = log_channel("app");
                                  log->warn("instance : {}", refusal.detail);
                                }),
                                [this](const std::stop_token& stop, const ThreadReport&) { serve(stop); });
    if (started) thread_ = std::move(*started);
  }

 private:
  void serve(const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      if (ConnectNamedPipe(pipe_, nullptr) == 0) {
        const DWORD error = GetLastError();
        if (error != ERROR_PIPE_CONNECTED) {
          // ERROR_NO_DATA (un client venu puis parti) et toute autre erreur : le canal se déconnecte avant
          // l'essai suivant, sinon chaque ConnectNamedPipe rend la même erreur et l'écoute est morte
          // (A19-07).
          DisconnectNamedPipe(pipe_);
          if (error != last_error_) log_channel("app")->info("instance : ConnectNamedPipe, erreur {}", error);
          last_error_ = error;
          std::this_thread::sleep_for(kSlice);
          continue;
        }
      }
      last_error_ = 0;
      if (stop.stop_requested()) break;
      const std::string message = read_message(stop);
      const auto files = instance_message_files(message);
      const auto log = log_channel("app");
      log->info("instance : {} fichier(s) reçu(s) d'un autre lancement", files.size());
      bool replied = false;
      hand_over(files, stop, [&] {
        DWORD written = 0;
        WriteFile(pipe_, kAck.data(), static_cast<DWORD>(kAck.size()), &written, nullptr);
        FlushFileBuffers(pipe_);
        DisconnectNamedPipe(pipe_);
        replied = true;
      });
      if (!replied) DisconnectNamedPipe(pipe_);
    }
  }

  // Le message d'un client, lu sans ReadFile bloquant : seulement ce que PeekNamedPipe dit disponible, au
  // plus kReadLimit, comme sous Linux. Un client connecté qui se tait ne bloque plus ce thread, ni la
  // destruction qui le joint (A30-06).
  std::string read_message(const std::stop_token& stop) const {
    std::string message;
    std::array<char, 4096> chunk{};
    const auto until = std::chrono::steady_clock::now() + kReadLimit;
    while (!message_complete(message) && !stop.stop_requested() && std::chrono::steady_clock::now() < until) {
      DWORD available = 0;
      if (PeekNamedPipe(pipe_, nullptr, 0, nullptr, &available, nullptr) == 0) break;  // client parti
      if (available == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      DWORD read = 0;
      const DWORD wanted = std::min(available, static_cast<DWORD>(chunk.size()));
      if (ReadFile(pipe_, chunk.data(), wanted, &read, nullptr) == 0 || read == 0) break;
      message.append(chunk.data(), read);
    }
    return message;
  }

  HANDLE pipe_;
  DWORD last_error_ = 0;  // la dernière erreur de ConnectNamedPipe journalisée
  std::wstring name_;
  std::jthread thread_;
};

namespace {

// Vrai si le lecteur qui tient le canal a accusé réception.
bool send_files(const std::wstring& name, const std::string& message) {
  HANDLE pipe = INVALID_HANDLE_VALUE;
  for (int attempt = 0; attempt < 3 && pipe == INVALID_HANDLE_VALUE; ++attempt) {
    pipe = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PIPE_BUSY) WaitNamedPipeW(name.c_str(), 1000);
  }
  if (pipe == INVALID_HANDLE_VALUE) return false;
  AllowSetForegroundWindow(ASFW_ANY);  // la fenêtre du premier lecteur peut passer devant
  DWORD written = 0;
  bool acked = WriteFile(pipe, message.data(), static_cast<DWORD>(message.size()), &written, nullptr) != 0 &&
               written == message.size();
  // L'accusé, sans lecture bloquante : un premier lecteur gelé ne retient pas celui-ci.
  std::string reply;
  const auto until = std::chrono::steady_clock::now() + kAckWait;
  while (acked && !reply.ends_with('\n') && std::chrono::steady_clock::now() < until) {
    DWORD available = 0;
    if (PeekNamedPipe(pipe, nullptr, 0, nullptr, &available, nullptr) == 0) break;
    if (available == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }
    std::array<char, 16> chunk{};
    DWORD read = 0;
    if (ReadFile(pipe, chunk.data(), static_cast<DWORD>(chunk.size()), &read, nullptr) == 0) break;
    reply.append(chunk.data(), read);
  }
  CloseHandle(pipe);
  return acked && reply == kAck;
}

}  // namespace

InstanceClaim claim_instance(const std::filesystem::path& /*channel_dir*/, std::string_view key,
                             const std::vector<std::string>& files, bool take_over) {
  try {
    const std::wstring name = pipe_name(key);
    const auto until = std::chrono::steady_clock::now() + kTakeOverWait;
    for (;;) {
      HANDLE pipe =
          CreateNamedPipeW(name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
                           PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1,
                           4096, 4096, 0, nullptr);
      if (pipe != INVALID_HANDLE_VALUE) {
        auto listener = std::make_shared<InstanceListener>(pipe, name);
        listener->start();
        return {.forwarded = false, .listener = std::move(listener)};
      }
      // Un canal déjà tenu : ERROR_ACCESS_DENIED (FILE_FLAG_FIRST_PIPE_INSTANCE), ou ERROR_PIPE_BUSY quand sa
      // seule instance est prise (nMaxInstances = 1).
      if (const DWORD error = GetLastError(); error != ERROR_ACCESS_DENIED && error != ERROR_PIPE_BUSY)
        return {};
      if (!take_over) break;
      // C31-03 : le lecteur figé qui vient de relancer celui-ci tient encore le canal ; il sort sous peu.
      if (std::chrono::steady_clock::now() >= until) {
        log_channel("app")->warn("instance : canal toujours tenu après la relance, ce lancement démarre seul");
        return {};
      }
      std::this_thread::sleep_for(kSlice);
    }
    if (send_files(name, instance_message(files))) return {.forwarded = true, .listener = nullptr};
    const auto log = log_channel("app");
    log->warn("instance : le lecteur ouvert ne répond pas, ce lancement démarre seul");
  } catch (...) {  // NOLINT(bugprone-empty-catch) : sans instance unique, le lecteur démarre quand même
  }
  return {};
}

#else

namespace {

class Fd {
 public:
  explicit Fd(int fd = -1) : fd_(fd) {}
  Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) {
      reset();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() { reset(); }
  int get() const { return fd_; }
  explicit operator bool() const { return fd_ >= 0; }

 private:
  void reset() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }
  int fd_;
};

#ifdef MSG_NOSIGNAL
constexpr int kNoSignal = MSG_NOSIGNAL;  // un pair parti ne tue pas le lecteur par SIGPIPE
#else
constexpr int kNoSignal = 0;  // macOS : SO_NOSIGPIPE sur le socket
#endif

Fd unix_socket() {
  Fd fd(::socket(AF_UNIX, SOCK_STREAM, 0));
  if (!fd) return fd;
  ::fcntl(fd.get(), F_SETFD, FD_CLOEXEC);  // les enfants (banc, trtexec) n'en héritent pas
#ifdef SO_NOSIGPIPE
  const int on = 1;
  ::setsockopt(fd.get(), SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
  return fd;
}

std::optional<sockaddr_un> socket_address(const std::filesystem::path& path) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::string text = core::to_utf8(path);
  if (text.size() >= sizeof(address.sun_path)) {
    // A19-10 : le lecteur marche sans instance unique, mais le journal dit pourquoi.
    const auto log = log_channel("app");
    log->warn("instance : chemin du socket trop long ({} octets sur {}), instance unique éteinte : {}",
              text.size(), sizeof(address.sun_path) - 1, text);
    return std::nullopt;
  }
  std::memcpy(static_cast<char*>(address.sun_path), text.c_str(), text.size() + 1);
  return address;
}

const sockaddr* as_sockaddr(const sockaddr_un& address) {
  return reinterpret_cast<const sockaddr*>(&address);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

bool wait_readable(int fd, std::chrono::milliseconds timeout) {
  pollfd watched{.fd = fd, .events = POLLIN, .revents = 0};
  return ::poll(&watched, 1, static_cast<int>(timeout.count())) > 0;
}

bool write_all(int fd, std::string_view text) {
  while (!text.empty()) {
    const ssize_t sent = ::send(fd, text.data(), text.size(), kNoSignal);
    if (sent <= 0) return false;
    text.remove_prefix(static_cast<std::size_t>(sent));
  }
  return true;
}

}  // namespace

class InstanceListener {
 public:
  InstanceListener(Fd socket, std::filesystem::path path)
      : socket_(std::move(socket)), path_(std::move(path)) {}
  InstanceListener(const InstanceListener&) = delete;
  InstanceListener& operator=(const InstanceListener&) = delete;
  ~InstanceListener() {
    if (thread_.joinable()) {
      thread_.request_stop();
      thread_.join();
    }
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  void start() {
    auto started = start_thread("instance", ThreadReport([](const Refusal& refusal) {
                                  const auto log = log_channel("app");
                                  log->warn("instance : {}", refusal.detail);
                                }),
                                [this](const std::stop_token& stop, const ThreadReport&) { serve(stop); });
    if (started) thread_ = std::move(*started);
  }

 private:
  void serve(const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      if (!wait_readable(socket_.get(), kSlice)) continue;
      const Fd client(::accept(socket_.get(), nullptr, nullptr));
      if (!client) continue;
      std::string message;
      std::array<char, 4096> chunk{};
      while (!message_complete(message) && wait_readable(client.get(), std::chrono::milliseconds(2000))) {
        const ssize_t got = ::recv(client.get(), chunk.data(), chunk.size(), 0);
        if (got <= 0) break;
        message.append(chunk.data(), static_cast<std::size_t>(got));
      }
      const auto files = instance_message_files(message);
      const auto log = log_channel("app");
      log->info("instance : {} fichier(s) reçu(s) d'un autre lancement", files.size());
      hand_over(files, stop, [&] { write_all(client.get(), kAck); });
    }
  }

  Fd socket_;
  std::filesystem::path path_;
  std::jthread thread_;
};

namespace {

// Vrai si le lecteur qui tient le socket a accusé réception.
bool send_files(int fd, const std::string& message) {
  if (!write_all(fd, message)) return false;
  std::string reply;
  std::array<char, 16> chunk{};
  const auto until = std::chrono::steady_clock::now() + kAckWait;
  while (!reply.ends_with('\n') && std::chrono::steady_clock::now() < until) {
    if (!wait_readable(fd, std::chrono::duration_cast<std::chrono::milliseconds>(
                               until - std::chrono::steady_clock::now())))
      break;
    const ssize_t got = ::recv(fd, chunk.data(), chunk.size(), 0);
    if (got <= 0) break;
    reply.append(chunk.data(), static_cast<std::size_t>(got));
  }
  return reply == kAck;
}

}  // namespace

InstanceClaim claim_instance(const std::filesystem::path& channel_dir, std::string_view /*key*/,
                             const std::vector<std::string>& files, bool take_over) {
  try {
    const std::filesystem::path path = channel_dir / "instance.sock";
    const auto address = socket_address(path);
    if (!address) return {};
    // Deux lancements simultanés (A19-13) : sans verrou, le second peut trouver le socket du premier lié mais
    // pas encore à l'écoute (ECONNREFUSED), le prendre pour orphelin et le retirer. Le verrou tient de la
    // connexion à l'écoute ; il tombe avec le descripteur, même si le lecteur est tué.
    const Fd lock(::open((channel_dir / "instance.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600));
    if (lock) ::flock(lock.get(), LOCK_EX);
    // Deux tours : un socket resté d'un lecteur arrêté net se retire, et un lancement simultané qui vient
    // de le prendre se trouve au second. take_over : un tour de plus tant que l'ancien écoute, pendant
    // kTakeOverWait au plus (C31-03).
    const auto until = std::chrono::steady_clock::now() + kTakeOverWait;
    for (int attempt = 0; attempt < 2; ++attempt) {
      Fd client = unix_socket();
      if (!client) return {};
      if (::connect(client.get(), as_sockaddr(*address), sizeof(sockaddr_un)) == 0) {
        if (take_over) {
          if (std::chrono::steady_clock::now() >= until) {
            log_channel("app")->warn("instance : canal toujours tenu après la relance, ce lancement démarre seul");
            return {};
          }
          std::this_thread::sleep_for(kSlice);
          attempt = -1;  // l'ancien lecteur n'est pas encore sorti : un tour de plus
          continue;
        }
        if (send_files(client.get(), instance_message(files)))
          return {.forwarded = true, .listener = nullptr};
        const auto log = log_channel("app");
        log->warn("instance : le lecteur ouvert ne répond pas, ce lancement démarre seul");
        return {};
      }
      if (errno == ECONNREFUSED) ::unlink(path.c_str());  // personne derrière : le socket est orphelin
      Fd server = unix_socket();
      if (!server) return {};
      if (::bind(server.get(), as_sockaddr(*address), sizeof(sockaddr_un)) == 0 &&
          ::listen(server.get(), 4) == 0) {
        ::chmod(path.c_str(), 0600);
        auto listener = std::make_shared<InstanceListener>(std::move(server), path);
        listener->start();
        return {.forwarded = false, .listener = std::move(listener)};
      }
      if (errno != EADDRINUSE) return {};
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch) : sans instance unique, le lecteur démarre quand même
  }
  return {};
}

#endif

}  // namespace ajp::app::play
