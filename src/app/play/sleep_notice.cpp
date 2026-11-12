#include "app/play/sleep_notice.h"

#include <atomic>
#include <cstdint>
#include <stop_token>
#include <thread>
#include <type_traits>

#include "core/log.h"
#include "core/thread.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ajp::app::play {

bool SleepGate::announce(std::chrono::milliseconds limit) {
  std::unique_lock lock(mutex_);
  announced_ = true;
  ready_ = false;
  lock.unlock();
  if (wake_ != nullptr) wake_(context_);
  lock.lock();
  const bool done = changed_.wait_for(lock, limit, [&] { return ready_; });
  announced_ = false;  // prise ou non, l'annonce ne vaut plus après la veille
  return done;
}

bool SleepGate::take() {
  const std::scoped_lock lock(mutex_);
  if (!announced_) return false;
  announced_ = false;
  return true;
}

void SleepGate::ready() {
  {
    const std::scoped_lock lock(mutex_);
    ready_ = true;
  }
  changed_.notify_all();
}

#if defined(__linux__) || defined(_WIN32)
namespace {

// Le temps laissé à la boucle pour se mettre en pause : logind retient la veille jusqu'à InhibitDelayMaxSec
// (5 s par défaut), Windows attend le rappel environ 2 s. Rien sur macOS, hors chantier.
constexpr std::chrono::milliseconds kPauseLimit{1500};

}  // namespace
#endif

#if defined(__linux__)

namespace {

// Ce que le lecteur prend de sd-bus (libsystemd.so.0), chargé à l'exécution : un système sans systemd
// garde SleepWatch seul.
struct SdBus {
  using Bus = void;
  using Message = void;
  using Slot = void;
  struct Error {
    const char* name = nullptr;
    const char* message = nullptr;
    int need_free = 0;
  };
  using Handler = int (*)(Message*, void*, Error*);
  int (*open_system)(Bus**) = nullptr;
  Bus* (*flush_close_unref)(Bus*) = nullptr;
  int (*match_signal)(Bus*, Slot**, const char*, const char*, const char*, const char*, Handler,
                      void*) = nullptr;
  int (*call_method)(Bus*, const char*, const char*, const char*, const char*, Error*, Message**, const char*,
                     ...) = nullptr;
  int (*message_read)(Message*, const char*, ...) = nullptr;
  Message* (*message_unref)(Message*) = nullptr;
  int (*process)(Bus*, Message**) = nullptr;
  int (*wait)(Bus*, std::uint64_t) = nullptr;
  void (*error_free)(Error*) = nullptr;
  void* library = nullptr;

  bool load() {
    library = dlopen("libsystemd.so.0", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) return false;
    const auto get = [&](auto& into, const char* name) {
      into = reinterpret_cast<std::remove_reference_t<decltype(into)>>(dlsym(library, name));
      return into != nullptr;
    };
    return get(open_system, "sd_bus_open_system") && get(flush_close_unref, "sd_bus_flush_close_unref") &&
           get(match_signal, "sd_bus_match_signal") && get(call_method, "sd_bus_call_method") &&
           get(message_read, "sd_bus_message_read") && get(message_unref, "sd_bus_message_unref") &&
           get(process, "sd_bus_process") && get(wait, "sd_bus_wait") && get(error_free, "sd_bus_error_free");
  }
  ~SdBus() {
    if (library != nullptr) dlclose(library);
  }
};

constexpr const char* kLogin = "org.freedesktop.login1";
constexpr const char* kLoginPath = "/org/freedesktop/login1";
constexpr const char* kManager = "org.freedesktop.login1.Manager";

}  // namespace

struct SleepNotice::Listener {
  SleepGate& gate;
  SdBus sd;
  SdBus::Bus* bus = nullptr;
  int lock_fd = -1;  // le verrou « delay » de logind ; -1 : aucun
  std::atomic<bool> sleeping{false};
  std::jthread thread;  // en dernier

  explicit Listener(SleepGate& g) : gate(g) {
    if (!sd.load()) {
      log_channel("app")->info("veille : libsystemd absente, la pause se fera au réveil");
      return;
    }
    if (sd.open_system(&bus) < 0) {
      bus = nullptr;
      log_channel("app")->info("veille : bus système injoignable, la pause se fera au réveil");
      return;
    }
    if (sd.match_signal(bus, nullptr, kLogin, kLoginPath, kManager, "PrepareForSleep", &Listener::on_signal,
                        this) < 0) {
      log_channel("app")->info("veille : signal PrepareForSleep refusé, la pause se fera au réveil");
      return;
    }
    take_lock();  // avant le fil : sd-bus ne se partage pas entre deux threads
    // Par start_thread (C30-08) : sans fil, la veille se voit au réveil, comme sans bus.
    auto started = start_thread("veille", ThreadReport([](const Refusal&) {}),
                                [this](const std::stop_token& stop, const ThreadReport&) {
                                  constexpr std::uint64_t kWaitUs =
                                      200'000;  // le fil revoit stop à ce rythme
                                  while (!stop.stop_requested()) {
                                    int r = 0;
                                    do {
                                      r = sd.process(bus, nullptr);
                                    } while (r > 0 && !stop.stop_requested());
                                    if (r < 0) break;
                                    sd.wait(bus, kWaitUs);
                                  }
                                });
    if (!started) {
      release_lock();
      log_channel("app")->info("veille : fil refusé ({}), la pause se fera au réveil",
                               started.error().detail);
      return;
    }
    thread = std::move(*started);
  }

  ~Listener() {
    thread.request_stop();
    if (thread.joinable()) thread.join();
    release_lock();
    if (bus != nullptr) sd.flush_close_unref(bus);
  }

  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;

  void take_lock() {
    SdBus::Error error;
    SdBus::Message* reply = nullptr;
    const int r = sd.call_method(bus, kLogin, kLoginPath, kManager, "Inhibit", &error, &reply, "ssss",
                                 "sleep", "ajp", "Pause the video before sleep", "delay");
    if (r < 0) {
      log_channel("app")->info("veille : verrou de logind refusé ({}), la pause se fera au réveil",
                               error.message != nullptr ? error.message : "?");
      sd.error_free(&error);
      return;
    }
    int fd = -1;
    if (sd.message_read(reply, "h", &fd) >= 0 && fd >= 0) lock_fd = fcntl(fd, F_DUPFD_CLOEXEC, 3);
    sd.message_unref(reply);
    sd.error_free(&error);
  }

  void release_lock() {
    if (lock_fd < 0) return;
    close(lock_fd);
    lock_fd = -1;
  }

  static int on_signal(SdBus::Message* message, void* self, SdBus::Error*) {
    auto& listener = *static_cast<Listener*>(self);
    int going = 0;
    if (listener.sd.message_read(message, "b", &going) < 0) return 0;
    if (going != 0) {
      log_channel("app")->info("veille annoncée par logind : pause de la lecture");
      if (!listener.gate.announce(kPauseLimit))
        log_channel("app")->warn("veille : la lecture ne s'est pas mise en pause à temps");
      listener.release_lock();  // la veille peut suivre
    } else {
      listener.take_lock();  // réveil : le verrou pour la prochaine veille
    }
    return 0;
  }
};

#elif defined(_WIN32)

namespace {

// powrprof.h, recopiés pour ne dépendre ni de la version du SDK ni de powrprof.lib.
struct NotifyParameters {
  ULONG(CALLBACK* callback)(PVOID context, ULONG type, PVOID setting);
  PVOID context;
};
constexpr DWORD kNotifyCallback = 2;  // DEVICE_NOTIFY_CALLBACK
constexpr ULONG kSuspend = 4;         // PBT_APMSUSPEND
using Register = DWORD(WINAPI*)(DWORD, HANDLE, PVOID*);
using Unregister = DWORD(WINAPI*)(PVOID);

}  // namespace

struct SleepNotice::Listener {
  SleepGate& gate;
  HMODULE library = nullptr;
  Unregister unregister = nullptr;
  PVOID handle = nullptr;
  NotifyParameters parameters{};

  explicit Listener(SleepGate& g) : gate(g) {
    library = LoadLibraryExW(L"powrprof.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (library == nullptr) return;
    const auto reg = reinterpret_cast<Register>(
        reinterpret_cast<void*>(GetProcAddress(library, "PowerRegisterSuspendResumeNotification")));
    unregister = reinterpret_cast<Unregister>(
        reinterpret_cast<void*>(GetProcAddress(library, "PowerUnregisterSuspendResumeNotification")));
    if (reg == nullptr || unregister == nullptr) return;
    parameters = {.callback = &Listener::on_power, .context = this};
    if (reg(kNotifyCallback, reinterpret_cast<HANDLE>(&parameters), &handle) != ERROR_SUCCESS) {
      handle = nullptr;
      log_channel("app")->info("veille : annonce de Windows refusée, la pause se fera au réveil");
    }
  }

  ~Listener() {
    if (handle != nullptr) unregister(handle);
    if (library != nullptr) FreeLibrary(library);
  }

  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;

  static ULONG CALLBACK on_power(PVOID context, ULONG type, PVOID) {
    if (type != kSuspend) return ERROR_SUCCESS;
    log_channel("app")->info("veille annoncée par Windows : pause de la lecture");
    if (!static_cast<Listener*>(context)->gate.announce(kPauseLimit))
      log_channel("app")->warn("veille : la lecture ne s'est pas mise en pause à temps");
    return ERROR_SUCCESS;
  }
};

#else

struct SleepNotice::Listener {
  explicit Listener(SleepGate&) {}
};

#endif

SleepNotice::SleepNotice(void (*wake)(void*), void* context)
    : gate_(wake, context), listener_(std::make_unique<Listener>(gate_)) {}

SleepNotice::~SleepNotice() = default;

}  // namespace ajp::app::play
