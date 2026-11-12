#include "app/play/sleep_notice.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>

#include "core/log.h"

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

void SleepGate::resumed() {
  {
    const std::scoped_lock lock(mutex_);
    resumed_ = true;
  }
  if (wake_ != nullptr) wake_(context_);
}

bool SleepGate::take_resumed() {
  const std::scoped_lock lock(mutex_);
  return std::exchange(resumed_, false);
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
    if (!connect()) return;
    thread = std::jthread([this](const std::stop_token& stop) { run(stop); });
  }

  // Le bus système, le signal PrepareForSleep et le verrou « delay ». Faux (et rien d'ouvert) au premier
  // refus.
  bool connect() {
    if (sd.open_system(&bus) < 0) {
      bus = nullptr;
      log_channel("app")->info("veille : bus système injoignable, la pause se fera au réveil");
      return false;
    }
    if (sd.match_signal(bus, nullptr, kLogin, kLoginPath, kManager, "PrepareForSleep", &Listener::on_signal,
                        this) < 0) {
      log_channel("app")->info("veille : signal PrepareForSleep refusé, la pause se fera au réveil");
      disconnect();
      return false;
    }
    take_lock();
    return true;
  }

  void disconnect() {
    release_lock();
    if (bus != nullptr) sd.flush_close_unref(bus);
    bus = nullptr;
  }

  // Le fil d'écoute. Une erreur du bus (logind ou D-Bus redémarré, mise à jour de systemd sur une
  // distribution en publication continue) ne le tue pas : il se reconnecte, avec un délai qui double jusqu'à
  // une demi-minute, et le dit au journal. Sans cela plus aucune veille n'était annoncée, sans un mot.
  void run(const std::stop_token& stop) {
    constexpr std::uint64_t kWaitUs = 200'000;  // le fil revoit stop à ce rythme
    constexpr std::chrono::seconds kFirstRetry{1};
    constexpr std::chrono::seconds kLastRetry{30};
    constexpr std::chrono::milliseconds kSlice{200};
    while (!stop.stop_requested()) {
      int r = 0;
      do {
        r = sd.process(bus, nullptr);
      } while (r > 0 && !stop.stop_requested());
      if (r >= 0) {
        sd.wait(bus, kWaitUs);
        continue;
      }
      log_channel("app")->warn("veille : le bus de logind a répondu {} ; nouvelle connexion", r);
      disconnect();
      std::chrono::seconds delay = kFirstRetry;
      while (!stop.stop_requested()) {
        for (std::chrono::milliseconds waited{0}; waited < delay && !stop.stop_requested(); waited += kSlice)
          std::this_thread::sleep_for(kSlice);
        if (stop.stop_requested()) return;
        if (connect()) {
          log_channel("app")->info("veille : reconnecté à logind");
          break;
        }
        delay = std::min(delay * 2, kLastRetry);
      }
    }
  }

  ~Listener() {
    thread.request_stop();
    if (thread.joinable()) thread.join();
    disconnect();
  }

  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;

  void take_lock() {
    release_lock();  // un seul verrou : un signal « réveil » sans « veille » avant ne doit pas en perdre un
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
      listener.take_lock();     // réveil : le verrou pour la prochaine veille
      listener.gate.resumed();  // la veille est finie, ou annulée : l'annonce ne vaut plus
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
constexpr DWORD kNotifyCallback = 2;           // DEVICE_NOTIFY_CALLBACK
constexpr ULONG kSuspend = 4;                  // PBT_APMSUSPEND
constexpr ULONG kResumeSuspend = 7;            // PBT_APMRESUMESUSPEND
constexpr ULONG kResumeAutomatic = 18;         // PBT_APMRESUMEAUTOMATIC
constexpr ULONG kPowerSettingChange = 0x8013;  // PBT_POWERSETTINGCHANGE
using Register = DWORD(WINAPI*)(DWORD, HANDLE, PVOID*);
using Unregister = DWORD(WINAPI*)(PVOID);
// PowerSettingRegisterNotification (powrprof.dll, Windows 8) : un réglage d'alimentation, ici l'état de
// l'écran. Même rappel que ci-dessus ; `setting` y pointe un POWERBROADCAST_SETTING.
using SettingRegister = DWORD(WINAPI*)(const GUID*, DWORD, HANDLE, PVOID*);
using SettingUnregister = DWORD(WINAPI*)(PVOID);
struct BroadcastSetting {  // POWERBROADCAST_SETTING
  GUID guid;
  DWORD length;
  UCHAR data[1];
};
// GUID_CONSOLE_DISPLAY_STATE : 0 éteint, 1 allumé, 2 atténué.
constexpr GUID kConsoleDisplayState{
    0x6fe69556, 0x704a, 0x47a0, {0x8f, 0x24, 0xc2, 0x8d, 0x93, 0x6f, 0xda, 0x47}};

}  // namespace

struct SleepNotice::Listener {
  SleepGate& gate;
  HMODULE library = nullptr;
  Unregister unregister = nullptr;
  PVOID handle = nullptr;
  NotifyParameters parameters{};
  SettingUnregister unregister_setting = nullptr;
  PVOID display_handle = nullptr;
  NotifyParameters display_parameters{};

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
    // La veille moderne (S0) ne dit pas PBT_APMSUSPEND : l'écran qui s'éteint (couvercle fermé, bouton)
    // l'annonce à la place.
    const auto setting = reinterpret_cast<SettingRegister>(
        reinterpret_cast<void*>(GetProcAddress(library, "PowerSettingRegisterNotification")));
    unregister_setting = reinterpret_cast<SettingUnregister>(
        reinterpret_cast<void*>(GetProcAddress(library, "PowerSettingUnregisterNotification")));
    if (setting == nullptr || unregister_setting == nullptr) return;
    display_parameters = {.callback = &Listener::on_setting, .context = this};
    if (setting(&kConsoleDisplayState, kNotifyCallback, reinterpret_cast<HANDLE>(&display_parameters),
                &display_handle) != ERROR_SUCCESS) {
      display_handle = nullptr;
      log_channel("app")->info("veille : état de l'écran refusé par Windows, la pause se fera au réveil");
    }
  }

  ~Listener() {
    if (display_handle != nullptr) unregister_setting(display_handle);
    if (handle != nullptr) unregister(handle);
    if (library != nullptr) FreeLibrary(library);
  }

  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;

  // L'état de l'écran : éteint, la veille moderne peut suivre sans autre annonce ; rallumé, elle est finie.
  static ULONG CALLBACK on_setting(PVOID context, ULONG type, PVOID setting) {
    if (type != kPowerSettingChange || setting == nullptr) return ERROR_SUCCESS;
    const auto* change = static_cast<const BroadcastSetting*>(setting);
    if (std::memcmp(&change->guid, &kConsoleDisplayState, sizeof(GUID)) != 0 ||
        change->length < sizeof(DWORD))
      return ERROR_SUCCESS;
    DWORD state = 0;
    std::memcpy(&state, change->data, sizeof(state));
    auto& gate = static_cast<Listener*>(context)->gate;
    if (state == 0) {
      log_channel("app")->info("écran éteint par Windows : pause de la lecture (veille possible)");
      if (!gate.announce(kPauseLimit))
        log_channel("app")->warn("veille : la lecture ne s'est pas mise en pause à temps");
    } else if (state == 1) {
      gate.resumed();
    }
    return ERROR_SUCCESS;
  }

  static ULONG CALLBACK on_power(PVOID context, ULONG type, PVOID) {
    if (type == kResumeSuspend || type == kResumeAutomatic) {
      static_cast<Listener*>(context)->gate.resumed();
      return ERROR_SUCCESS;
    }
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
