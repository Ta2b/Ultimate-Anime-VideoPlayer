#include "core/thread_role.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <optional>

#include "core/thread.h"

#if defined(_WIN32)
// windows.h avant avrt.h, qui en dépend.
// clang-format off
#include <windows.h>
#include <avrt.h>
// clang-format on
#elif defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ajp {

namespace {
// Posé une fois au démarrage.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<PriorityRaiser> g_raiser{nullptr};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<RoleRefusalReporter> g_refusal_reporter{nullptr};
// Un drapeau par rôle : le premier refus seulement se dit.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::array<std::atomic<bool>, 3> g_refusal_told{};

#if defined(__linux__)
// Le nice du thread principal avant sa hausse (C34-01) : sous Linux, la priorité est propre à chaque thread
// mais un thread ou un processus créé l'hérite de son créateur. Sans cette valeur rendue à chaque thread de
// module, la hausse du principal (nice -10, par SDL ou rtkit) passait à tous les threads créés ensuite et à
// leurs enfants (trtexec, curl) : la hiérarchie lecture > reste > fond n'existait que sous Windows, où
// MMCSS ne s'hérite pas. INT_MIN : pas de hausse posée, rien à rendre.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<int> g_base_nice{INT_MIN};

pid_t current_tid() noexcept {
  return static_cast<pid_t>(syscall(SYS_gettid));
}

// Le nice du thread appelant ; rien si le système ne le dit pas (getpriority rend -1, une valeur valable).
std::optional<int> own_nice() noexcept {
  errno = 0;
  const int nice = getpriority(PRIO_PROCESS, static_cast<id_t>(current_tid()));
  if (nice == -1 && errno != 0) return std::nullopt;
  return nice;
}
#endif

[[maybe_unused]] bool raise_if_set() noexcept {  // seul Linux s'en sert
  const PriorityRaiser raiser = g_raiser.load(std::memory_order_acquire);
  return raiser != nullptr && raiser();
}

bool refused(ThreadRole role, const char* what) noexcept {
  const auto index = static_cast<std::size_t>(role);
  const RoleRefusalReporter reporter = g_refusal_reporter.load(std::memory_order_acquire);
  if (reporter != nullptr && index < g_refusal_told.size() &&
      !g_refusal_told[index].exchange(true, std::memory_order_acq_rel))
    reporter(role, what);
  return false;
}
}  // namespace

void set_role_refusal_reporter(RoleRefusalReporter reporter) noexcept {
  g_refusal_reporter.store(reporter, std::memory_order_release);
}

bool set_thread_priority_raiser(PriorityRaiser raiser) noexcept {
#if defined(__linux__)
  // La base est lue avant la hausse, une fois : un second crochet ne prend pas le nice déjà relevé pour elle.
  if (raiser == nullptr) {
    g_base_nice.store(INT_MIN, std::memory_order_release);
  } else if (g_base_nice.load(std::memory_order_acquire) == INT_MIN) {
    if (const std::optional<int> nice = own_nice()) g_base_nice.store(*nice, std::memory_order_release);
  }
#endif
  g_raiser.store(raiser, std::memory_order_release);
#if defined(__linux__)
  return raise_if_set();
#else
  return false;  // Windows (MMCSS) et macOS (QoS) n'en ont pas besoin
#endif
}

bool set_thread_role(ThreadRole role) noexcept {
#if defined(_WIN32)
  switch (role) {
    case ThreadRole::Playback:
    case ThreadRole::Audio: {
      // MMCSS garde au thread sa part du processeur à cadence fixe, même sous une charge de fond. La tâche
      // reste inscrite jusqu'à la fin du thread.
      DWORD index = 0;
      if (AvSetMmThreadCharacteristicsW(role == ThreadRole::Audio ? L"Audio" : L"Playback", &index) ==
          nullptr)
        return refused(role, "MMCSS (AvSetMmThreadCharacteristicsW) refusé");
      return true;
    }
    case ThreadRole::Background:
      // Le mode de fond baisse aussi la priorité d'E/S et de mémoire du thread, comme le nice 10 de Linux
      // (A20-06) ; sans lui (refusé), la priorité basse seule.
      if (SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN) != 0) return true;
      if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) != 0)
        return refused(role, "mode de fond refusé, priorité basse seule");
      return refused(role, "mode de fond et priorité basse refusés");
  }
  return true;
#elif defined(__APPLE__)
  const qos_class_t qos = role == ThreadRole::Background ? QOS_CLASS_UTILITY : QOS_CLASS_USER_INTERACTIVE;
  if (pthread_set_qos_class_self_np(qos, 0) != 0)
    return refused(role, "classe de qualité de service refusée");
  return true;
#elif defined(__linux__)
  // Sous Linux, la priorité nice est propre à chaque thread (PRIO_PROCESS avec l'identifiant du thread).
  if (role == ThreadRole::Background) {
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10) != 0)
      return refused(role, "nice 10 refusé");
    return true;
  }
  // Sans crochet (les tests, avant SDL_Init), rien n'est demandé : rien de refusé non plus.
  if (g_raiser.load(std::memory_order_acquire) == nullptr) return true;
  if (!raise_if_set()) return refused(role, "priorité haute refusée (rtkit absent ou limite RLIMIT_RTPRIO)");
  return true;
#else
  static_cast<void>(role);
  return true;
#endif
}

void return_to_base_priority() noexcept {
#if defined(__linux__)
  const int base = g_base_nice.load(std::memory_order_acquire);
  if (base == INT_MIN) return;
  // Seulement une hausse héritée : un thread né d'un thread de fond (nice 10) le reste, comme FFmpeg.
  if (const std::optional<int> nice = own_nice(); nice && *nice < base)
    setpriority(PRIO_PROCESS, static_cast<id_t>(current_tid()), base);
#endif
}

void reserve_crash_stack() noexcept {
#if defined(_WIN32)
  ULONG guarantee = kCrashStackReserve;
  SetThreadStackGuarantee(&guarantee);
#endif
}

namespace {

// Le début de chaque thread de module (thread_start, core/thread.h) : nom, pile de secours, priorité de base.
void start_for_system([[maybe_unused]] const char* name) noexcept {
#if defined(_WIN32)
  // C34-05 : la place qu'un débordement de pile laisse au filtre des exceptions (crash_dump.cpp) ; sans
  // elle, il tourne sur la pile épuisée et le vidage n'est jamais écrit.
  reserve_crash_stack();
#endif
  return_to_base_priority();
  if (name == nullptr) return;
#if defined(_WIN32)
  std::array<wchar_t, 64> wide{};
  const int written =
      MultiByteToWideChar(CP_UTF8, 0, name, -1, wide.data(), static_cast<int>(wide.size()) - 1);
  // Pris dans kernel32 à l'exécution : SetThreadDescription n'existe que depuis Windows 10 1607.
  using Describe = HRESULT(WINAPI*)(HANDLE, PCWSTR);
  static const auto describe =
      reinterpret_cast<Describe>(  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
          reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription")));
  if (written > 0 && describe != nullptr) describe(GetCurrentThread(), wide.data());
#elif defined(__APPLE__)
  pthread_setname_np(name);
#elif defined(__linux__)
  // 16 octets, le zéro final compris : un nom plus long est refusé entier, il est donc coupé.
  std::array<char, 16> cut{};
  std::strncpy(cut.data(), name, cut.size() - 1);
  pthread_setname_np(pthread_self(), cut.data());
#endif
}

}  // namespace

void install_system_thread_names() noexcept {
  thread_start().store(&start_for_system, std::memory_order_release);
}

}  // namespace ajp
