#include "core/thread_role.h"

#include <array>
#include <atomic>
#include <cstring>

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

[[maybe_unused]] bool raise_if_set() noexcept {  // seul Linux s'en sert
  const PriorityRaiser raiser = g_raiser.load(std::memory_order_acquire);
  return raiser != nullptr && raiser();
}
}  // namespace

bool set_thread_priority_raiser(PriorityRaiser raiser) noexcept {
  g_raiser.store(raiser, std::memory_order_release);
#if defined(__linux__)
  return raise_if_set();
#else
  return false;  // Windows (MMCSS) et macOS (QoS) n'en ont pas besoin
#endif
}

void set_thread_role(ThreadRole role) noexcept {
#if defined(_WIN32)
  switch (role) {
    case ThreadRole::Playback:
    case ThreadRole::Audio: {
      // MMCSS garde au thread sa part du processeur à cadence fixe, même sous une charge de fond. La tâche
      // reste inscrite jusqu'à la fin du thread.
      DWORD index = 0;
      AvSetMmThreadCharacteristicsW(role == ThreadRole::Audio ? L"Audio" : L"Playback", &index);
      break;
    }
    case ThreadRole::Background:
      // Le mode de fond baisse aussi la priorité d'E/S et de mémoire du thread, comme le nice 10 de Linux
      // (A20-06) ; sans lui (refusé), la priorité basse seule.
      if (SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN) == 0)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
      break;
  }
#elif defined(__APPLE__)
  switch (role) {
    case ThreadRole::Playback:
    case ThreadRole::Audio:
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      break;
    case ThreadRole::Background:
      pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
      break;
  }
#elif defined(__linux__)
  // Sous Linux, la priorité nice est propre à chaque thread (PRIO_PROCESS avec l'identifiant du thread).
  if (role == ThreadRole::Background) {
    setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
  } else {
    raise_if_set();
  }
#else
  static_cast<void>(role);
#endif
}

namespace {

void name_for_system([[maybe_unused]] const char* name) noexcept {
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
  thread_namer().store(&name_for_system, std::memory_order_release);
}

}  // namespace ajp
