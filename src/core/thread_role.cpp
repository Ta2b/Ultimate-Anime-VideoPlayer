#include "core/thread_role.h"

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
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ajp {

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
  if (role == ThreadRole::Background) setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
#else
  static_cast<void>(role);
#endif
}

}  // namespace ajp
