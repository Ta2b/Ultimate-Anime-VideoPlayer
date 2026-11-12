// L'annonce de la veille du système, avant qu'elle n'arrive (B31-01, B08-10 ; décision 7 : pause avant la
// veille, recalage au réveil). SleepWatch (sleep_watch.h) ne voit la veille qu'au réveil, par le trou dans
// le temps ; ici le système la dit avant :
//
// - Linux : logind, par son bus système (sd-bus de libsystemd, chargée à l'exécution). Un verrou « delay »
//   (Inhibit) retient la veille le temps que la lecture se mette en pause ; le signal PrepareForSleep(true)
//   l'annonce, le verrou est rendu dès la pause faite, et repris au réveil (PrepareForSleep(false)).
// - Windows : PowerRegisterSuspendResumeNotification (powrprof.dll), PBT_APMSUSPEND. Le rappel vient d'un fil
//   du système, qui attend la pause un moment borné.
// - macOS : rien (hors de la cible) ; SleepWatch reste le filet partout.
//
// L'annonce passe au thread main par SleepGate : le fil du système appelle announce, qui réveille la boucle
// et attend (borné) que la boucle ait mis en pause et appelé ready.
#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace ajp::app::play {

class SleepGate {
 public:
  SleepGate(void (*wake)(void*), void* context) : wake_(wake), context_(context) {}

  // Fil du système : la veille arrive. Réveille la boucle, puis attend au plus `limit` qu'elle soit prête.
  // Vrai si elle l'a été.
  bool announce(std::chrono::milliseconds limit);
  // Thread main : une veille annoncée, une fois.
  bool take();
  // Thread main : la lecture est en pause, la veille peut suivre.
  void ready();

 private:
  void (*wake_)(void*);
  void* context_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool announced_ = false;  // à prendre par la boucle
  bool ready_ = false;      // la boucle a fini
};

class SleepNotice {
 public:
  // wake(context) réveille la boucle de lecture ; il doit vivre plus longtemps que SleepNotice.
  SleepNotice(void (*wake)(void*), void* context);
  ~SleepNotice();
  SleepNotice(const SleepNotice&) = delete;
  SleepNotice& operator=(const SleepNotice&) = delete;

  bool take() { return gate_.take(); }
  void ready() { gate_.ready(); }

 private:
  SleepGate gate_;
  struct Listener;
  std::unique_ptr<Listener> listener_;  // en dernier : arrêté avant la porte
};

}  // namespace ajp::app::play
