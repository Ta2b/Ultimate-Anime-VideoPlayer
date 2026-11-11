// Écritures de fichiers hors du thread d'affichage (play.toml, position de reprise, mémoire du mode Auto) :
// un thread d'écriture qui ne garde, par fichier, que la dernière version demandée. Une écriture lente
// (disque qui dort, fsync, partage réseau) ne retient plus la présentation des images : l'attente de fin
// est bornée (flush), et un refus se rapporte à l'affichage (take_failure).
#pragma once

#include <chrono>
#include <condition_variable>
#include <expected>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "core/types.h"

namespace ajp::app::play {

class LatestWriter {
 public:
  using Job = std::function<std::expected<void, Refusal>()>;
  // Le temps que flush et la destruction donnent aux écritures (B30-05, B36-11) : au-delà, l'affichage
  // reprend, et l'écriture finit sur son thread.
  static constexpr std::chrono::milliseconds kFlushLimit{3000};

  // Sans thread (le système le refuse), chaque écriture se fait sur place, comme avant.
  LatestWriter();
  // Finit les écritures en attente (au plus kFlushLimit), puis arrête le thread. Une écriture encore en cours
  // passé ce délai continue seule ; wait_abandoned l'attend une dernière fois à la sortie du processus.
  ~LatestWriter();
  LatestWriter(const LatestWriter&) = delete;
  LatestWriter& operator=(const LatestWriter&) = delete;

  // L'écriture de key (un nom de fichier) : elle remplace celle de même clé pas encore faite. Un refus va au
  // journal et se reprend par take_failure.
  void post(const std::string& key, Job job);
  // Attend que toutes les écritures demandées soient faites, au plus limit ; faux si elles ne le sont pas.
  bool flush(std::chrono::milliseconds limit = kFlushLimit);

  // Une écriture refusée que l'affichage n'a pas encore vue : sa clé et son refus, une seule fois. Une
  // écriture de la même clé qui réussit ensuite efface le refus.
  struct Failure {
    std::string key;
    Refusal refusal;
  };
  std::optional<Failure> take_failure();
  // Dit par l'affichage : l'avis d'un refus ne paraît qu'une fois par écrivain.
  bool failure_said = false;

  // À la sortie du processus : attend, au plus limit, les écritures que des writers détruits ont laissées
  // en cours.
  static void wait_abandoned(std::chrono::milliseconds limit);

 private:
  struct State {
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable done;
    std::map<std::string, Job> pending;
    std::map<std::string, Refusal> failed;
    int running = 0;  // écritures sorties de pending et pas encore finies
    bool stop = false;
    bool idle() const { return pending.empty() && running == 0; }
  };
  static void run(const std::shared_ptr<State>& state);
  static void finish(State& state, const std::string& key, const std::expected<void, Refusal>& written);

  std::shared_ptr<State> state_;
  std::jthread thread_;
};

}  // namespace ajp::app::play
