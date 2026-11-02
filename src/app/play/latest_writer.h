// Écritures de fichiers hors du thread d'affichage (play.toml, position de reprise, mémoire du mode Auto) :
// un thread d'écriture qui ne garde, par fichier, que la dernière version demandée. Une écriture lente
// (disque qui dort, fsync, partage réseau) ne retient plus la présentation des images.
#pragma once

#include <condition_variable>
#include <expected>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "core/types.h"

namespace ajp::app::play {

class LatestWriter {
 public:
  using Job = std::function<std::expected<void, Refusal>()>;

  // Sans thread (le système le refuse), chaque écriture se fait sur place, comme avant.
  LatestWriter();
  // Finit les écritures en attente, puis arrête le thread.
  ~LatestWriter();
  LatestWriter(const LatestWriter&) = delete;
  LatestWriter& operator=(const LatestWriter&) = delete;

  // L'écriture de key (un nom de fichier) : elle remplace celle de même clé pas encore faite. Un refus va au
  // journal.
  void post(const std::string& key, Job job);
  // Attend que toutes les écritures demandées soient faites.
  void flush();

 private:
  void run(const std::stop_token& stop);

  std::mutex mutex_;
  std::condition_variable_any wake_;
  std::condition_variable done_;
  std::map<std::string, Job> pending_;
  int running_ = 0;  // écritures sorties de pending_ et pas encore finies
  std::jthread thread_;
};

}  // namespace ajp::app::play
