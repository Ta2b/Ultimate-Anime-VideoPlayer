// Téléchargement par libcurl vers un fichier .part, que l'on reprend là où il
// s'est arrêté (en-tête Range) : après une coupure, au prochain essai, ou au
// prochain lancement. La taille attendue vient de l'index (core/model_index.h) ;
// le SHA-256 se vérifie ensuite, dans install.h, avant toute installation.
//
// Délais : connexion bornée, et une connexion qui ne reçoit presque plus rien
// pendant stall_time est coupée puis reprise. Pas de délai total : un paquet de
// plusieurs Gio sur une ligne lente doit pouvoir finir.
#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>

#include "core/types.h"

namespace ajp::app {

struct DownloadProgress {
  std::uint64_t received = 0;  // octets dans le .part, reprise comprise
  std::uint64_t total = 0;     // taille attendue
};

struct DownloadOptions {
  std::chrono::milliseconds connect_timeout{15'000};
  std::chrono::seconds stall_time{30};  // moins de stall_rate octets/s pendant ce temps : coupure
  std::uint32_t stall_rate = 512;       // octets par seconde
  int attempts = 4;                     // essais en tout, chacun reprend le .part
  std::chrono::milliseconds retry_delay{2'000};
  // Protocoles permis (liste libcurl), redirections comprises ; les tests locaux passent « http ».
  std::string protocols = "https";
  // Appelé depuis le thread qui télécharge, au plus tous les progress_step octets et à la fin.
  std::function<void(const DownloadProgress&)> on_progress;
  std::uint64_t progress_step = 1 << 20;
};

struct DownloadRequest {
  std::string url;
  std::filesystem::path part_file;  // créé ou repris ; jamais lu comme installé
  std::uint64_t size = 0;           // taille exacte attendue
};

// Possède l'initialisation globale de libcurl, comptée par libcurl elle-même :
// main en crée un avant le premier téléchargement et le passe à qui en a besoin.
class Downloader {
 public:
  // Refus DownloadFailed si libcurl ne s'initialise pas.
  static std::expected<Downloader, Refusal> create();
  Downloader(Downloader&& other) noexcept;
  Downloader& operator=(Downloader&& other) noexcept;
  Downloader(const Downloader&) = delete;
  Downloader& operator=(const Downloader&) = delete;
  ~Downloader();

  // Complète part_file jusqu'à request.size octets. Refus :
  //   FieldOutOfRange taille hors de core::limits::kDownloadSize, ou options incohérentes ;
  //   Cancelled       stop demandé (vérifié au moins une fois par seconde) ;
  //   DiskFull        place libre inférieure à ce qui reste à recevoir ;
  //   HashMismatch    le serveur envoie plus d'octets que la taille attendue ;
  //   FileUnwritable  le .part ne s'écrit pas ;
  //   DownloadFailed  réseau, HTTP (le code est cité) ou délai, après tous les essais.
  // Le .part reste en place pour la reprise, sauf s'il ne peut plus servir.
  std::expected<void, Refusal> fetch(const DownloadRequest& request, const DownloadOptions& options,
                                     const std::stop_token& stop = {}) const;

 private:
  Downloader() = default;
  bool owner_ = false;
};

}  // namespace ajp::app
