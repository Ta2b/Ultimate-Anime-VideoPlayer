// Petit serveur HTTP/1.1 local pour les tests du téléchargement : 127.0.0.1, port
// choisi par le système, une connexion à la fois, « Connection: close ». Il sait
// reprendre (Range: bytes=N-), couper, se taire, échouer, pour exercer chaque
// chemin de app::Downloader sans réseau.
#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace ajp::app::test {

struct Route {
  std::string body;
  bool ranges = true;  // honore « Range: bytes=N- » par 206 ; sinon rend toujours 200 et tout le corps
  int status = 200;    // autre que 200 : réponse vide avec ce code
  int failures = 0;    // les premières réponses rendent 503
  std::size_t cut_after = std::numeric_limits<std::size_t>::max();  // coupe le corps après tant d'octets…
  int cuts = 0;                                                     // … pour les premières réponses servies
  std::chrono::milliseconds stall{0};                               // silence avant le corps…
  int stalls = 0;                                                   // … pour les premières réponses servies
};

class HttpServer {
 public:
  HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  ~HttpServer();

  void serve(const std::string& path, Route route);
  std::string url(std::string_view path) const;  // « http://127.0.0.1:port/path »
  int hits(const std::string& path) const;
  std::vector<std::string> ranges(const std::string& path) const;  // en-tête Range de chaque requête, "" sans

  // Un port où personne n'écoute, pour les refus de connexion.
  static int closed_port();

 private:
  struct Served {
    Route route;
    int hits = 0;
    std::vector<std::string> ranges;
  };
  void run(const std::stop_token& stop);
  void answer(std::intptr_t client, const std::stop_token& stop);

  std::intptr_t listener_ = -1;
  int port_ = 0;
  mutable std::mutex mutex_;
  std::map<std::string, Served> routes_;
  std::jthread thread_;
};

}  // namespace ajp::app::test
