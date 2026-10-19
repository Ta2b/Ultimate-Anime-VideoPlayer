// Téléchargement contre le serveur HTTP local du test : reprise, coupures, délais,
// annulation, refus nommés. Aucun accès réseau.
#include "app/download.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "http_server.h"
#include "support.h"

using namespace ajp;
using namespace ajp::app;
using namespace ajp::app::test;
using namespace std::chrono_literals;

namespace {

constexpr std::size_t kSize = 300 * 1024;

DownloadOptions local_options() {
  return DownloadOptions{.connect_timeout = 2'000ms,
                         .stall_time = 1s,
                         .stall_rate = 1024,
                         .attempts = 3,
                         .retry_delay = 10ms,
                         .protocols = "http",
                         .progress_step = 16 * 1024};
}

const Downloader& downloader() {
  static const Downloader instance = *Downloader::create();
  return instance;
}

struct Fixture {
  TempDir dir;
  HttpServer server;
  std::string body = test_bytes(kSize);
  DownloadRequest request{.part_file = dir / "downloads" / "file.part", .size = kSize};

  explicit Fixture(Route route) {
    if (route.body.empty() && route.status == 200) route.body = body;
    server.serve("/file", std::move(route));
    request.url = server.url("/file");
  }
};

}  // namespace

TEST_CASE("Téléchargement complet, avec la progression jusqu'à la taille de l'index", "[app][download]") {
  Fixture f{Route{}};
  std::vector<DownloadProgress> seen;
  DownloadOptions options = local_options();
  options.on_progress = [&](const DownloadProgress& p) { seen.push_back(p); };

  REQUIRE(downloader().fetch(f.request, options));
  CHECK(read_file(f.request.part_file) == f.body);
  CHECK(f.server.hits("/file") == 1);
  REQUIRE_FALSE(seen.empty());
  CHECK(seen.back().received == kSize);
  CHECK(seen.back().total == kSize);
  CHECK(std::ranges::is_sorted(seen, {}, &DownloadProgress::received));

  // Déjà complet : rien ne se redemande.
  REQUIRE(downloader().fetch(f.request, options));
  CHECK(f.server.hits("/file") == 1);
}

TEST_CASE("Téléchargement : une coupure se reprend là où elle s'est arrêtée", "[app][download]") {
  Fixture f{Route{.cut_after = 100'000, .cuts = 1}};
  REQUIRE(downloader().fetch(f.request, local_options()));
  CHECK(read_file(f.request.part_file) == f.body);
  CHECK(f.server.ranges("/file") == std::vector<std::string>{"", "100000-"});
}

TEST_CASE("Téléchargement : un .part d'un lancement précédent se reprend", "[app][download]") {
  Fixture f{Route{}};
  write_file(f.request.part_file, std::string_view(f.body).substr(0, 4096));
  REQUIRE(downloader().fetch(f.request, local_options()));
  CHECK(read_file(f.request.part_file) == f.body);
  CHECK(f.server.ranges("/file") == std::vector<std::string>{"4096-"});
}

TEST_CASE("Téléchargement : un serveur qui ne sait pas reprendre fait repartir de zéro", "[app][download]") {
  Fixture f{Route{.ranges = false}};
  write_file(f.request.part_file, std::string(4096, 'x'));  // reste faux d'un autre essai
  REQUIRE(downloader().fetch(f.request, local_options()));
  CHECK(read_file(f.request.part_file) == f.body);
  CHECK(f.server.hits("/file") == 2);
  CHECK(f.server.ranges("/file") == std::vector<std::string>{"4096-", ""});
}

TEST_CASE("Téléchargement : un .part plus long que l'attendu est jeté", "[app][download]") {
  Fixture f{Route{}};
  write_file(f.request.part_file, std::string(kSize + 1, 'x'));
  REQUIRE(downloader().fetch(f.request, local_options()));
  CHECK(read_file(f.request.part_file) == f.body);
}

TEST_CASE("Téléchargement : une panne passagère se réessaie, une erreur HTTP définitive non",
          "[app][download]") {
  SECTION("503 deux fois, puis le fichier") {
    Fixture f{Route{.failures = 2}};
    REQUIRE(downloader().fetch(f.request, local_options()));
    CHECK(f.server.hits("/file") == 3);
  }
  SECTION("503 à chaque essai : refus DownloadFailed qui cite le dernier code") {
    Fixture f{Route{.failures = 10}};
    const auto result = downloader().fetch(f.request, local_options());
    REQUIRE_FALSE(result);
    CHECK(result.error().code == RefusalCode::DownloadFailed);
    CHECK(result.error().detail.find("3 essais") != std::string::npos);
    CHECK(result.error().detail.find("HTTP 503") != std::string::npos);
    CHECK(f.server.hits("/file") == 3);
  }
  SECTION("404 : refus immédiat") {
    Fixture f{Route{.status = 404}};
    const auto result = downloader().fetch(f.request, local_options());
    REQUIRE_FALSE(result);
    CHECK(result.error().code == RefusalCode::DownloadFailed);
    CHECK(result.error().detail.find("HTTP 404") != std::string::npos);
    CHECK(f.server.hits("/file") == 1);
  }
}

TEST_CASE("Téléchargement : une connexion muette est coupée puis reprise", "[app][download]") {
  // Premier essai : 4 s de silence, coupé après stall_time (1 s) ; deuxième : le fichier.
  Fixture f{Route{.stall = 4'000ms, .stalls = 1}};
  const auto start = std::chrono::steady_clock::now();
  REQUIRE(downloader().fetch(f.request, local_options()));
  CHECK(read_file(f.request.part_file) == f.body);
  CHECK(f.server.hits("/file") == 2);
  CHECK(std::chrono::steady_clock::now() - start < 4s);
}

TEST_CASE("Téléchargement : l'annulation s'entend en moins d'une seconde, le .part reste",
          "[app][download]") {
  Fixture f{Route{.stall = 10'000ms, .stalls = 1}};
  std::stop_source stop;
  const std::jthread canceller([&] {
    std::this_thread::sleep_for(300ms);
    stop.request_stop();
  });
  DownloadOptions options = local_options();
  options.stall_time = 30s;
  const auto start = std::chrono::steady_clock::now();
  const auto result = downloader().fetch(f.request, options, stop.get_token());
  REQUIRE_FALSE(result);
  CHECK(result.error().code == RefusalCode::Cancelled);
  CHECK(std::chrono::steady_clock::now() - start < 3s);
  CHECK(fs::exists(f.request.part_file));
}

TEST_CASE("Téléchargement : refus nommés du serveur, du protocole et du disque", "[app][download]") {
  SECTION("plus d'octets que l'index : HashMismatch, et le .part est jeté") {
    Fixture f{Route{.body = test_bytes(kSize + 10)}};
    const auto result = downloader().fetch(f.request, local_options());
    REQUIRE_FALSE(result);
    CHECK(result.error().code == RefusalCode::HashMismatch);
    CHECK_FALSE(fs::exists(f.request.part_file));
  }
  SECTION("http refusé quand seul https est permis, sans aucune requête") {
    Fixture f{Route{}};
    DownloadOptions options = local_options();
    options.protocols = "https";
    const auto result = downloader().fetch(f.request, options);
    REQUIRE_FALSE(result);
    CHECK(result.error().code == RefusalCode::DownloadFailed);
    CHECK(f.server.hits("/file") == 0);
  }
  SECTION("liste de protocoles refusée par libcurl : refus, jamais tous les protocoles permis") {
    Fixture f{Route{}};
    DownloadOptions options = local_options();
    options.protocols = "protocole-inconnu";
    const auto result = downloader().fetch(f.request, options);
    REQUIRE_FALSE(result);
    CHECK(result.error().code == RefusalCode::DownloadFailed);
    CHECK(result.error().detail.find("CURLOPT_PROTOCOLS_STR") != std::string::npos);
    CHECK(f.server.hits("/file") == 0);
  }
  SECTION("personne n'écoute : DownloadFailed après tous les essais") {
    TempDir dir;
    const DownloadRequest request{.url = std::format("http://127.0.0.1:{}/file", HttpServer::closed_port()),
                                  .part_file = dir / "file.part",
                                  .size = 10};
    const auto result = downloader().fetch(request, local_options());
    REQUIRE_FALSE(result);
    CHECK(result.error().code == RefusalCode::DownloadFailed);
  }
  SECTION("dossier du .part impossible à créer : FileUnwritable") {
    Fixture f{Route{}};
    write_file(f.dir / "blocked", "fichier, pas dossier");
    f.request.part_file = f.dir / "blocked" / "file.part";
    const auto result = downloader().fetch(f.request, local_options());
    REQUIRE_FALSE(result);
    CHECK(result.error().code == RefusalCode::FileUnwritable);
  }
  SECTION("taille hors de l'index ou options incohérentes : FieldOutOfRange") {
    Fixture f{Route{}};
    f.request.size = 0;
    CHECK(downloader().fetch(f.request, local_options()).error().code == RefusalCode::FieldOutOfRange);
    f.request.size = kSize;
    DownloadOptions options = local_options();
    options.attempts = 0;
    CHECK(downloader().fetch(f.request, options).error().code == RefusalCode::FieldOutOfRange);
  }
}
