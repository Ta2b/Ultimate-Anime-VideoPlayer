// Le thread d'écriture (latest_writer.h) : par fichier, la dernière version demandée ; flush attend tout.
#include "app/play/latest_writer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/save_failure.h"

namespace ajp::app::play {

TEST_CASE("Écritures : la dernière version de chaque fichier, faite avant la fin de flush", "[app][writer]") {
  std::mutex mutex;
  std::vector<std::string> written;
  std::atomic<bool> release{false};
  LatestWriter writer;
  // La première écriture tient le thread : les suivantes de même clé se remplacent pendant ce temps.
  writer.post("a", [&]() -> std::expected<void, Refusal> {
    while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::scoped_lock lock(mutex);
    written.emplace_back("a0");
    return {};
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (int i = 1; i <= 3; ++i) {
    writer.post("a", [&, i]() -> std::expected<void, Refusal> {
      const std::scoped_lock lock(mutex);
      written.push_back("a" + std::to_string(i));
      return {};
    });
  }
  writer.post("b", [&]() -> std::expected<void, Refusal> {
    const std::scoped_lock lock(mutex);
    written.emplace_back("b");
    return std::unexpected(Refusal{RefusalCode::FileUnwritable, "refus pour le journal"});
  });
  release = true;
  writer.flush();
  const std::scoped_lock lock(mutex);
  REQUIRE(written.size() == 3);
  CHECK(written[0] == "a0");
  CHECK(std::ranges::find(written, "a3") != written.end());
  CHECK(std::ranges::find(written, "b") != written.end());
  CHECK(std::ranges::find(written, "a1") == written.end());
}

TEST_CASE("Écritures : un job qui lève ne bloque ni flush ni les suivantes", "[app][writer]") {
  LatestWriter writer;
  writer.post("a", []() -> std::expected<void, Refusal> { throw std::runtime_error("disque"); });
  writer.flush();
  std::atomic<bool> done{false};
  writer.post("b", [&]() -> std::expected<void, Refusal> {
    done = true;
    return {};
  });
  writer.flush();
  CHECK(done.load());
}

TEST_CASE("Écritures : flush borné, une écriture lente ne retient pas l'affichage (B30-05, B36-11)",
          "[app][writer]") {
  std::atomic<bool> release{false};
  std::atomic<bool> finished{false};
  {
    LatestWriter writer;
    writer.post("lente", [&]() -> std::expected<void, Refusal> {
      while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      finished = true;
      return {};
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto begun = std::chrono::steady_clock::now();
    CHECK_FALSE(writer.flush(std::chrono::milliseconds(100)));
    const auto waited = std::chrono::steady_clock::now() - begun;
    CHECK(waited < std::chrono::seconds(2));  // la borne, pas la durée de l'écriture
    release = true;
    CHECK(writer.flush(std::chrono::seconds(5)));  // finie : flush le voit
    CHECK(finished.load());
  }
}

TEST_CASE("Écritures : un refus se rapporte une fois, une réussite l'efface (B31-08, B36-01)",
          "[app][writer]") {
  LatestWriter writer;
  CHECK(writer.take_failures().empty());
  writer.post("play.toml", []() -> std::expected<void, Refusal> {
    return std::unexpected(Refusal{RefusalCode::FileUnwritable, "disque plein"});
  });
  REQUIRE(writer.flush());
  const auto failed = writer.take_failures();
  REQUIRE(failed.size() == 1);
  CHECK(failed[0].key == "play.toml");
  CHECK(failed[0].refusal.code == RefusalCode::FileUnwritable);
  CHECK(writer.take_failures().empty());  // une seule fois
  // Refusé, puis réussi avant que l'affichage ne l'ait vu : rien à dire.
  writer.post("resume", []() -> std::expected<void, Refusal> {
    return std::unexpected(Refusal{RefusalCode::FileUnwritable, "verrou"});
  });
  REQUIRE(writer.flush());
  writer.post("resume", []() -> std::expected<void, Refusal> { return {}; });
  REQUIRE(writer.flush());
  CHECK(writer.take_failures().empty());
}

// C22-02 : deux clés refusées ensemble sont relevées toutes les deux, dans l'ordre des clés.
TEST_CASE("Écritures : tous les refus sont relevés d'un coup (C22-02)", "[app][writer]") {
  LatestWriter writer;
  const auto fail = []() -> std::expected<void, Refusal> {
    return std::unexpected(Refusal{RefusalCode::FileUnwritable, "disque plein"});
  };
  writer.post("auto_memory", fail);
  writer.post("play.toml", fail);
  REQUIRE(writer.flush());
  const auto failed = writer.take_failures();
  REQUIRE(failed.size() == 2);
  CHECK(failed[0].key == "auto_memory");
  CHECK(failed[1].key == "play.toml");
  CHECK(writer.take_failures().empty());
}

// C22-02 : à la sortie, le dernier essai de play.toml a lieu même si une autre clé a échoué avant lui.
TEST_CASE("Écritures : un refus d'une autre clé ne cache pas celui de play.toml à la sortie (C22-02)",
          "[app][writer]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-finish-writes";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  LatestWriter writer;
  const auto fail = []() -> std::expected<void, Refusal> {
    return std::unexpected(Refusal{RefusalCode::FileUnwritable, "disque plein"});
  };
  writer.post("auto_memory", fail);
  writer.post("play.toml", fail);
  finish_settings_writes(writer, dir / "play.toml", PlaySettings{});
  CHECK(std::filesystem::exists(dir / "play.toml"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("Écritures : détruit pendant une écriture lente, le writer rend la main et la finit seule",
          "[app][writer]") {
  std::atomic<bool> release{false};
  std::atomic<bool> finished{false};
  {
    LatestWriter writer;
    writer.post("lente", [&]() -> std::expected<void, Refusal> {
      while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      finished = true;
      return {};
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    release = true;  // l'écriture finit pendant la destruction ou juste après
  }
  LatestWriter::wait_abandoned(std::chrono::seconds(5));
  CHECK(finished.load());
}

}  // namespace ajp::app::play
