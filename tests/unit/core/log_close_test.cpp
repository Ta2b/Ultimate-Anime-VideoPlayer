// close_log, dans un binaire à lui : il ferme spdlog pour tout le processus, les autres tests du
// journal (log_test.cpp) ne s'en remettraient pas.
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/basic_file_sink.h>

#include "core/log.h"

TEST_CASE("Journal : close_log vide le fichier, et un journal après lui ne plante pas (A18-01)", "[log]") {
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "ajp-log-close-test.log";
  std::filesystem::remove(file);
  // Comme main.cpp : un puits fichier derrière un journal asynchrone.
  ajp::install_log_sinks({std::make_shared<spdlog::sinks::basic_file_sink_mt>(file.string(), true)},
                         spdlog::level::info, true);
  ajp::log_channel("app")->info("ligne avant la fermeture");
  ajp::close_log();
  std::ifstream in(file);
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  CHECK(text.contains("ligne avant la fermeture"));
  // Un thread détaché qui répond après la fermeture : rien n'est écrit, rien ne plante.
  ajp::log_channel("app")->info("ligne après la fermeture");
  ajp::log_channel("nouveau")->warn("canal créé après la fermeture");
  ajp::close_log();  // une seconde fermeture (gestionnaire de std::terminate) ne plante pas non plus
  in.close();
  std::filesystem::remove(file);
}
