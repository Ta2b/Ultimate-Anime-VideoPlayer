#include "core/log.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/callback_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include "core/log_file.h"
#include "core/log_redact.h"

namespace {

// Un puits qui garde « canal : message » de chaque ligne reçue.
class Capture {
 public:
  spdlog::sink_ptr sink() {
    return std::make_shared<spdlog::sinks::callback_sink_mt>([this](const spdlog::details::log_msg& msg) {
      const std::scoped_lock lock(mutex_);
      lines_.push_back(std::string(msg.logger_name.begin(), msg.logger_name.end()) + " : " +
                       std::string(msg.payload.begin(), msg.payload.end()));
    });
  }
  std::vector<std::string> lines() const {
    const std::scoped_lock lock(mutex_);
    return lines_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::string> lines_;
};

}  // namespace

TEST_CASE("Journal : un puits installé après coup reçoit les messages d'un canal déjà créé", "[log]") {
  Capture first;
  Capture later;
  ajp::install_log_sinks({first.sink()}, spdlog::level::info);
  const std::shared_ptr<spdlog::logger> media = ajp::log_channel("media");
  media->info("avant");

  ajp::install_log_sinks({later.sink()}, spdlog::level::info);
  media->info("après");
  ajp::log_channel("audio")->warn("nouveau canal");

  CHECK(first.lines() == std::vector<std::string>{"media : avant"});
  CHECK(later.lines() == std::vector<std::string>{"media : après", "audio : nouveau canal"});
}

TEST_CASE("Journal : le niveau installé filtre les canaux créés ensuite", "[log]") {
  Capture capture;
  ajp::install_log_sinks({capture.sink()}, spdlog::level::warn);
  const std::shared_ptr<spdlog::logger> subs = ajp::log_channel("subs");
  subs->info("ignoré");
  subs->error("gardé");
  CHECK(capture.lines() == std::vector<std::string>{"subs : gardé"});
  CHECK(subs->level() == spdlog::level::warn);
}

TEST_CASE("Journal : un canal n'entre pas au registre de spdlog", "[log]") {
  ajp::install_log_sinks({std::make_shared<spdlog::sinks::null_sink_mt>()}, spdlog::level::info);
  const std::shared_ptr<spdlog::logger> channel = ajp::log_channel("prepare");
  CHECK(channel->name() == "prepare");
  CHECK(spdlog::get("prepare") == nullptr);
  CHECK(channel == ajp::log_channel("prepare"));  // 1.11 : un canal par module, gardé
}

TEST_CASE("Journal : un seul canal par module, refait quand les puits changent", "[core][log]") {
  Capture capture;
  ajp::install_log_sinks({capture.sink()}, spdlog::level::info);
  const auto first = ajp::log_channel("unique");
  CHECK(ajp::log_channel("unique") == first);  // le même logger, pas un nouveau à chaque appel
  CHECK(ajp::log_channel("autre") != first);
  // Un niveau réinstallé : le canal se refait et suit le niveau.
  ajp::install_log_sinks({capture.sink()}, spdlog::level::warn);
  const auto again = ajp::log_channel("unique");
  CHECK(again != first);
  CHECK(again->level() == spdlog::level::warn);
  CHECK(ajp::log_channel("unique") == again);
  ajp::install_log_sinks({std::make_shared<spdlog::sinks::null_sink_mt>()}, spdlog::level::info);
}

TEST_CASE("Journal nettoyé : le dossier personnel devient ~ sous ses deux séparateurs", "[log]") {
  ajp::set_log_home("C:\\Users\\tatoub\\");
  CHECK(ajp::redact_log_text("ouvert C:\\Users\\tatoub\\Videos\\a.mkv et C:/Users/tatoub/x") ==
        "ouvert ~\\Videos\\a.mkv et ~/x");
  ajp::set_log_home("/");
  CHECK(ajp::redact_log_text("/usr/lib") == "/usr/lib");
  ajp::set_log_home("");
}

TEST_CASE("Journal nettoyé : frontière de composant, disques montés, casse sous Windows (A20-03)", "[log]") {
  ajp::set_log_home("/home/al");
  CHECK(ajp::redact_log_text("cache : /home/alice/.cache/ajp") == "cache : /home/alice/.cache/ajp");
  CHECK(ajp::redact_log_text("cache : /home/al/.cache, et /home/al") == "cache : ~/.cache, et ~");
  CHECK(ajp::redact_log_text("vidéo : /run/media/al/Données SSD/a.mkv") ==
        "vidéo : /run/media/<user>/Données SSD/a.mkv");
  CHECK(ajp::redact_log_text("/media/al/x et /media/alice/y") == "/media/<user>/x et /media/alice/y");
  ajp::set_log_home("C:\\Users\\Jean");
#ifdef _WIN32
  CHECK(ajp::redact_log_text("c:\\users\\jean\\a.mkv") == "~\\a.mkv");
#else
  CHECK(ajp::redact_log_text("c:\\users\\jean\\a.mkv") == "c:\\users\\jean\\a.mkv");
#endif
  ajp::set_log_home("");
}

TEST_CASE("Journal nettoyé : un point final et les barres inverses doublées (B20-02)", "[log]") {
  ajp::set_log_home("/home/jean");
  CHECK(ajp::redact_log_text("dossier /home/jean.") == "dossier ~.");
  CHECK(ajp::redact_log_text("dossier /home/jean.x") == "dossier /home/jean.x");  // un autre dossier
  CHECK(ajp::redact_log_text("/home/jean/.cache") == "~/.cache");
  ajp::set_log_home("C:\\Users\\jean");
  CHECK(ajp::redact_log_text("C:\\\\Users\\\\jean\\\\x") == "~\\\\x");
  CHECK(ajp::redact_log_text("C:\\Users\\jean\\x") == "~\\x");
  ajp::set_log_home("");
}

TEST_CASE("Journal nettoyé : préfixe NT et chemins sans lecteur (C39-11)", "[log]") {
  CHECK(ajp::reduce_log_paths("path=\\??\\C:\\Users\\Jean\\a") == "path=…/a");
  CHECK(ajp::reduce_log_paths("chemin \\Users\\Jean\\Videos\\a.mkv") == "chemin …/a.mkv");
  CHECK(ajp::reduce_log_paths("chemin Users\\Jean\\Videos\\a.mkv") == "chemin …/a.mkv");
}

TEST_CASE("Journal nettoyé : les dossiers du système restent lisibles (C39-11)", "[log]") {
  CHECK(ajp::reduce_log_paths("gpu /dev/dri/renderD128") == "gpu /dev/dri/renderD128");
  CHECK(ajp::reduce_log_paths("lib /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1, ok") ==
        "lib /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1, ok");
  CHECK(ajp::reduce_log_paths("icd /usr/share/vulkan/icd.d/nvidia_icd.json") ==
        "icd /usr/share/vulkan/icd.d/nvidia_icd.json");
  CHECK(ajp::reduce_log_paths("dll C:\\Windows\\System32\\nvcuda.dll") ==
        "dll C:\\Windows\\System32\\nvcuda.dll");
}

TEST_CASE("Journal nettoyé : les chemins sont réduits au nom de leur fichier (B39-05)", "[log]") {
  CHECK(ajp::reduce_log_paths("moteur span D:\\AnimeJaNai\\engines\\a.engine") == "moteur span …/a.engine");
  CHECK(ajp::reduce_log_paths("journal : ~/.local/share/ajp/ajp.log · ajp 1.0") ==
        "journal : …/ajp.log · ajp 1.0");
  CHECK(ajp::reduce_log_paths("vidéo /run/media/x/Données SSD/a.mkv, 24 i/s") == "vidéo …/a.mkv, 24 i/s");
  CHECK(ajp::reduce_log_paths("1/2 et /s") == "1/2 et /s");  // pas des dossiers
  // Le puits : le dossier personnel, puis les chemins.
  std::vector<std::string> seen;
  ajp::set_log_home("/home/jean");
  const auto sink = ajp::redacting_sink(
      std::make_shared<spdlog::sinks::callback_sink_mt>([&](const spdlog::details::log_msg& msg) {
        seen.emplace_back(msg.payload.begin(), msg.payload.end());
      }));
  sink->log(spdlog::details::log_msg("app", spdlog::level::info, "cache /home/jean/.cache/ajp/x.bin"));
  ajp::set_log_home("");
  REQUIRE(seen.size() == 1);
  CHECK(seen.front() == "cache …/x.bin");
}

TEST_CASE("Journal : le fichier s'ouvre par son chemin et tourne en parts (B33-06)", "[log]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "ajp-log-file-test" / "données ユーザー";
  fs::remove_all(dir.parent_path());
  fs::create_directories(dir);
  const fs::path file = dir / "ajp.log";
  CHECK(ajp::log_part(file, 0) == file);
  CHECK(ajp::log_part(file, 2) == dir / "ajp.2.log");
  auto sink = ajp::log_file_sink(file, 64, 2);
  REQUIRE(sink != nullptr);
  sink->set_pattern("%v");
  const auto write = [&](const char* text) {
    sink->log(spdlog::details::log_msg("app", spdlog::level::info, text));
  };
  write("premiere ligne de quarante caracteres..");
  write("deuxieme ligne de quarante caracteres..");  // dépasse 64 : la première part passe en .1
  write("troisieme ligne de quarante caracteres.");
  write("quatrieme ligne de quarante caracteres.");
  sink->flush();
  const auto read = [](const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  };
  CHECK(read(file).contains("quatrieme"));
  CHECK(read(dir / "ajp.1.log").contains("troisieme"));
  CHECK(read(dir / "ajp.2.log").contains("deuxieme"));
  CHECK_FALSE(fs::exists(dir / "ajp.3.log"));  // la première, au-delà des deux parts, est partie
  CHECK(ajp::log_file_sink(dir / "absent" / "ajp.log", 64, 2) == nullptr);
  sink.reset();  // ajp.log refermé : Windows refuse de retirer un fichier ouvert
  fs::remove_all(dir.parent_path());
}

TEST_CASE("Journal : une exception sortie du corps est journalisée et vaut 1", "[log]") {
  Capture capture;
  ajp::install_log_sinks({capture.sink()}, spdlog::level::info);
  const int code = ajp::run_logged([]() -> int {
    ajp::log_channel("app")->info("avant");
    throw std::runtime_error("forcée");
  });
  CHECK(code == 1);
  const std::vector<std::string> lines = capture.lines();
  REQUIRE(lines.size() == 2);
  CHECK(lines[0] == "app : avant");
  CHECK(lines[1] == "app : arrêt sur une exception non rattrapée : forcée");
  CHECK(ajp::run_logged([] { return 7; }) == 7);
}
