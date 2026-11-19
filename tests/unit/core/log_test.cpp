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
  kirari::install_log_sinks({first.sink()}, spdlog::level::info);
  const std::shared_ptr<spdlog::logger> media = kirari::log_channel("media");
  media->info("avant");

  kirari::install_log_sinks({later.sink()}, spdlog::level::info);
  media->info("après");
  kirari::log_channel("audio")->warn("nouveau canal");

  CHECK(first.lines() == std::vector<std::string>{"media : avant"});
  CHECK(later.lines() == std::vector<std::string>{"media : après", "audio : nouveau canal"});
}

TEST_CASE("Journal : le niveau installé filtre les canaux créés ensuite", "[log]") {
  Capture capture;
  kirari::install_log_sinks({capture.sink()}, spdlog::level::warn);
  const std::shared_ptr<spdlog::logger> subs = kirari::log_channel("subs");
  subs->info("ignoré");
  subs->error("gardé");
  CHECK(capture.lines() == std::vector<std::string>{"subs : gardé"});
  CHECK(subs->level() == spdlog::level::warn);
}

TEST_CASE("Journal : un canal n'entre pas au registre de spdlog", "[log]") {
  kirari::install_log_sinks({std::make_shared<spdlog::sinks::null_sink_mt>()}, spdlog::level::info);
  const std::shared_ptr<spdlog::logger> channel = kirari::log_channel("prepare");
  CHECK(channel->name() == "prepare");
  CHECK(spdlog::get("prepare") == nullptr);
  CHECK(channel == kirari::log_channel("prepare"));  // 1.11 : un canal par module, gardé
}

TEST_CASE("Journal : un seul canal par module, refait quand les puits changent", "[core][log]") {
  Capture capture;
  kirari::install_log_sinks({capture.sink()}, spdlog::level::info);
  const auto first = kirari::log_channel("unique");
  CHECK(kirari::log_channel("unique") == first);  // le même logger, pas un nouveau à chaque appel
  CHECK(kirari::log_channel("autre") != first);
  // Un niveau réinstallé : le canal se refait et suit le niveau.
  kirari::install_log_sinks({capture.sink()}, spdlog::level::warn);
  const auto again = kirari::log_channel("unique");
  CHECK(again != first);
  CHECK(again->level() == spdlog::level::warn);
  CHECK(kirari::log_channel("unique") == again);
  kirari::install_log_sinks({std::make_shared<spdlog::sinks::null_sink_mt>()}, spdlog::level::info);
}

TEST_CASE("Journal nettoyé : le dossier personnel devient ~ sous ses deux séparateurs", "[log]") {
  kirari::set_log_home("C:\\Users\\tatoub\\");
  CHECK(kirari::redact_log_text("ouvert C:\\Users\\tatoub\\Videos\\a.mkv et C:/Users/tatoub/x") ==
        "ouvert ~\\Videos\\a.mkv et ~/x");
  kirari::set_log_home("/");
  CHECK(kirari::redact_log_text("/usr/lib") == "/usr/lib");
  kirari::set_log_home("");
}

TEST_CASE("Journal nettoyé : frontière de composant, disques montés, casse sous Windows (A20-03)", "[log]") {
  kirari::set_log_home("/home/al");
  CHECK(kirari::redact_log_text("cache : /home/alice/.cache/kirari") == "cache : /home/alice/.cache/kirari");
  CHECK(kirari::redact_log_text("cache : /home/al/.cache, et /home/al") == "cache : ~/.cache, et ~");
  CHECK(kirari::redact_log_text("vidéo : /run/media/al/Données SSD/a.mkv") ==
        "vidéo : /run/media/<user>/Données SSD/a.mkv");
  CHECK(kirari::redact_log_text("/media/al/x et /media/alice/y") == "/media/<user>/x et /media/alice/y");
  kirari::set_log_home("C:\\Users\\Jean");
#ifdef _WIN32
  CHECK(kirari::redact_log_text("c:\\users\\jean\\a.mkv") == "~\\a.mkv");
#else
  CHECK(kirari::redact_log_text("c:\\users\\jean\\a.mkv") == "c:\\users\\jean\\a.mkv");
#endif
  kirari::set_log_home("");
}

TEST_CASE("Journal nettoyé : un point final et les barres inverses doublées (B20-02)", "[log]") {
  kirari::set_log_home("/home/jean");
  CHECK(kirari::redact_log_text("dossier /home/jean.") == "dossier ~.");
  CHECK(kirari::redact_log_text("dossier /home/jean.x") == "dossier /home/jean.x");  // un autre dossier
  CHECK(kirari::redact_log_text("/home/jean/.cache") == "~/.cache");
  kirari::set_log_home("C:\\Users\\jean");
  CHECK(kirari::redact_log_text("C:\\\\Users\\\\jean\\\\x") == "~\\\\x");
  CHECK(kirari::redact_log_text("C:\\Users\\jean\\x") == "~\\x");
  kirari::set_log_home("");
}

TEST_CASE("Journal nettoyé : préfixe NT et chemins sans lecteur (C39-11)", "[log]") {
  CHECK(kirari::reduce_log_paths("path=\\??\\C:\\Users\\Jean\\a") == "path=…/a");
  CHECK(kirari::reduce_log_paths("chemin \\Users\\Jean\\Videos\\a.mkv") == "chemin …/a.mkv");
  CHECK(kirari::reduce_log_paths("chemin Users\\Jean\\Videos\\a.mkv") == "chemin …/a.mkv");
}

TEST_CASE("Journal nettoyé : les dossiers du système restent lisibles (C39-11)", "[log]") {
  CHECK(kirari::reduce_log_paths("gpu /dev/dri/renderD128") == "gpu /dev/dri/renderD128");
  CHECK(kirari::reduce_log_paths("lib /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1, ok") ==
        "lib /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1, ok");
  CHECK(kirari::reduce_log_paths("icd /usr/share/vulkan/icd.d/nvidia_icd.json") ==
        "icd /usr/share/vulkan/icd.d/nvidia_icd.json");
  CHECK(kirari::reduce_log_paths("dll C:\\Windows\\System32\\nvcuda.dll") ==
        "dll C:\\Windows\\System32\\nvcuda.dll");
}

TEST_CASE("Journal nettoyé : les chemins sont réduits au nom de leur fichier (B39-05)", "[log]") {
  CHECK(kirari::reduce_log_paths("moteur span D:\\AnimeJaNai\\engines\\a.engine") ==
        "moteur span …/a.engine");
  CHECK(kirari::reduce_log_paths("journal : ~/.local/share/kirari/kirari.log · kirari 1.0") ==
        "journal : …/kirari.log · kirari 1.0");
  CHECK(kirari::reduce_log_paths("vidéo /run/media/x/Données SSD/a.mkv, 24 i/s") == "vidéo …/a.mkv, 24 i/s");
  CHECK(kirari::reduce_log_paths("1/2 et /s") == "1/2 et /s");  // pas des dossiers
  // C20-01 : une fraction ou une parenthèse après le nom ne le mange plus.
  CHECK(kirari::reduce_log_paths("ouvre /home/jean/Videos/ep01.mkv (24000/1001 fps, 1920x1080)") ==
        "ouvre …/ep01.mkv (24000/1001 fps, 1920x1080)");
  CHECK(kirari::reduce_log_paths("modèle /srv/kirari/models/span.onnx chargé en 0.8 s (cache 3/4)") ==
        "modèle …/span.onnx chargé en 0.8 s (cache 3/4)");
  CHECK(kirari::reduce_log_paths("échec '/home/jean/Videos/ep.mkv' (lu 1/2)") == "échec '…/ep.mkv' (lu 1/2)");
  CHECK(kirari::reduce_log_paths("vidéo /home/jean/Anime (2020)/ep.mkv") == "vidéo …/ep.mkv");
  // Le puits : le dossier personnel, puis les chemins.
  std::vector<std::string> seen;
  kirari::set_log_home("/home/jean");
  const auto sink = kirari::redacting_sink(
      std::make_shared<spdlog::sinks::callback_sink_mt>([&](const spdlog::details::log_msg& msg) {
        seen.emplace_back(msg.payload.begin(), msg.payload.end());
      }));
  sink->log(spdlog::details::log_msg("app", spdlog::level::info, "cache /home/jean/.cache/kirari/x.bin"));
  kirari::set_log_home("");
  REQUIRE(seen.size() == 1);
  CHECK(seen.front() == "cache …/x.bin");
  // C35-05 : un nom de fichier qui porte un saut de ligne ou une séquence ESC n'écrit pas de fausse ligne.
  sink->log(
      spdlog::details::log_msg("app", spdlog::level::info, "ouvre Ep 01\n[app] [info] faux\x1b[31m.mkv"));
  REQUIRE(seen.size() == 2);
  CHECK(seen.back() == "ouvre Ep 01\n    [app] [info] faux\\x1B[31m.mkv");
}

TEST_CASE("Journal : le fichier s'ouvre par son chemin et tourne en parts (B33-06)", "[log]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "kirari-log-file-test" / "données ユーザー";
  fs::remove_all(dir.parent_path());
  fs::create_directories(dir);
  const fs::path file = dir / "kirari.log";
  CHECK(kirari::log_part(file, 0) == file);
  CHECK(kirari::log_part(file, 2) == dir / "kirari.2.log");
  auto sink = kirari::log_file_sink(file, 64, 2);
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
  CHECK(read(dir / "kirari.1.log").contains("troisieme"));
  CHECK(read(dir / "kirari.2.log").contains("deuxieme"));
  CHECK_FALSE(fs::exists(dir / "kirari.3.log"));  // la première, au-delà des deux parts, est partie
  CHECK(kirari::log_file_sink(dir / "absent" / "kirari.log", 64, 2) == nullptr);
  sink.reset();  // kirari.log refermé : Windows refuse de retirer un fichier ouvert
  fs::remove_all(dir.parent_path());
}

TEST_CASE("Journal : les séances d'avant tournent avec leurs parts (C18-10)", "[log]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "kirari-log-rotate-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const auto put = [&](const char* name, const char* text) { std::ofstream(dir / name) << text; };
  const auto read = [&](const char* name) {
    std::ifstream in(dir / name, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  };
  put("kirari.log", "séance 3");
  put("kirari.1.log", "séance 3, part 1");
  put("kirari.log.1", "séance 2");
  put("kirari.1.log.1",
      "séance 1, part 1");  // plus ancienne que kirari.log.1 : la séance 2 n'avait qu'une part
  put("kirari.log.2", "séance 1");
  put("kirari.log.3", "séance 0");
  kirari::rotate_session_logs(dir / "kirari.log", 3, 2);
  CHECK_FALSE(fs::exists(dir / "kirari.log"));
  CHECK(read("kirari.log.1") == "séance 3");
  CHECK(read("kirari.1.log.1") == "séance 3, part 1");
  CHECK(read("kirari.log.2") == "séance 2");
  CHECK(read("kirari.1.log.2") == "séance 1, part 1");
  CHECK(read("kirari.log.3") == "séance 1");  // la séance 0, au-delà des trois gardées, est partie
  CHECK_FALSE(fs::exists(dir / "kirari.log.4"));
  fs::remove_all(dir);
}

TEST_CASE("Journal : une exception sortie du corps est journalisée et vaut 1", "[log]") {
  Capture capture;
  kirari::install_log_sinks({capture.sink()}, spdlog::level::info);
  const int code = kirari::run_logged([]() -> int {
    kirari::log_channel("app")->info("avant");
    throw std::runtime_error("forcée");
  });
  CHECK(code == 1);
  const std::vector<std::string> lines = capture.lines();
  REQUIRE(lines.size() == 2);
  CHECK(lines[0] == "app : avant");
  CHECK(lines[1] == "app : arrêt sur une exception non rattrapée : forcée");
  CHECK(kirari::run_logged([] { return 7; }) == 7);
}
