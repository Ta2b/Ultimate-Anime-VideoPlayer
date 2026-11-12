// « Signaler un problème » : nettoyage des chemins, composition du rapport, marque de séance (A39-01 à
// A39-03, A38-01).
#include "app/play/problem_report.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"

using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

fs::path fresh_dir(const char* name) {
  const fs::path dir = fs::temp_directory_path() / name;
  std::error_code error;
  fs::remove_all(dir, error);
  fs::create_directories(dir, error);
  return dir;
}

void write(const fs::path& file, const std::string& text) {
  std::ofstream(file, std::ios::binary) << text;
}

}  // namespace

TEST_CASE("Rapport : un chemin hors du dossier personnel ne garde que le nom du fichier", "[app][report]") {
  const std::string text =
      "lecture de /run/media/ta2b/Données SSD/Anime/Série/ep01.mkv\n"
      "épisode suivant : D:\\Torrents\\[Groupe] Série\\ep02.mkv\n"
      "refus (/mnt/nas/anime/ep03.mkv) code 3\n"
      "lecture de ~/Vidéos/Série/ep04.mkv\n"
      "partage \\\\serveur\\anime\\ep05.mkv\n";
  const std::string out = sanitize_report_paths(text, "/home/ta2b");
  CHECK(out ==
        "lecture de …/ep01.mkv\n"
        "épisode suivant : …/ep02.mkv\n"
        "refus (…/ep03.mkv) code 3\n"
        "lecture de …/ep04.mkv\n"
        "partage …/ep05.mkv\n");
}

TEST_CASE("Rapport : ce qui n'est pas un dossier reste lisible", "[app][report]") {
  const std::string text = "débit 10 /s, ratio 16/9, https://github.com/Ta2b/x, SPAN/RIFE, a / b\n";
  CHECK(sanitize_report_paths(text, "") == text);
}

TEST_CASE("Rapport : aucun nom d'utilisateur ne survit au nettoyage", "[app][report]") {
  const std::string out = sanitize_report_paths(
      "ouverture C:\\Users\\Jean Dupont\\Videos\\a.mkv puis /home/ta2b/.local/share/ajp/b.onnx",
      "/home/ta2b");
  CHECK(out.find("Dupont") == std::string::npos);
  CHECK(out.find("ta2b") == std::string::npos);
  CHECK(out.find("a.mkv") != std::string::npos);
}

TEST_CASE("Rapport : accolades, flèches, file:// et dossier personnel nu ne laissent pas le nom",
          "[app][report]") {
  CHECK(sanitize_report_paths("x {/home/tatoub/Videos/a.mkv}", "").find("tatoub") == std::string::npos);
  CHECK(sanitize_report_paths("→/home/tatoub/x/y.mkv", "").find("tatoub") == std::string::npos);
  CHECK(sanitize_report_paths("dossier : /home/tatoub", "").find("tatoub") == std::string::npos);
  CHECK(sanitize_report_paths("file:///home/tatoub/Videos/a.mkv", "").find("tatoub") == std::string::npos);
  CHECK(sanitize_report_paths("a|/home/tatoub/b.mkv", "").find("tatoub") == std::string::npos);
  CHECK(sanitize_report_paths("lecture de /home/tatoub/Videos/a.mkv", "").find("a.mkv") != std::string::npos);
}

TEST_CASE("Rapport : les faits sont gardés dans l'ordre, un même nom remplace sa valeur", "[app][report]") {
  clear_report_facts();
  note_report_fact("Carte", "RTX 4070");
  note_report_fact("Pilote", "580.82");
  note_report_fact("Carte", "RTX 4050");
  const auto facts = report_facts();
  REQUIRE(facts.size() == 2);
  CHECK(facts[0] == std::pair<std::string, std::string>{"Carte", "RTX 4050"});
  CHECK(facts[1].first == "Pilote");
  clear_report_facts();
}

TEST_CASE("Rapport : en-tête de faits, séance d'avant, minidump et bornes", "[app][report]") {
  ReportParts parts;
  parts.facts = {{"ajp", "0.10.3"}, {"Système", "Arch Linux"}, {"Préréglage", "Auto"}};
  parts.previous_log = "démarrage\nPLANTAGE segfault\n";
  parts.current_log = "nouvelle séance\n";
  parts.dump_name = "ajp-20261109-120000-42.dmp";
  parts.home = "/home/u";
  const std::string report = compose_report(parts, 10000);
  CHECK(report.starts_with("ajp : 0.10.3\nSystème : Arch Linux\nPréréglage : Auto\n"));
  CHECK(report.find("ajp-20261109-120000-42.dmp") != std::string::npos);
  const auto before = report.find("séance précédente");
  const auto crash = report.find("PLANTAGE segfault");
  const auto now = report.find("nouvelle séance");
  REQUIRE(before != std::string::npos);
  CHECK(before < crash);
  CHECK(crash < report.find("--- séance en cours"));
  CHECK(report.find("--- séance en cours") < now);

  // Sans plantage : pas d'en-tête de séance d'avant.
  parts.previous_log.clear();
  CHECK(compose_report(parts, 10000).find("séance précédente") == std::string::npos);

  // Borné : le début des journaux saute, les faits restent.
  parts.current_log.assign(50000, 'x');
  parts.current_log.insert(0, "premiere ligne\n");
  parts.current_log += "\n";
  const std::string small = compose_report(parts, 2000);
  CHECK(small.size() <= 2000);
  CHECK(small.starts_with("ajp : 0.10.3"));
  CHECK(small.find("premiere ligne") == std::string::npos);
  CHECK(small.find("début du journal omis") != std::string::npos);
}

TEST_CASE("Rapport : la fin d'un journal commence à une ligne entière", "[app][report]") {
  const fs::path dir = fresh_dir("ajp-report-tail");
  write(dir / "ajp.log", "aaaa\nbbbb\ncccc\n");
  CHECK(read_tail(dir / "ajp.log", 100) == "aaaa\nbbbb\ncccc\n");
  const std::string cut = read_tail(dir / "ajp.log", 8);
  CHECK(cut.ends_with("\ncccc\n"));
  CHECK(cut.find("début du journal omis") != std::string::npos);  // le rapport le dit (A23-14)
  CHECK(read_tail(dir / "ajp.log", 100).find("omis") == std::string::npos);
  CHECK(read_tail(dir / "absent.log", 8).empty());
}

TEST_CASE("Rapport : séance ouverte, retrouvée au lancement suivant seulement si elle ne s'est pas fermée",
          "[app][report]") {
  const fs::path dir = fresh_dir("ajp-report-session");
  CHECK_FALSE(begin_session(dir));
  CHECK_FALSE(previous_session_crashed());
  CHECK(fs::exists(dir / "session.open"));
  end_session();
  CHECK_FALSE(fs::exists(dir / "session.open"));

  // Une séance jamais refermée (plantage) : le lancement suivant le voit.
  CHECK_FALSE(begin_session(dir));
  CHECK(begin_session(dir));
  CHECK(previous_session_crashed());
  end_session();
}

TEST_CASE("Rapport : le rapport complet joint la séance d'avant quand elle a planté", "[app][report]") {
  const fs::path dir = fresh_dir("ajp-report-full");
  write(dir / "ajp.log", "séance neuve\n");
  write(dir / "ajp.log.1", "lecture de /run/media/u/d/ep.mkv\nsegfault dans nvinfer\n");
  fs::create_directories(dir / "crashes");
  write(dir / "crashes" / "ajp-1.dmp", "x");
  set_report_log_file(dir / "ajp.log");
  clear_report_facts();
  note_report_fact("ajp", "0.10.3");

  end_session();
  begin_session(dir);  // pas de marque d'avant : propre
  CHECK(current_report("").find("segfault") == std::string::npos);
  CHECK(current_report("").find("ajp-1.dmp") ==
        std::string::npos);  // séance propre : pas de minidump nommé (C39-07)

  CHECK(begin_session(dir));  // la marque est restée : plantage
  const std::string report = current_report("");
  CHECK(report.find("ajp-1.dmp") != std::string::npos);
  CHECK(report.find("segfault dans nvinfer") != std::string::npos);
  CHECK(report.find("/run/media") == std::string::npos);
  CHECK(report.find("…/ep.mkv") != std::string::npos);
  end_session();
  clear_report_facts();
}

TEST_CASE("Rapport : la fin du journal de construction part avec le refus (B39-03)", "[app][report]") {
  const fs::path dir = fresh_dir("ajp-report-build");
  std::string log = "commande trtexec --onnx=modele.onnx\n";
  for (int i = 0; i < 100; ++i) log += "[I] tactique " + std::to_string(i) + "\n";
  log += "[E] Error Code 2: OutOfMemory (Requested size was 9000000000 bytes.)\n";
  write(dir / "span.engine.build.log", log);
  clear_report_facts();

  note_build_failure("BuilderFailed", "trtexec a échoué ; dernière erreur : OutOfMemory ; journal : " +
                                          (dir / "span.engine.build.log").string());
  const auto facts = report_facts();
  REQUIRE(facts.size() == 1);
  CHECK(facts[0].first == "Construction");
  const std::string& text = facts[0].second;
  CHECK(text.starts_with("BuilderFailed : trtexec a échoué ; dernière erreur : OutOfMemory\n"));
  CHECK(text.contains("Error Code 2: OutOfMemory"));
  CHECK(text.contains("tactique 99"));
  CHECK_FALSE(text.contains("tactique 10\n"));  // les dernières lignes seulement
  CHECK_FALSE(text.contains("commande trtexec"));
  CHECK_FALSE(text.contains("journal :"));

  // Sans journal nommé ou lisible : la cause reste.
  note_build_failure("BuilderTimeout", "trtexec a dépassé son délai de 600.0 s");
  CHECK(report_facts()[0].second == "BuilderTimeout : trtexec a dépassé son délai de 600.0 s");
  note_build_failure("BuilderFailed", "échec ; journal : " + (dir / "absent.build.log").string());
  CHECK(report_facts()[0].second == "BuilderFailed : échec");
  clear_report_facts();
  fs::remove_all(dir);
}

TEST_CASE("Rapport : après une relance voulue, la séance qui a perdu le GPU est jointe (B39-01)",
          "[app][report]") {
  const fs::path dir = fresh_dir("ajp-report-relaunched");
  write(dir / "ajp.log", "séance neuve\n");
  write(dir / "ajp.log.1", "appareil Vulkan de l'écran perdu (TDR, pilote rechargé)\n");
  set_report_log_file(dir / "ajp.log");
  clear_report_facts();
  begin_session(dir);
  end_session();  // la relance voulue retire la marque (main.cpp)
  CHECK_FALSE(begin_session(dir));
  CHECK(current_report("").find("perdu") == std::string::npos);  // avant : rien de la cause
  note_previous_session_relaunched("gpu-lost");
  const std::string report = current_report("");
  CHECK(report.find("appareil Vulkan de l'écran perdu") != std::string::npos);
  CHECK(report.find("séance précédente (relancée : gpu-lost)") != std::string::npos);
  note_previous_session_relaunched("");
  end_session();
}

#ifdef __linux__
// B39-04 : le build-id de l'exécutable accompagne le rapport, pour résoudre une pile brute avec les symboles
// de la release (le binaire livré est dépouillé). CMakeLists.txt demande --build-id à l'éditeur de liens.
TEST_CASE("Rapport : le build-id de l'exécutable se lit, en hexadécimal", "[app][report]") {
  const std::string id = build_id_description();
  REQUIRE(id.size() >= 16);
  CHECK(std::ranges::all_of(id, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }));
}
#endif

// C39-06 : un refus d'un moteur précédent ne reste pas dans le rapport quand le moteur suivant n'en a aucun.
TEST_CASE("le fait « Moteurs refusés » disparaît quand le moteur suivant n'a rien refusé",
          "[problem_report]") {
  clear_report_facts();
  note_engine_facts("a", "x : refusé");
  const auto has_refused = [] {
    const auto facts = report_facts();
    return std::ranges::any_of(facts, [](const auto& fact) { return fact.first == "Moteurs refusés"; });
  };
  CHECK(has_refused());
  note_engine_facts("b", "");
  CHECK_FALSE(has_refused());
  clear_report_facts();
}

// C39-07 : un vieux minidump sans lien avec la séance d'avant n'est pas nommé.
TEST_CASE("le rapport nomme le minidump seulement après une séance plantée", "[problem_report]") {
  const fs::path dir = fresh_dir("ajp-report-dump-test");
  write(dir / "ajp.log", "ligne\n");
  fs::create_directories(dir / "crashes");
  write(dir / "crashes" / "ajp-20240101-000000-1.dmp", "x");
  set_report_log_file(dir / "ajp.log");
  begin_session(dir);
  end_session();
  CHECK_FALSE(begin_session(dir));  // la séance d'avant s'est fermée proprement
  CHECK(current_report("").find("Minidump") == std::string::npos);
  CHECK(begin_session(dir));  // la marque est restée : plantage
  CHECK(current_report("").find("ajp-20240101-000000-1.dmp") != std::string::npos);
  end_session();
}

// C39-01 : les options à l'essai allumées se lisent dans le rapport ; tout au défaut ne laisse rien.
TEST_CASE("les réglages changés sont listés, les défauts non", "[problem_report]") {
  CHECK(settings_changes_text(PlaySettings{}) == "(aucun)");
  PlaySettings s;
  s.cubic_warp = true;
  const std::string text = settings_changes_text(s);
  CHECK(text.find("cubic_warp") != std::string::npos);
  CHECK(text.find("true") != std::string::npos);
}
