// Exécutable d'aide des tests de prepare.
//
// Copié sous le nom « trtexec-<mode> », il joue un faux trtexec : il imprime
// une progression semblable à celle de trtexec 11, puis réussit, échoue,
// plante, se bloque ou modifie le modèle qu'il lit. Lancé directement, son
// premier argument choisit un comportement pour les tests du lanceur et du cache.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "prepare/cache.h"
#include "prepare/cache_io.h"
#include "prepare/key_lock.h"

namespace {

namespace fs = std::filesystem;
using std::chrono::milliseconds;

void say(std::string_view line) {
  std::fwrite(line.data(), 1, line.size(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

void append_line(const fs::path& file, const std::string& line) {
  std::ofstream out(file, std::ios::app);
  out << line << '\n';
}

std::string value_of(const std::vector<std::string>& args, std::string_view flag) {
  for (const std::string& arg : args) {
    if (arg.starts_with(flag)) return arg.substr(flag.size());
  }
  return {};
}

int fake_trtexec(std::string_view mode, const std::vector<std::string>& args) {
  say("&&&& RUNNING TensorRT.trtexec [TensorRT v110300] [b99] # trtexec");
  for (const std::string& arg : args) say("[I] arg " + arg);
  const fs::path engine = value_of(args, "--saveEngine=");
  const fs::path timing_cache = value_of(args, "--timingCacheFile=");
  append_line(value_of(args, "--onnx=") + ".calls", "appel");
  say("[10/18/2026-15:29:26] [I] Start parsing network model.");
  if (mode == "fail") {
    say("[10/18/2026-15:29:26] [E] Error[4]: fausse erreur de construction");
    say("&&&& FAILED TensorRT.trtexec [TensorRT v110300] [b99]");
    return 1;
  }
  say("[10/18/2026-15:29:26] [I] Finished parsing network model. Parse time: 0.01");
  if (mode == "crash") {
    append_line(timing_cache, "à moitié");  // comme trtexec tué pendant qu'il écrit son cache
    std::abort();
  }
  if (mode == "hang") std::this_thread::sleep_for(std::chrono::seconds(60));
  if (mode == "slow") std::this_thread::sleep_for(milliseconds(300));
  if (mode == "mutate") append_line(value_of(args, "--onnx="), "octets ajoutés pendant la construction");
  say("[I] [TRT] Total Activation Memory: 1024 bytes");
  say("[I] [TRT] Engine generation completed in 0.01 seconds.");
  {
    std::ofstream out(engine, std::ios::binary);
    if (mode != "empty") out << "moteur factice pour " << value_of(args, "--onnx=");
  }
  append_line(timing_cache, "passe");  // le cache reçu, plus les temps de cette construction
  if (const fs::path layers = value_of(args, "--exportLayerInfo="); !layers.empty() && mode != "no-layers") {
    std::ofstream(layers) << R"j({"Layers": [{"Name": "input", "LayerType": "Reformat"},
 {"Name": "/conv1/Conv + PWN(/act/PRelu)", "LayerType":"CaskConvolution"},
 {"Name": "output", "LayerType" : "Reformat"}], "Bindings": ["input", "output"]})j";
  }
  say("[I] Engine built in 0.02 sec.");
  say("&&&& PASSED TensorRT.trtexec [TensorRT v110300] [b99]");
  return 0;
}

// Écrivain concurrent : une seule production doit avoir lieu, tous processus confondus.
int cache_writer(const std::vector<std::string>& args) {
  const fs::path destination = args.at(0);
  const fs::path calls = args.at(1);
  const auto delay = milliseconds(std::stoi(args.at(2)));
  auto produced = ajp::prepare::produce_once({.destination = destination}, [&](const fs::path& output) {
    append_line(calls, "production");
    std::this_thread::sleep_for(delay);
    std::ofstream(output, std::ios::binary) << "contenu unique";
    return std::expected<void, ajp::Refusal>{};
  });
  if (!produced) {
    say("refus " + produced.error().detail);
    return 2;
  }
  say(*produced == ajp::prepare::Production::Produced ? "produit" : "repris");
  return 0;
}

// Constructeur qui meurt en pleine production : son dossier de travail reste.
int cache_orphan(const std::vector<std::string>& args) {
  const ajp::prepare::CacheEntry entry{.destination = args.at(0)};
  auto lock = ajp::prepare::KeyLock::acquire(entry.lock_file());
  auto stage =
      ajp::prepare::reserve_stage(entry.destination.parent_path(), ajp::prepare::kStagePrefix, entry.owner());
  if (!lock || !stage) return 2;
  std::ofstream(*stage / "model.engine", std::ios::binary) << "moitié";
  say("orphelin");
  std::_Exit(0);
}

int hold_lock(const std::vector<std::string>& args) {
  auto lock = ajp::prepare::KeyLock::acquire(args.at(0));
  if (!lock) return 2;
  say("verrou pris");
  std::this_thread::sleep_for(milliseconds(std::stoi(args.at(1))));
  return 0;
}

int direct(std::string_view mode, const std::vector<std::string>& args) {
  if (mode == "lines") {
    for (int i = 1; i <= std::stoi(args.at(0)); ++i) say("ligne " + std::to_string(i));
    return 0;
  }
  if (mode == "exit") {
    say("sortie " + args.at(0));
    return std::stoi(args.at(0));
  }
  if (mode == "crash") {
    say("avant plantage");
    std::abort();
  }
  if (mode == "hang") {
    say("en attente");
    std::this_thread::sleep_for(std::chrono::seconds(60));
    return 0;
  }
  if (mode == "streams") {
    std::fputs("sur stderr\n", stderr);
    std::fflush(stderr);
    say("sur stdout");
    return 0;
  }
  if (mode == "progress") {
    std::fputs("10%\r20%\r30%\r\nfin sans retour", stdout);
    return 0;
  }
  if (mode == "args") {
    for (const std::string& arg : args) say("[" + arg + "]");
    return 0;
  }
  if (mode == "cache-writer") return cache_writer(args);
  if (mode == "cache-orphan") return cache_orphan(args);
  if (mode == "hold-lock") return hold_lock(args);
  return 64;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> all(argv + 1, argv + argc);
  const std::string name = fs::path(argv[0]).stem().string();
  if (name.starts_with("trtexec-")) return fake_trtexec(std::string_view(name).substr(8), all);
  if (all.empty()) return 64;
  return direct(all.front(), std::vector<std::string>(all.begin() + 1, all.end()));
}
