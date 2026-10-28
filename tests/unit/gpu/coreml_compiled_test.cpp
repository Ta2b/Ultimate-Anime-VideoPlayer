// Le modèle compilé que Core ML direct reprend (gpu/coreml/coreml_compiled.h) : les noms qu'ONNX
// Runtime donne au programme ML, et le seul dossier .mlmodelc du cache, refusé quand le réseau est
// coupé en morceaux.
#include "gpu/coreml/coreml_compiled.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::coreml_rt {
namespace {

struct TempDir {
  explicit TempDir(const char* test)
      : path(std::filesystem::temp_directory_path() / std::format("ajp-coreml-{}", test)) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::filesystem::path path;
};

// Un modèle compilé tel que Core ML l'écrit : un dossier, avec des fichiers et des dossiers dedans.
void compiled_at(const std::filesystem::path& dir) {
  std::filesystem::create_directories(dir / "weights");
  std::ofstream(dir / "model.mil") << "program";
  std::ofstream(dir / "weights" / "weight.bin") << "w";
}

}  // namespace

TEST_CASE("Core ML : noms des entrées et sorties dans le programme ML", "[gpu][coreml]") {
  CHECK(ml_program_name("input") == "input");
  CHECK(ml_program_name("img_0") == "img_0");
  CHECK(ml_program_name("input.1") == "input_1");
  CHECK(ml_program_name("0") == "_0");
  CHECK(ml_program_name("12:out/x") == "_12_out_x");
  CHECK(ml_program_name("é") == "__");  // deux octets UTF-8, chacun remplacé
  CHECK(ml_program_name("").empty());
}

TEST_CASE("Core ML : le seul modèle compilé du cache", "[gpu][coreml]") {
  TempDir dir("compiled");
  const std::filesystem::path cache = dir.path / "net.coreml.mlcache";
  // Aucun : dossier absent, puis vide, puis un paquet ML sans modèle compilé.
  CHECK_FALSE(single_compiled(cache).has_value());
  std::filesystem::create_directories(cache / "ajp" / "0_dynamic_mlprogram" / "model.mlpackage");
  CHECK_FALSE(single_compiled(cache).has_value());
  // Un seul, à n'importe quelle profondeur ; un fichier au nom en .mlmodelc ne compte pas.
  const std::filesystem::path one = cache / "ajp" / "0_dynamic_mlprogram" / "compiled_model.mlmodelc";
  compiled_at(one);
  std::ofstream(cache / "ajp" / "notes.mlmodelc") << "pas un dossier";
  CHECK(single_compiled(cache) == std::optional(one));
  // Deux morceaux : rien.
  compiled_at(cache / "ajp" / "1_dynamic_mlprogram" / "compiled_model.mlmodelc");
  CHECK_FALSE(single_compiled(cache).has_value());
}

}  // namespace ajp::gpu::coreml_rt
