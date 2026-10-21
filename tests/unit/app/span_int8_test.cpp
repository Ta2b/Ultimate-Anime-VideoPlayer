// La variante INT8 d'un SPAN n'est prise que si son reçu (tools/span/quantize_span.py) est celui de
// ce modèle et de ce fichier, et passe les garde-fous du lecteur, pas seulement ceux du script.
#include "app/play/span_int8.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

struct Folder {
  Folder() : root(fs::temp_directory_path() / "ajp-span-int8-test") {
    fs::remove_all(root);
    fs::create_directories(root / "modeles");
    span = root / "2x_AnimeJaNai_HD_V3.1_Sharp1_Performance.onnx";
    write(span, "graphe fp16");
    sha = prepare::sha256_file(span).value();
    stem = "2x_AnimeJaNai_HD_V3.1_Sharp1_Performance-int8-" + sha.substr(0, 16);
    write(variant(), "graphe int8");
  }
  ~Folder() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  Folder(const Folder&) = delete;
  Folder& operator=(const Folder&) = delete;

  static void write(const fs::path& file, std::string_view text) {
    std::ofstream(file, std::ios::binary) << text;
  }
  fs::path dir() const { return root / "modeles"; }
  fs::path variant() const { return dir() / (stem + ".onnx"); }
  void receipt(double psnr_mean, double psnr_worst, double ssim, bool passed = true,
               std::string source_sha = {}, std::string model_sha = {}) const {
    if (source_sha.empty()) source_sha = sha;
    if (model_sha.empty()) model_sha = prepare::sha256_file(variant()).value();
    write(dir() / (stem + ".toml"),
          std::format("# reçu\nformat = \"ajp-span-int8-v1\"\nsource_sha256 = \"{}\"\nmodel_sha256 = \"{}\"\n"
                      "psnr_mean_db = {:.4f}\npsnr_worst_db = {:.4f}\nssim_worst = {:.4f}\npassed = {}\n",
                      source_sha, model_sha, psnr_mean, psnr_worst, ssim, passed ? "true" : "false"));
  }

  fs::path root;
  fs::path span;
  std::string sha;
  std::string stem;
};

TEST_CASE("SPAN INT8 : la variante d'un reçu qui passe les garde-fous", "[app][int8]") {
  const Folder f;
  CHECK_FALSE(find_span_int8(f.span, f.dir()).variant);  // sans reçu : rien, et rien à dire
  CHECK(find_span_int8(f.span, f.dir()).note.empty());

  f.receipt(45.2, 41.0, 0.9931);
  const SpanInt8Lookup found = find_span_int8(f.span, f.dir());
  REQUIRE(found.variant);
  const SpanInt8& variant = found.variant.value();
  CHECK(variant.onnx == f.variant());
  CHECK(variant.psnr_worst_db == 41.0);
  CHECK(found.note.find("PSNR moyen 45.2 dB") != std::string::npos);
}

TEST_CASE("SPAN INT8 : reçu refusé, le SPAN d'origine reste", "[app][int8]") {
  const Folder f;
  SECTION("sous les garde-fous du lecteur, même marqué passé") {
    f.receipt(41.9, 40.0, 0.99);
    CHECK(find_span_int8(f.span, f.dir()).note.find("sous les garde-fous") != std::string::npos);
    f.receipt(43.0, 37.9, 0.99);
    CHECK_FALSE(find_span_int8(f.span, f.dir()).variant);
    f.receipt(43.0, 40.0, 0.984);
    CHECK_FALSE(find_span_int8(f.span, f.dir()).variant);
  }
  SECTION("échec du script") {
    f.receipt(45.0, 41.0, 0.99, false);
    CHECK_FALSE(find_span_int8(f.span, f.dir()).variant);
  }
  SECTION("reçu d'un autre modèle") {
    f.receipt(45.0, 41.0, 0.99, true, std::string(64, 'a'));
    CHECK(find_span_int8(f.span, f.dir()).note.find("autre modèle") != std::string::npos);
  }
  SECTION("variante modifiée depuis son reçu") {
    f.receipt(45.0, 41.0, 0.99);
    Folder::write(f.variant(), "autre graphe");
    CHECK(find_span_int8(f.span, f.dir()).note.find("modifié depuis son reçu") != std::string::npos);
  }
  SECTION("variante absente") {
    f.receipt(45.0, 41.0, 0.99, true, {}, std::string(64, 'b'));
    fs::remove(f.variant());
    CHECK_FALSE(find_span_int8(f.span, f.dir()).variant);
  }
  SECTION("reçu illisible") {
    Folder::write(f.dir() / (f.stem + ".toml"), "format = ");
    CHECK(find_span_int8(f.span, f.dir()).note.find("illisible") != std::string::npos);
  }
  CHECK_FALSE(find_span_int8(f.span, f.dir()).variant);
}

TEST_CASE("SPAN INT8 : le nom d'origine se lit dans celui de la variante", "[app][int8]") {
  CHECK(span_int8_source("2x_AnimeJaNai_HD_V3.1_Sharp1_Performance-int8-0123456789abcdef.onnx") ==
        "2x_AnimeJaNai_HD_V3.1_Sharp1_Performance.onnx");
  CHECK_FALSE(span_int8_source("2x_AnimeJaNai_HD_V3.1_Sharp1_Performance.onnx"));
  CHECK_FALSE(span_int8_source("modele-int8-0123456789ABCDEF.onnx"));  // le script écrit en minuscules
  CHECK_FALSE(span_int8_source("modele-int8-0123456789abcde.onnx"));
  CHECK_FALSE(span_int8_source("-int8-0123456789abcdef.onnx"));
}

}  // namespace
}  // namespace ajp::app::play
