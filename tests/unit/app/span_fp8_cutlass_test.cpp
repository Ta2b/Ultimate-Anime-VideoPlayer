// La variante « -fp8c- » d'un SPAN (paquet « .span8 » des noyaux CUTLASS, tools/span/derive_span_fp8.py) :
// prise seulement avec un reçu de ce modèle et de ce fichier aux seuils de l'INT8, pas à ceux du FP8 par Q/DQ
// (sans GPU).
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>

#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>

#include "app/play/span_int8.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;

struct Folder {
  Folder()
      : root(fs::temp_directory_path() /
             std::format("ajp-span-fp8c-test-{:x}",
                         std::hash<std::string>{}(Catch::getResultCapture().getCurrentTestName()))) {
    fs::remove_all(root);
    fs::create_directories(root / "modeles");
    span = root / "2x_AnimeJaNai_HD_V3.1_Balanced.onnx";
    write(span, "graphe fp16");
    sha = prepare::sha256_file(span).value();
    stem = "2x_AnimeJaNai_HD_V3.1_Balanced-fp8c-" + sha.substr(0, 16);
    write(package(), "paquet de poids");
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
  fs::path package() const { return dir() / (stem + ".span8"); }
  void receipt(double psnr_mean, double psnr_worst, double ssim, std::string_view format = "ajp-span-fp8c-v1",
               bool passed = true) const {
    write(
        dir() / (stem + ".toml"),
        std::format("format = \"{}\"\nsource_sha256 = \"{}\"\nmodel_sha256 = \"{}\"\npsnr_mean_db = {:.4f}\n"
                    "psnr_worst_db = {:.4f}\nssim_worst = {:.4f}\npassed = {}\n",
                    format, sha, prepare::sha256_file(package()).value(), psnr_mean, psnr_worst, ssim,
                    passed ? "true" : "false"));
  }

  fs::path root;
  fs::path span;
  std::string sha;
  std::string stem;
};

}  // namespace

TEST_CASE("SPAN FP8 CUTLASS : le paquet d'un reçu qui passe les seuils de l'INT8", "[app][int8][fp8c]") {
  const Folder f;
  CHECK_FALSE(find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir()).variant);  // sans reçu : rien
  f.receipt(52.0, 49.3, 0.9967);
  const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir());
  REQUIRE(found.variant);
  CHECK(found.variant->kind == SpanVariantKind::Fp8Cutlass);
  CHECK(found.variant->onnx.extension() ==
        ".span8");  // le chemin du paquet, que le backend construit sans trtexec
  CHECK(found.variant->onnx == f.package());
  CHECK(found.note.find("PSNR moyen 52.0 dB") != std::string::npos);
  CHECK(variant_text(SpanVariantKind::Fp8Cutlass) == "FP8 CUTLASS");
}

TEST_CASE("SPAN FP8 CUTLASS : un score que le FP8 par Q/DQ accepterait ne suffit pas", "[app][int8][fp8c]") {
  const Folder f;
  f.receipt(35.0, 30.0, 0.95);  // au-dessus des seuils lâches du FP8 (30, 26, 0,93), sous ceux de l'INT8
  const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir());
  CHECK_FALSE(found.variant);
  CHECK(found.note.find("sous les garde-fous") != std::string::npos);
  f.receipt(52.0, 37.0, 0.9967);  // le pire PSNR sous 38 dB
  CHECK_FALSE(find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir()).variant);
  f.receipt(52.0, 49.0, 0.980);  // le pire SSIM sous 0,985
  CHECK_FALSE(find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir()).variant);
  f.receipt(52.0, 49.0, 0.9967, "ajp-span-fp8c-v1", false);  // passed = false
  CHECK_FALSE(find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir()).variant);
}

TEST_CASE("SPAN FP8 CUTLASS : reçu d'un autre format, d'un autre modèle ou paquet modifié",
          "[app][int8][fp8c]") {
  const Folder f;
  f.receipt(52.0, 49.0, 0.9967, "ajp-span-fp8-v1");  // le format du FP8 par Q/DQ
  const SpanInt8Lookup wrong = find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir());
  CHECK_FALSE(wrong.variant);
  CHECK(wrong.note.find("format inconnu") != std::string::npos);

  f.receipt(52.0, 49.0, 0.9967);
  REQUIRE(find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir()).variant);
  Folder::write(f.span, "un autre graphe");  // le SPAN d'origine a changé : le reçu n'est plus le sien
  CHECK_FALSE(find_span_variant(SpanVariantKind::Fp8Cutlass, f.span, f.dir()).variant);

  const Folder g;
  g.receipt(52.0, 49.0, 0.9967);
  Folder::write(g.package(), "paquet retouché");
  const SpanInt8Lookup edited = find_span_variant(SpanVariantKind::Fp8Cutlass, g.span, g.dir());
  CHECK_FALSE(edited.variant);
  CHECK(edited.note.find("modifié depuis son reçu") != std::string::npos);
}

TEST_CASE("SPAN FP8 CUTLASS : le cran Performance demande le moteur qui sait le paquet",
          "[app][int8][fp8c]") {
  const Folder f;
  f.receipt(52.0, 49.0, 0.9967);
  CHECK(span_tiers(f.span, f.dir(), true, true).performance);
  CHECK_FALSE(span_tiers(f.span, f.dir(), true, false).performance);  // moteur FP8 sans les noyaux CUTLASS
  CHECK_FALSE(span_tiers(f.span, f.dir(), false, false).performance);
  CHECK_FALSE(span_tiers(f.span, f.dir(), true).performance);  // l'appel d'avant : la variante n'est pas lue
}

TEST_CASE("SPAN FP8 CUTLASS : le nom d'origine se lit dans celui du paquet", "[app][int8][fp8c]") {
  SpanVariantKind kind = SpanVariantKind::Int8;
  CHECK(span_int8_source("modele-fp8c-0123456789abcdef.span8", &kind) == "modele.onnx");
  CHECK(kind == SpanVariantKind::Fp8Cutlass);
  CHECK_FALSE(span_int8_source("modele-fp8c-0123456789abcdef.onnx"));  // un paquet n'est pas un ONNX
  CHECK_FALSE(span_int8_source("modele-fp8-0123456789abcdef.span8"));
  CHECK(span_int8_source("modele-fp8-0123456789abcdef.onnx", &kind) == "modele.onnx");
  CHECK(kind == SpanVariantKind::Fp8);
}

}  // namespace ajp::app::play
