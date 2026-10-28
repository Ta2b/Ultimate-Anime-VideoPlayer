// Ce que toute dérivation du paquet « .span8 » partage : échelles, verdict, reçu, tailles prises (sans GPU).
#include "gpu/span_fp8/derive.h"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::span_fp8 {

TEST_CASE("les échelles sont le maximum observé sur 448", "[span_fp8][derive]") {
  std::vector<float> peaks(kScaleCount, 4.48F);
  peaks[3] = 44.8F;
  const auto scales = scales_from_peaks(peaks);
  REQUIRE(scales);
  CHECK((*scales)[0] == Catch::Approx(0.01F));
  CHECK((*scales)[3] == Catch::Approx(0.1F));
}

TEST_CASE("un maximum nul, infini ou absent refuse la dérivation", "[span_fp8][derive]") {
  std::vector<float> peaks(kScaleCount, 1.0F);
  peaks[7] = 0.0F;
  const auto zero = scales_from_peaks(peaks);
  REQUIRE_FALSE(zero);
  CHECK(zero.error().detail.find("nul partout") != std::string::npos);

  peaks[7] = std::numeric_limits<float>::infinity();
  CHECK_FALSE(scales_from_peaks(peaks));
  peaks[7] = std::nanf("");
  CHECK_FALSE(scales_from_peaks(peaks));
  peaks.pop_back();
  CHECK_FALSE(scales_from_peaks(peaks));
}

TEST_CASE("le verdict tient aux trois seuils du reçu", "[span_fp8][derive]") {
  const std::vector<ImageScore> good{{50.0, 0.995}, {46.0, 0.99}, {44.0, 0.991}, {47.0, 0.993}};
  const Verdict ok = judge(good);
  CHECK(ok.passed);
  CHECK(ok.psnr_mean_db == Catch::Approx(46.75));
  CHECK(ok.psnr_worst_db == Catch::Approx(44.0));
  CHECK(ok.ssim_worst == Catch::Approx(0.99));

  CHECK(judge(std::vector<ImageScore>{{40.0, 0.99}, {44.5, 0.99}}).passed);         // moyenne 42,25, pire 40
  CHECK_FALSE(judge(std::vector<ImageScore>{{40.0, 0.99}, {41.0, 0.99}}).passed);   // moyenne sous 42
  CHECK_FALSE(judge(std::vector<ImageScore>{{50.0, 0.99}, {37.9, 0.99}}).passed);   // pire sous 38
  CHECK_FALSE(judge(std::vector<ImageScore>{{50.0, 0.99}, {50.0, 0.984}}).passed);  // SSIM sous 0,985
  CHECK_FALSE(judge({}).passed);                                                    // aucune image
}

TEST_CASE("le reçu a les clés et la mise en forme de l'outil Python", "[span_fp8][derive]") {
  Receipt receipt;
  receipt.source = "2x_AnimeJaNai.onnx";
  receipt.source_sha256 = "aa";
  receipt.model_sha256 = "bb";
  receipt.quantizer = "ajp \"natif\"";
  receipt.kernels_ms = 9.236;
  receipt.calibration_images = 8;
  receipt.validation_images = 4;
  receipt.validation_size = "1920x1080";
  receipt.verdict =
      Verdict{.psnr_mean_db = 52.04087, .psnr_worst_db = 49.2715, .ssim_worst = 0.99673, .passed = true};
  receipt.reference = "cutlass-fp16";
  const std::string text = receipt_text(receipt, "essai");
  CHECK(text.starts_with("# essai\nformat = \"ajp-span-fp8c-v1\"\nsource = \"2x_AnimeJaNai.onnx\"\n"));
  CHECK(text.find("quantizer = \"ajp \\\"natif\\\"\"\n") != std::string::npos);
  CHECK(text.find("precision = \"fp8\"\n") != std::string::npos);
  CHECK(text.find("kernels_ms = 9.2360\n") != std::string::npos);
  CHECK(text.find("psnr_mean_db = 52.0409\n") != std::string::npos);
  CHECK(text.find("ssim_worst = 0.9967\n") != std::string::npos);
  CHECK(text.find("passed = true\n") != std::string::npos);
  CHECK(text.find("reference = \"cutlass-fp16\"\n") != std::string::npos);
  CHECK(text.find("derived_by") == std::string::npos);  // vide : omis
}

TEST_CASE("les images de dérivation se ramènent à une taille que les noyaux prennent", "[span_fp8][derive]") {
  CHECK(kernel_crop(1920, 1080).width == 1920);
  CHECK(kernel_crop(1920, 1080).height == 1080);
  CHECK(kernel_crop(3840, 2160).width == 1920);  // une découpe, pas une réduction
  CHECK(kernel_crop(3840, 2160).height == 1088);
  CHECK(kernel_crop(1280, 720).width == 1280);
  CHECK(kernel_crop(1000, 1000).width == 896);  // multiple de 128
  const Crop small = kernel_crop(100, 600);
  CHECK(small.width == 0);
  CHECK(small.height == 0);
  for (const auto& [w, h] : {std::pair{1920, 1080}, {1280, 720}, {3840, 2160}, {1000, 1000}, {2560, 1440}}) {
    const Crop crop = kernel_crop(w, h);
    CHECK(size_supported(crop.height, crop.width));
  }
}

}  // namespace ajp::gpu::span_fp8
