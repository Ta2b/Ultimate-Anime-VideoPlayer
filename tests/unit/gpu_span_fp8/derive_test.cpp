// Ce que toute dérivation du paquet « .span8 » partage : échelles, verdict, reçu, tailles prises (sans GPU).
#include "gpu/span_fp8/derive.h"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gpu/span_fp8/metrics.h"

namespace ajp::gpu::span_fp8 {

TEST_CASE("les échelles sont le maximum observé, avec sa marge de tête, sur 448 (B02-02)",
          "[span_fp8][derive]") {
  std::vector<float> peaks(kScaleCount, 4.48F);
  peaks[3] = 44.8F;
  const auto scales = scales_from_peaks(peaks);
  REQUIRE(scales);
  // Une activation 1,25 fois au-dessus du maximum de la calibration tient encore sans écrêter.
  CHECK((*scales)[0] == Catch::Approx(0.0125F));
  CHECK((*scales)[3] == Catch::Approx(0.125F));
  CHECK(1.25F * 44.8F / (*scales)[3] <= Catch::Approx(448.0F));
}

TEST_CASE("à 5 blocs, seules les 16 premières échelles servent (3.2b)", "[span_fp8][derive]") {
  std::vector<float> peaks(kScaleCount, 0.0F);  // au-delà de 1 + 3·5 : nuls, ignorés
  for (std::size_t i = 0; i < 16; ++i) peaks[i] = 4.48F;
  const auto scales = scales_from_peaks(peaks, 5);
  REQUIRE(scales);
  CHECK((*scales)[15] == Catch::Approx(0.0125F));
  CHECK((*scales)[16] == 1.0F);
  CHECK_FALSE(scales_from_peaks(peaks));  // à 8 blocs, le maximum 16 est nul
  CHECK_FALSE(scales_from_peaks(peaks, 0));
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

TEST_CASE("la porte durcie tient au 99,9e centile, au sombre et aux cas durs (3.7a)", "[span_fp8][derive]") {
  const std::vector<ImageScore> good{{50.0, 0.995, 2.0 / 255.0, 1.0 / 255.0},
                                     {46.0, 0.99, 3.0 / 255.0, 1.5 / 255.0}};
  const std::vector<ImageScore> hard{{40.0, 0.99, 3.5 / 255.0, 1.0 / 255.0}, {37.0, 0.98, 1.0 / 255.0, 0.0}};
  const Verdict ok = judge_hardened(good, hard);
  CHECK(ok.passed);
  CHECK(ok.error_p999_worst == Catch::Approx(3.0 / 255.0));  // les images de validation seules
  CHECK(ok.dark_error_p999_worst == Catch::Approx(1.5 / 255.0));
  CHECK(ok.hard_psnr_worst_db == Catch::Approx(37.0));
  CHECK(ok.hard_images == 2);
  CHECK(judge_hardened(good, {}).passed);  // sans cas dur : la partie centile seule

  std::vector<ImageScore> loud = good;
  loud[1].error_p999 = 10.5 / 255.0;
  CHECK_FALSE(judge_hardened(loud, hard).passed);
  std::vector<ImageScore> dark = good;
  dark[0].dark_error_p999 = 5.0 / 255.0;
  CHECK_FALSE(judge_hardened(dark, hard).passed);
  // Les cas durs synthétiques sont relevés sans décider (porte 2, séance de mesure).
  std::vector<ImageScore> broken = hard;
  broken[1].psnr_db = 16.4;
  broken[0].error_p999 = 125.0 / 255.0;
  const Verdict relaxed = judge_hardened(good, broken);
  CHECK(relaxed.passed);
  CHECK(relaxed.hard_psnr_worst_db == Catch::Approx(16.4));
  CHECK(relaxed.hard_error_p999_worst == Catch::Approx(125.0 / 255.0));
  CHECK(relaxed.error_p999_worst == Catch::Approx(3.0 / 255.0));  // celui des vraies images
  CHECK_FALSE(judge_hardened({}, hard).passed);                   // sans validation, rien ne passe
}

TEST_CASE("un score non fini refuse, où qu'il soit dans la liste (A02-01)", "[span_fp8][derive]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const ImageScore good{50.0, 0.995, 2.0 / 255.0, 1.0 / 255.0};
  CHECK(judge_hardened(std::vector<ImageScore>{good, good, good}, {}).passed);
  // std::min et std::max ignorent un NaN qui n'est pas le premier : chaque champ, à chaque rang.
  for (std::size_t at = 0; at < 3; ++at) {
    for (int field = 0; field < 4; ++field) {
      std::vector<ImageScore> scores{good, good, good};
      (field == 0   ? scores[at].psnr_db
       : field == 1 ? scores[at].ssim
       : field == 2 ? scores[at].error_p999
                    : scores[at].dark_error_p999) = nan;
      CAPTURE(at, field);
      CHECK_FALSE(judge(scores).passed);
      CHECK_FALSE(judge_hardened(scores, {}).passed);
    }
  }
  // L'image du rapport : deux bonnes et une dont les mesures voient maintenant le NaN (0 dB, SSIM 0, erreur
  // 1).
  const std::vector<ImageScore> with_nan{good, good, {0.0, 0.0, kNotFiniteError, kNotFiniteError}};
  CHECK_FALSE(judge_hardened(with_nan, {}).passed);
  CHECK_FALSE(keep_acc16(judge_hardened(std::vector<ImageScore>{good, good, good}, {}),
                         judge_hardened(with_nan, {}), 9.3, 8.9));
}

TEST_CASE("l'accumulateur f16 n'est gardé qu'à 0,3 dB de PSNR moyen du f32, porte passée (3.5a)",
          "[span_fp8][derive]") {
  const Verdict f32{.psnr_mean_db = 48.0, .psnr_worst_db = 44.0, .ssim_worst = 0.99, .passed = true};
  Verdict f16 = f32;
  f16.psnr_mean_db = 47.75;
  CHECK(keep_acc16(f32, f16, 9.3, 8.9));
  f16.psnr_mean_db = 47.6;
  CHECK_FALSE(keep_acc16(f32, f16, 9.3, 8.9));  // perd 0,4 dB
  f16.psnr_mean_db = 48.2;
  f16.passed = false;
  CHECK_FALSE(keep_acc16(f32, f16, 9.3, 8.9));  // meilleur en moyenne, mais la porte ne passe pas
  Verdict failed = f32;
  failed.passed = false;
  f16.passed = true;
  CHECK_FALSE(keep_acc16(failed, f16, 9.3, 8.9));
}

TEST_CASE("l'accumulateur f16 n'est gardé que s'il va plus vite (C02-03)", "[span_fp8][derive]") {
  // Le cas du rapport d'audit : −0,25 dB, et une variante plus lente ; sans les temps, elle était gardée.
  const Verdict f32{.psnr_mean_db = 48.0, .psnr_worst_db = 44.0, .ssim_worst = 0.99, .passed = true};
  Verdict f16 = f32;
  f16.psnr_mean_db = 47.75;
  CHECK_FALSE(keep_acc16(f32, f16, 9.3, 9.6));   // plus lente
  CHECK_FALSE(keep_acc16(f32, f16, 9.3, 9.25));  // 0,5 % : sous kMinAcc16Gain
  CHECK(keep_acc16(f32, f16, 9.3, 9.3 / (1.0 + kMinAcc16Gain)));
  CHECK_FALSE(keep_acc16(f32, f16, 0.0, 8.9));  // un temps non mesuré ne garde rien
  CHECK_FALSE(keep_acc16(f32, f16, 9.3, 0.0));
}

TEST_CASE("un tenseur qui dépasse 448 × son échelle sur la validation refuse (C02-04)",
          "[span_fp8][derive]") {
  std::array<float, kScaleCount> peaks{};
  peaks.fill(10.0F);
  const auto scales = scales_from_peaks(peaks, kBlocks);
  REQUIRE(scales);
  const Verdict passed{.psnr_mean_db = 48.0, .psnr_worst_db = 44.0, .ssim_worst = 0.99, .passed = true};
  Verdict verdict = passed;
  judge_peaks(verdict, peaks, *scales);  // les maxima de la calibration : la marge de 1,25
  CHECK(verdict.passed);
  CHECK(verdict.clipped_tensors == 0);
  CHECK(verdict.peak_ratio == Catch::Approx(1.0 / kScaleHeadroom));
  std::array<float, kScaleCount> seen = peaks;
  seen[7] = 12.5F;  // juste à la marge : pas d'écrêtage
  verdict = passed;
  judge_peaks(verdict, seen, *scales);
  CHECK(verdict.passed);
  seen[7] = 13.0F;  // au-delà
  seen[24] = std::numeric_limits<float>::infinity();
  verdict = passed;
  judge_peaks(verdict, seen, *scales);
  CHECK_FALSE(verdict.passed);
  CHECK(verdict.clipped_tensors == 2);
  CHECK(std::isinf(verdict.peak_ratio));
  // À 5 blocs, les tenseurs au-delà du 16e ne servent pas.
  verdict = passed;
  judge_peaks(verdict, seen, *scales, 5);
  CHECK(verdict.clipped_tensors == 1);
  CHECK(verdict.peak_ratio == Catch::Approx(13.0 / 12.5));
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
  CHECK(text.starts_with("# essai\nformat = \"ajp-span-fp8c-v2\"\nsource = \"2x_AnimeJaNai.onnx\"\n"));
  CHECK(text.find("quantizer = \"ajp \\\"natif\\\"\"\n") != std::string::npos);
  CHECK(text.find("precision = \"fp8\"\n") != std::string::npos);
  CHECK(text.find("kernels_ms = 9.2360\n") != std::string::npos);
  CHECK(text.find("psnr_mean_db = 52.0409\n") != std::string::npos);
  CHECK(text.find("ssim_worst = 0.9967\n") != std::string::npos);
  CHECK(text.find("passed = true\n") != std::string::npos);
  CHECK(text.find("reference = \"cutlass-fp16\"\n") != std::string::npos);
  CHECK(text.find("derived_by") == std::string::npos);  // vide : omis
}

TEST_CASE("le reçu v2 lie le paquet à l'appareil, aux noyaux et aux tailles jugées (3.7b)",
          "[span_fp8][derive]") {
  Receipt receipt;
  receipt.gpu_arch = "sm_89";
  receipt.kernels_version = 1;
  receipt.blocks = 5;
  receipt.channels = 48;
  receipt.acc16 = true;
  receipt.verdict.error_p999_worst = 3.0 / 255.0;
  receipt.verdict.hard_images = 9;
  receipt.validated_sizes = {"1280x720"};
  receipt.timed_size = "1280x720";
  receipt.acc16_psnr_loss_db = 0.21;
  receipt.kernels_ms_f32 = 9.3;
  receipt.verdict.peak_ratio = 0.9;
  const std::string text = receipt_text(receipt, "essai");
  // Chantier E01 : la taille des temps, l'écrêtage, ce que la variante f16 a coûté.
  CHECK(text.find("timed_size = \"1280x720\"\nclipped_tensors = 0\npeak_ratio = 0.9000\n") !=
        std::string::npos);
  CHECK(text.find("acc16_psnr_loss_db = 0.2100\nkernels_ms_f32 = 9.3000\n") != std::string::npos);
  CHECK(text.find("gpu_arch = \"sm_89\"\n") != std::string::npos);
  CHECK(text.find("kernels_version = 1\n") != std::string::npos);
  CHECK(text.find("blocks = 5\nchannels = 48\naccumulator = \"f16\"\n") != std::string::npos);
  CHECK(text.find("error_p999_worst = 0.011765\n") != std::string::npos);
  CHECK(text.find("hard_images = 9\n") != std::string::npos);
  CHECK(text.ends_with("validated_sizes = [\"1280x720\"]\nrefused_sizes = []\n"));

  const std::string more = with_sizes(text, {"1280x720", "1920x1080"}, {"960x540"});
  CHECK(more.starts_with("# essai\n"));
  CHECK(more.ends_with("validated_sizes = [\"1280x720\", \"1920x1080\"]\nrefused_sizes = [\"960x540\"]\n"));
  CHECK(more.find("validated_sizes = [\"1280x720\"]\n") == std::string::npos);  // l'ancienne ligne remplacée
  CHECK(size_key(1280, 720) == "1280x720");
  // C02-05 : la taille de travail entière, paire ; une taille au-delà de la découpe a sa propre clé.
  CHECK(size_key(1919, 1081) == "1918x1080");
  CHECK(size_key(2560, 1440) != size_key(1920, 1088));
  CHECK(size_key(3840, 2160) != size_key(2560, 1440));
  CHECK(size_key(3840, 2160) == "3840x2160");
}

TEST_CASE("les images de dérivation se ramènent à une taille que les noyaux prennent", "[span_fp8][derive]") {
  CHECK(kernel_crop(1920, 1080).width == 1920);
  CHECK(kernel_crop(1920, 1080).height == 1080);
  CHECK(kernel_crop(3840, 2160).width == 1920);  // une découpe, pas une réduction
  CHECK(kernel_crop(3840, 2160).height == 1088);
  CHECK(kernel_crop(3840, 2160).width == kCropMaxWidth);  // C02-11 : une seule constante
  CHECK(kernel_crop(3840, 2160).height == kCropMaxHeight);
  CHECK(kernel_crop(1280, 720).width == 1280);
  CHECK(kernel_crop(1000, 1000).width == 1000);  // toute largeur paire (étape 3.1)
  CHECK(kernel_crop(999, 1000).width == 998);
  const Crop small = kernel_crop(1, 600);  // moins de deux colonnes : rien
  CHECK(small.width == 0);
  CHECK(small.height == 0);
  for (const auto& [w, h] : {std::pair{1920, 1080}, {1280, 720}, {3840, 2160}, {1000, 1000}, {2560, 1440}}) {
    const Crop crop = kernel_crop(w, h);
    CHECK(size_supported(crop.height, crop.width));
  }
}

}  // namespace ajp::gpu::span_fp8
