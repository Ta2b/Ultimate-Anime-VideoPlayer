// Table des encodeurs et refus des réglages, sans ouvrir FFmpeg.
#include <catch2/catch_test_macros.hpp>

#include "export/encoder_settings.h"
#include "export/hdr10.h"

namespace ajp::exporter {
namespace {

RefusalCode code_of(const EncoderSettings& settings, int width = 1920, int height = 1080) {
  auto result = validate_settings(settings, width, height);
  REQUIRE_FALSE(result.has_value());
  return result.error().code;
}

EncoderSettings nvenc_cq(int quality) {
  return EncoderSettings{.encoder = "hevc_nvenc",
                         .format = PixelFormat::P010,
                         .rate = RateControl::ConstantQuality,
                         .quality = quality};
}

}  // namespace

TEST_CASE("la table nomme les encodeurs pilotés", "[export][settings]") {
  for (const char* name : {"ffv1", "hevc_nvenc", "av1_nvenc", "libsvtav1", "hevc_vulkan", "av1_vulkan"}) {
    auto spec = find_encoder_spec(name);
    REQUIRE(spec.has_value());
    CHECK((*spec)->name == name);
  }
  CHECK(find_encoder_spec("libx265").error().code == RefusalCode::EncoderUnavailable);
}

TEST_CASE("des réglages valides passent", "[export][settings]") {
  CHECK(validate_settings(EncoderSettings{}, 1920, 1080).has_value());
  CHECK(validate_settings(EncoderSettings{.format = PixelFormat::Yuv420p10}, 1920, 1080).has_value());
  CHECK(validate_settings(nvenc_cq(20), 3840, 2160).has_value());
  const EncoderSettings bitrate{.encoder = "av1_nvenc",
                                .format = PixelFormat::P010,
                                .rate = RateControl::Bitrate,
                                .bitrate_kbps = 20'000,
                                .max_bitrate_kbps = 40'000,
                                .preset = 7,
                                .keyframe_interval = 240};
  CHECK(validate_settings(bitrate, 1920, 1080).has_value());
  const EncoderSettings svt{.encoder = "libsvtav1",
                            .format = PixelFormat::Yuv420p10,
                            .rate = RateControl::ConstantQuality,
                            .quality = 24,
                            .preset = 0};
  CHECK(validate_settings(svt, 1920, 1080).has_value());
}

TEST_CASE("encodeur inconnu refusé par son nom", "[export][settings]") {
  auto result = validate_settings(EncoderSettings{.encoder = "x264"}, 1920, 1080);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == RefusalCode::EncoderUnavailable);
  CHECK(result.error().detail.find("x264") != std::string::npos);
}

TEST_CASE("taille impaire refusée en 4:2:0, permise en RGB", "[export][settings]") {
  CHECK(code_of(EncoderSettings{.format = PixelFormat::Yuv420p10}, 1919, 1080) ==
        RefusalCode::FieldOutOfRange);
  CHECK(code_of(nvenc_cq(20), 1920, 1081) == RefusalCode::FieldOutOfRange);
  CHECK(validate_settings(EncoderSettings{}, 1919, 1081).has_value());
  CHECK(code_of(EncoderSettings{}, 8, 1080) == RefusalCode::FieldOutOfRange);
  CHECK(code_of(EncoderSettings{}, 1920, 20000) == RefusalCode::FieldOutOfRange);
}

TEST_CASE("options hors plage ou sans objet refusées, jamais corrigées", "[export][settings]") {
  CHECK(code_of(nvenc_cq(0)) == RefusalCode::FieldOutOfRange);
  CHECK(code_of(nvenc_cq(52)) == RefusalCode::FieldOutOfRange);
  CHECK(validate_settings(EncoderSettings{.encoder = "av1_nvenc",
                                          .format = PixelFormat::P010,
                                          .rate = RateControl::ConstantQuality,
                                          .quality = 60},
                          1920, 1080)
            .has_value());
  auto preset = nvenc_cq(20);
  preset.preset = 8;
  CHECK(code_of(preset) == RefusalCode::FieldOutOfRange);
  CHECK(code_of(EncoderSettings{.preset = 3}) == RefusalCode::FieldOutOfRange);  // FFV1 n'a pas de preset
  CHECK(code_of(EncoderSettings{.quality = 10}) == RefusalCode::FieldOutOfRange);
  CHECK(code_of(EncoderSettings{.keyframe_interval = 0}) == RefusalCode::FieldOutOfRange);
  const EncoderSettings low{
      .encoder = "hevc_nvenc", .format = PixelFormat::P010, .rate = RateControl::Bitrate, .bitrate_kbps = 50};
  CHECK(code_of(low) == RefusalCode::FieldOutOfRange);
  auto capped = low;
  capped.bitrate_kbps = 10'000;
  capped.max_bitrate_kbps = 5'000;
  CHECK(code_of(capped) == RefusalCode::FieldOutOfRange);
  auto stray = nvenc_cq(20);
  stray.bitrate_kbps = 8000;
  CHECK(code_of(stray) == RefusalCode::FieldOutOfRange);
}

TEST_CASE("format ou mode que l'encodeur ne prend pas refusés", "[export][settings]") {
  CHECK(code_of(EncoderSettings{.format = PixelFormat::P010}) == RefusalCode::FormatUnsupported);
  CHECK(code_of(EncoderSettings{.encoder = "hevc_nvenc", .format = PixelFormat::Rgb16}) ==
        RefusalCode::FormatUnsupported);
  CHECK(code_of(EncoderSettings{.encoder = "hevc_nvenc", .format = PixelFormat::P010}) ==
        RefusalCode::FormatUnsupported);  // NVENC n'est pas sans perte ici
  CHECK(code_of(EncoderSettings{.rate = RateControl::ConstantQuality, .quality = 10}) ==
        RefusalCode::FormatUnsupported);
}

TEST_CASE("mastering HDR10 validé", "[export][settings]") {
  CHECK(validate_mastering(Hdr10Mastering{}).has_value());
  CHECK(validate_mastering(Hdr10Mastering{.max_nits = 400.0}).has_value());
  CHECK(validate_mastering(Hdr10Mastering{.max_nits = 20000.0}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(validate_mastering(Hdr10Mastering{.max_nits = 0.5}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(validate_mastering(Hdr10Mastering{.max_nits = 100.0, .min_nits = 100.0}).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(validate_mastering(Hdr10Mastering{.red = {.x = 1.2, .y = 0.3}}).error().code ==
        RefusalCode::FieldOutOfRange);
}

}  // namespace ajp::exporter
