// Le choix de l'encodeur d'un export : l'ordre suit le GPU du rendu, H.264 en 8 bits, FFV1 seul
// sans perte, et le jeu réduit d'options toujours compris dans le jeu complet.
#include "export/encoder_choice.h"

#include <algorithm>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace ajp::exporter {
namespace {

std::vector<std::string> names(const std::vector<EncoderCandidate>& candidates) {
  std::vector<std::string> out;
  for (const EncoderCandidate& c : candidates) out.push_back(c.name);
  return out;
}

bool has(const std::vector<EncoderOption>& options, const std::string& key, const std::string& value) {
  return std::ranges::find(options, EncoderOption{.key = key, .value = value}) != options.end();
}

constexpr GpuVendor kVendors[] = {GpuVendor::Nvidia, GpuVendor::Amd, GpuVendor::Intel, GpuVendor::Apple,
                                  GpuVendor::Other};

}  // namespace

TEST_CASE("Encodeur : le constructeur d'un identifiant PCI", "[export]") {
  CHECK(vendor_of(0x10DE) == GpuVendor::Nvidia);
  CHECK(vendor_of(0x1002) == GpuVendor::Amd);
  CHECK(vendor_of(0x1022) == GpuVendor::Amd);
  CHECK(vendor_of(0x8086) == GpuVendor::Intel);
  CHECK(vendor_of(0x106B) == GpuVendor::Apple);
  CHECK(vendor_of(0x13B5) == GpuVendor::Other);
}

TEST_CASE("Encodeur : celui du GPU du rendu d'abord", "[export]") {
  CHECK(names(encoder_candidates(Codec::Hevc, Quality::VeryHigh, GpuVendor::Nvidia)).front() == "hevc_nvenc");
  CHECK(names(encoder_candidates(Codec::Hevc, Quality::VeryHigh, GpuVendor::Amd)).front() == "hevc_amf");
  CHECK(names(encoder_candidates(Codec::Hevc, Quality::VeryHigh, GpuVendor::Intel)).front() == "hevc_qsv");
  CHECK(names(encoder_candidates(Codec::Hevc, Quality::VeryHigh, GpuVendor::Apple)).front() ==
        "hevc_videotoolbox");
  CHECK(names(encoder_candidates(Codec::Av1, Quality::High, GpuVendor::Nvidia)).front() == "av1_nvenc");
  CHECK(names(encoder_candidates(Codec::H264, Quality::High, GpuVendor::Amd)).front() == "h264_amf");
}

TEST_CASE("Encodeur : VA-API et Vulkan Video chargent leurs surfaces depuis l'hôte", "[export]") {
  for (const EncoderCandidate& c : encoder_candidates(Codec::Hevc, Quality::High, GpuVendor::Amd)) {
    if (c.name == "hevc_vaapi")
      CHECK(c.upload == Upload::Vaapi);
    else if (c.name == "hevc_vulkan")
      CHECK(c.upload == Upload::Vulkan);
    else
      CHECK(c.upload == Upload::Host);
  }
}

TEST_CASE("Encodeur : 10 bits sauf H.264, pas d'AV1 par Quick Sync ni VideoToolbox", "[export]") {
  for (const GpuVendor vendor : kVendors) {
    for (const EncoderCandidate& c : encoder_candidates(Codec::Hevc, Quality::High, vendor)) CHECK(c.ten_bit);
    for (const EncoderCandidate& c : encoder_candidates(Codec::H264, Quality::High, vendor))
      CHECK_FALSE(c.ten_bit);
    for (const EncoderCandidate& c : encoder_candidates(Codec::Av1, Quality::High, vendor)) {
      CHECK(c.name != "av1_qsv");
      CHECK(c.name != "av1_videotoolbox");
    }
  }
}

TEST_CASE("Encodeur : sans perte, FFV1 seul, quel que soit le GPU", "[export]") {
  for (const GpuVendor vendor : kVendors) {
    const std::vector<EncoderCandidate> candidates =
        encoder_candidates(Codec::Lossless, Quality::VeryHigh, vendor);
    REQUIRE(candidates.size() == 1);
    CHECK(candidates.front().name == "ffv1");
    CHECK(candidates.front().upload == Upload::Host);
  }
}

TEST_CASE("Encodeur : toujours au moins un candidat, chacun avec des options", "[export]") {
  for (int codec = 0; codec < kCodecCount; ++codec) {
    for (int quality = 0; quality < kQualityCount; ++quality) {
      for (const GpuVendor vendor : kVendors) {
        const auto candidates =
            encoder_candidates(static_cast<Codec>(codec), static_cast<Quality>(quality), vendor);
        REQUIRE_FALSE(candidates.empty());
        for (const EncoderCandidate& c : candidates) {
          CHECK_FALSE(c.name.empty());
          CHECK_FALSE(c.label.empty());
          CHECK_FALSE(c.options.empty());
          CHECK_FALSE(c.fallback.empty());
        }
      }
    }
  }
}

TEST_CASE("Encodeur : NVENC, quantification et passes selon la qualité", "[export]") {
  const EncoderCandidate best = encoder_candidates(Codec::Hevc, Quality::VeryHigh, GpuVendor::Nvidia).front();
  CHECK(has(best.options, "cq", "17"));
  CHECK(has(best.options, "multipass", "qres"));
  CHECK(has(best.options, "temporal-aq", "1"));
  // Le jeu réduit garde le contrôle de débit, sans l'AQ temporelle qu'un vieux GPU refuse.
  CHECK(has(best.fallback, "cq", "17"));
  CHECK_FALSE(has(best.fallback, "temporal-aq", "1"));
  for (const EncoderOption& option : best.fallback) CHECK(has(best.options, option.key, option.value));

  const EncoderCandidate standard =
      encoder_candidates(Codec::Hevc, Quality::Standard, GpuVendor::Nvidia).front();
  CHECK(has(standard.options, "cq", "24"));
  CHECK_FALSE(has(standard.options, "multipass", "qres"));
}

TEST_CASE("Encodeur : libellés", "[export]") {
  CHECK(label(Codec::Hevc) == "HEVC 10 bits");
  CHECK(label(Codec::Lossless) == "Sans perte (FFV1)");
  CHECK(label(Quality::VeryHigh) == "Très haute");
  CHECK(label(Quality::Standard) == "Standard");
}

}  // namespace ajp::exporter
