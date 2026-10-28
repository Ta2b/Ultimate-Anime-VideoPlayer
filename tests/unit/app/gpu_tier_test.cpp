#include "app/play/gpu_tier.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

using namespace ajp;
using namespace ajp::app::play;

namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;

TierEstimate of(std::string name, GpuVendor vendor, std::uint64_t vram_gib = 8, bool tensor = false,
                bool wmma = false) {
  return estimate_tier(GpuFacts{.name = std::move(name),
                                .vendor = vendor,
                                .vram_bytes = vram_gib * kGiB,
                                .tensor_cores = tensor,
                                .wmma = wmma});
}

}  // namespace

TEST_CASE("la table embarquée se lit et couvre les quatre fabricants", "[gpu_tier]") {
  const TierTable& table = builtin_tier_table();
  REQUIRE(table.rules.size() > 50);
  for (const GpuVendor vendor : {GpuVendor::Nvidia, GpuVendor::Amd, GpuVendor::Intel, GpuVendor::Apple}) {
    CHECK(std::ranges::any_of(table.rules, [&](const TierRule& rule) { return rule.vendor == vendor; }));
  }
}

TEST_CASE("des cartes connues tombent dans la tranche attendue", "[gpu_tier]") {
  CHECK(of("NVIDIA GeForce RTX 4090", GpuVendor::Nvidia, 24, true).tier == Tier::Ultra);
  CHECK(of("NVIDIA GeForce RTX 4070 Ti", GpuVendor::Nvidia, 12, true).tier == Tier::Fast);
  CHECK(of("NVIDIA GeForce RTX 3060", GpuVendor::Nvidia, 12, true).tier == Tier::Normal);
  CHECK(of("NVIDIA GeForce GTX 1060 6GB", GpuVendor::Nvidia, 6).tier == Tier::Light);
  CHECK(of("NVIDIA GeForce GT 1030", GpuVendor::Nvidia, 2).tier == Tier::Economy);
  CHECK(of("AMD Radeon RX 7900 XTX (RADV NAVI31)", GpuVendor::Amd, 24, false, true).tier == Tier::Ultra);
  CHECK(of("AMD Radeon RX 6600", GpuVendor::Amd, 8).tier == Tier::Normal);
  CHECK(of("AMD Radeon RX 580 Series", GpuVendor::Amd, 8).tier == Tier::Light);
  CHECK(of("Intel(R) Arc(tm) A770 Graphics", GpuVendor::Intel, 16).tier == Tier::Normal);
  CHECK(of("Intel(R) UHD Graphics 630", GpuVendor::Intel, 1).tier == Tier::Economy);
  CHECK(of("Apple M3 Pro", GpuVendor::Apple, 18).tier == Tier::Fast);
  CHECK(of("Apple M2", GpuVendor::Apple, 8).tier == Tier::Normal);
}

TEST_CASE("les iGPU sont signalés", "[gpu_tier]") {
  CHECK(of("AMD Radeon Graphics (RADV GFX1103_R1)", GpuVendor::Amd, 1).integrated);
  CHECK(of("Intel(R) Iris(R) Xe Graphics", GpuVendor::Intel, 1).integrated);
  CHECK_FALSE(of("NVIDIA GeForce RTX 4070 Ti", GpuVendor::Nvidia, 12, true).integrated);
}

TEST_CASE("une carte inconnue reçoit une estimation par ses faits", "[gpu_tier]") {
  const TierEstimate rtx = of("NVIDIA GeForce RTX 9999", GpuVendor::Nvidia, 32, true);
  CHECK_FALSE(rtx.matched);
  CHECK(rtx.tier == Tier::Fast);
  CHECK(of("NVIDIA Future 1", GpuVendor::Nvidia, 8, true).tier == Tier::Normal);
  CHECK(of("Mystery GPU", GpuVendor::Other, 8).tier == Tier::Light);
  CHECK(of("Mystery GPU", GpuVendor::Other, 1).tier == Tier::Economy);
}

TEST_CASE("une règle plus précise passe avant une plus générale", "[gpu_tier]") {
  const auto table = parse_tier_table(R"(rules = [
    { vendor = "nvidia", contains = ["rtx 4070"], tier = 3 },
    { vendor = "nvidia", contains = ["rtx 40"], tier = 2 } ])");
  REQUIRE(table);
  const GpuFacts gpu{.name = "NVIDIA GeForce RTX 4070", .vendor = GpuVendor::Nvidia};
  CHECK(estimate_tier(*table, gpu).tier == Tier::Fast);
  CHECK(
      estimate_tier(*table, GpuFacts{.name = "NVIDIA GeForce RTX 4060", .vendor = GpuVendor::Nvidia}).tier ==
      Tier::Normal);
}

TEST_CASE("un gpu_tiers.toml invalide est refusé", "[gpu_tier]") {
  CHECK_FALSE(parse_tier_table("rules = ["));
  CHECK_FALSE(parse_tier_table("version = 1"));
  CHECK_FALSE(parse_tier_table(R"(rules = [ { vendor = "matrox", contains = ["x"], tier = 1 } ])"));
  CHECK_FALSE(parse_tier_table(R"(rules = [ { vendor = "amd", contains = ["x"], tier = 9 } ])"));
  CHECK_FALSE(parse_tier_table(R"(rules = [ { vendor = "amd", contains = [], tier = 1 } ])"));
}

TEST_CASE("les noms de tranche", "[gpu_tier]") {
  CHECK(tier_label(Tier::Economy) == "Économe");
  CHECK(tier_label(Tier::Ultra) == "Ultra");
}
