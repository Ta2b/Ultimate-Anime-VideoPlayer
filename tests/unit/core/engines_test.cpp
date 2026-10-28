#include "core/engines.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::core;

namespace {

const EngineOffer* find(const std::vector<EngineOffer>& offers, EngineId id) {
  auto it = std::ranges::find(offers, id, &EngineOffer::id);
  return it == offers.end() ? nullptr : &*it;
}

bool has(const EngineOffer& offer, Badge badge) {
  return std::ranges::contains(offer.badges, badge);
}

}  // namespace

TEST_CASE("Moteurs : RTX 4070 Ti sous Windows", "[engines]") {
  SystemInfo system{
      .os = Os::Windows,
      .gpus = {GpuInfo{.vendor = GpuVendor::Nvidia, .name = "RTX 4070 Ti", .tensor_cores = true}}};
  auto offers = recommend_engines(system);

  const EngineOffer* vulkan = find(offers, EngineId::Vulkan);
  REQUIRE(vulkan);
  CHECK(vulkan->delivery == Delivery::Bundled);
  CHECK(vulkan->preselected);
  CHECK(has(*vulkan, Badge::Compatibility));

  const EngineOffer* trt = find(offers, EngineId::TensorRt);
  REQUIRE(trt);
  CHECK(trt->preselected);
  CHECK(has(*trt, Badge::Recommended));
  CHECK(has(*trt, Badge::NvidiaOnly));

  const EngineOffer* rtx_hdr = find(offers, EngineId::RtxVideoSdk);
  REQUIRE(rtx_hdr);
  CHECK(rtx_hdr->delivery == Delivery::Import);
  CHECK(rtx_hdr->preselected);

  CHECK_FALSE(find(offers, EngineId::MaxineVfx));
  CHECK_FALSE(find(offers, EngineId::Metal));
  CHECK_FALSE(find(offers, EngineId::MiGraphX)->relevant);
  CHECK_FALSE(find(offers, EngineId::DirectMl));
}

TEST_CASE("Moteurs : AMD récent sous Linux", "[engines]") {
  SystemInfo system{
      .os = Os::Linux,
      .gpus = {GpuInfo{.vendor = GpuVendor::Amd, .name = "RX 7900 XTX", .rocm_supported = true}}};
  auto offers = recommend_engines(system);
  CHECK(find(offers, EngineId::MiGraphX)->preselected);
  CHECK(has(*find(offers, EngineId::MiGraphX), Badge::AmdOnly));
  CHECK(find(offers, EngineId::MaxineVfx));
  CHECK_FALSE(find(offers, EngineId::MaxineVfx)->preselected);
  CHECK_FALSE(find(offers, EngineId::DirectMl));
  CHECK(has(*find(offers, EngineId::Vulkan), Badge::Compatibility));
}

TEST_CASE("Moteurs : ancien AMD sous Windows, Vulkan recommandé et aucune offre DirectML", "[engines]") {
  SystemInfo system{.os = Os::Windows, .gpus = {GpuInfo{.vendor = GpuVendor::Amd, .name = "RX 580"}}};
  auto offers = recommend_engines(system);
  CHECK_FALSE(find(offers, EngineId::DirectMl));
  CHECK(has(*find(offers, EngineId::Vulkan), Badge::Recommended));
}

TEST_CASE("Moteurs : Radeon 860M sous Windows, MIGraphX de Windows ML à l'essai", "[engines]") {
  SystemInfo system{
      .os = Os::Windows,
      .gpus = {
          GpuInfo{.vendor = GpuVendor::Amd, .name = "AMD Radeon(TM) 860M Graphics", .winml_migraphx = true}}};
  auto offers = recommend_engines(system);
  const EngineOffer* migraphx = find(offers, EngineId::MiGraphX);
  REQUIRE(migraphx);
  CHECK(migraphx->relevant);
  CHECK_FALSE(migraphx->preselected);
  CHECK(migraphx->reason == Reason::CompareWithVulkan);
  CHECK(migraphx->delivery == Delivery::Download);
  CHECK(has(*migraphx, Badge::AmdOnly));
  CHECK_FALSE(has(*migraphx, Badge::Recommended));
  CHECK_FALSE(find(offers, EngineId::DirectMl));
  // À l'essai seulement : Vulkan reste le moteur recommandé.
  CHECK(has(*find(offers, EngineId::Vulkan), Badge::Recommended));
}

TEST_CASE("Moteurs : ROCm ne compte que sous Linux, Windows ML que sous Windows", "[engines]") {
  SystemInfo windows{.os = Os::Windows,
                     .gpus = {GpuInfo{.vendor = GpuVendor::Amd, .name = "RX 6800", .rocm_supported = true}}};
  CHECK_FALSE(find(recommend_engines(windows), EngineId::MiGraphX)->relevant);
  CHECK_FALSE(find(recommend_engines(windows), EngineId::DirectMl));
  SystemInfo linux_apu{
      .os = Os::Linux,
      .gpus = {GpuInfo{.vendor = GpuVendor::Amd, .name = "AMD Radeon 860M", .winml_migraphx = true}}};
  CHECK_FALSE(find(recommend_engines(linux_apu), EngineId::MiGraphX)->relevant);
}

TEST_CASE("Moteurs : Mac, Metal inclus et recommandé", "[engines]") {
  SystemInfo system{.os = Os::MacOs, .gpus = {GpuInfo{.vendor = GpuVendor::Apple, .name = "M3"}}};
  auto offers = recommend_engines(system);
  const EngineOffer* metal = find(offers, EngineId::Metal);
  REQUIRE(metal);
  CHECK(metal->delivery == Delivery::Bundled);
  CHECK(has(*metal, Badge::Recommended));
  CHECK_FALSE(find(offers, EngineId::TensorRt));
  CHECK(has(*find(offers, EngineId::Vulkan), Badge::Compatibility));
}

TEST_CASE("Moteurs : sans GPU reconnu, Vulkan seul est recommandé", "[engines]") {
  auto offers = recommend_engines(SystemInfo{.os = Os::Linux});
  CHECK(has(*find(offers, EngineId::Vulkan), Badge::Recommended));
  CHECK(std::ranges::none_of(offers,
                             [](const EngineOffer& o) { return o.id != EngineId::Vulkan && o.preselected; }));
}
