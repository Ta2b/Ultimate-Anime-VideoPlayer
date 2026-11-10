#include "core/engines.h"

#include <algorithm>
#include <array>

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

  CHECK_FALSE(find(offers, EngineId::Metal));
  CHECK(offers.size() == 2);
}

TEST_CASE("Moteurs : AMD et Intel passent par Vulkan, seul proposé", "[engines]") {
  const std::array systems{
      SystemInfo{.os = Os::Linux, .gpus = {GpuInfo{.vendor = GpuVendor::Amd, .name = "RX 7900 XTX"}}},
      SystemInfo{.os = Os::Windows,
                 .gpus = {GpuInfo{.vendor = GpuVendor::Amd, .name = "AMD Radeon(TM) 860M Graphics"}}},
      SystemInfo{.os = Os::Windows, .gpus = {GpuInfo{.vendor = GpuVendor::Amd, .name = "RX 580"}}},
      SystemInfo{.os = Os::Linux, .gpus = {GpuInfo{.vendor = GpuVendor::Intel, .name = "Arc A770"}}}};
  for (const SystemInfo& system : systems) {
    const auto offers = recommend_engines(system);
    CHECK(offers.size() == 1);  // P3.5 : ni case ni écran Moteurs hors RTX
    CHECK(has(*find(offers, EngineId::Vulkan), Badge::Recommended));
  }
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
