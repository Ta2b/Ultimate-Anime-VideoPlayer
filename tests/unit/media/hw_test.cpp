// Chemins matériels qui peuvent tourner ici : refus nommés sans pilote CUDA,
// appareil Vulkan sur lavapipe, appareil partagé par un autre module.
#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/hwcontext.h>
}

#include "media/cuda_driver.h"
#include "media/hw_device.h"

using namespace ajp::media;
using ajp::RefusalCode;

namespace {

// HEVC a un hwaccel Vulkan dans cette construction de FFmpeg.
constexpr AVCodecID kVulkanCodec = AV_CODEC_ID_HEVC;

class SharedDevice final : public HwDeviceSource {
 public:
  SharedDevice(HwApi api, AVBufferRef* device) : api_(api), device_(device) {}
  AVBufferRef* device(HwApi api) const override { return api == api_ ? av_buffer_ref(device_) : nullptr; }

 private:
  HwApi api_;
  AVBufferRef* device_;
};

}  // namespace

TEST_CASE("hw : sans pilote CUDA, le chargement est un refus nommé", "[media][hw]") {
  auto driver = CudaDriver::load();
  if (driver) SKIP("pilote CUDA présent : chemin de refus non exercé");
  CHECK(driver.error().code == RefusalCode::HwDecodeUnavailable);
  CHECK_FALSE(driver.error().detail.empty());
}

TEST_CASE("hw : CUDA exige l'UUID du rendu", "[media][hw]") {
  auto device = open_hw_device(AV_CODEC_ID_HEVC, HwApi::Cuda, DecoderOptions{});
  REQUIRE_FALSE(device);
  CHECK(device.error().code == RefusalCode::HwDecodeUnavailable);
  CHECK(device.error().detail.find("UUID") != std::string::npos);
}

TEST_CASE("hw : une API sans décodeur pour ce codec est écartée avant tout appareil", "[media][hw]") {
  auto device = open_hw_device(AV_CODEC_ID_FFV1, HwApi::Vaapi, DecoderOptions{});
  REQUIRE_FALSE(device);
  CHECK(device.error().code == RefusalCode::HwDecodeUnavailable);
  CHECK(device.error().detail.starts_with("vaapi"));
  auto software = open_hw_device(AV_CODEC_ID_HEVC, HwApi::Software, DecoderOptions{});
  REQUIRE_FALSE(software);
}

TEST_CASE("hw : appareil Vulkan choisi par l'UUID du rendu (lavapipe)", "[media][hw][vulkan]") {
  auto any = open_hw_device(kVulkanCodec, HwApi::Vulkan, DecoderOptions{});
  if (!any) SKIP("aucun appareil Vulkan : " << any.error().detail);
  REQUIRE(any->device);
  CHECK(any->format == AV_PIX_FMT_VULKAN);
  const std::optional<DeviceUuid> uuid = vulkan_device_uuid(*any->device);
  REQUIRE(uuid.has_value());

  auto same = open_hw_device(kVulkanCodec, HwApi::Vulkan, {.render_device = uuid});
  REQUIRE(same.has_value());
  CHECK(vulkan_device_uuid(*same->device) == uuid);

  DeviceUuid other = *uuid;
  other[0] = static_cast<std::uint8_t>(other[0] ^ 0xFFU);
  auto mismatch = open_hw_device(kVulkanCodec, HwApi::Vulkan, {.render_device = other});
  REQUIRE_FALSE(mismatch);
  CHECK(mismatch.error().code == RefusalCode::HwDecodeUnavailable);
  CHECK(mismatch.error().detail.find("UUID") != std::string::npos);
}

TEST_CASE("hw : un appareil partagé est repris tel quel, d'un autre type il est refusé",
          "[media][hw][vulkan]") {
  auto created = open_hw_device(kVulkanCodec, HwApi::Vulkan, DecoderOptions{});
  if (!created) SKIP("aucun appareil Vulkan : " << created.error().detail);

  const SharedDevice shared(HwApi::Vulkan, created->device.get());
  auto reused = open_hw_device(kVulkanCodec, HwApi::Vulkan, {.devices = &shared});
  REQUIRE(reused.has_value());
  CHECK(reused->device->data == created->device->data);
  CHECK_FALSE(reused->stream);

  const SharedDevice wrong(HwApi::Cuda, created->device.get());  // un Vulkan présenté comme CUDA
  auto refused = open_hw_device(kVulkanCodec, HwApi::Cuda, {.devices = &wrong});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().detail.find("autre type") != std::string::npos);
}

TEST_CASE("hw : noms des API", "[media][hw]") {
  CHECK(name(HwApi::Cuda) == "cuda");
  CHECK(name(HwApi::D3d11va) == "d3d11va");
  CHECK(name(HwApi::Software) == "logiciel");
}
