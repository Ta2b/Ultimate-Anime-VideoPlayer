// Les options du décodeur par backend : TensorRT lit les surfaces NVDEC en place, sans copie de
// sécurité ni copie en mémoire hôte ; les autres backends gardent la copie de FFmpeg.
#include <catch2/catch_test_macros.hpp>

#include "app/play/chain_setup.h"
#include "core/gpu_device.h"
#include "media/hw_api.h"

namespace ajp::app::play {
namespace {

TEST_CASE("decoder_options : surfaces directes sous TensorRT seulement", "[app][decoder]") {
  GpuDevice render;
  render.uuid[0] = 7;
  const media::DecoderOptions trt = decoder_options("tensorrt", render, nullptr);
  CHECK(trt.attempts == std::vector<media::HwApi>{media::HwApi::Cuda});
  CHECK(trt.render_device == render.uuid);
  CHECK_FALSE(trt.download);
  CHECK(trt.direct_surfaces == kHeldDecodedFrames);
  CHECK(trt.direct_surfaces <= media::kMaxExtraHwFrames);

  const media::DecoderOptions vulkan = decoder_options("vulkan", render, nullptr);
  CHECK(vulkan.download);
  CHECK(vulkan.direct_surfaces == 0);
  CHECK(vulkan.attempts.back() == media::HwApi::Software);
}

}  // namespace
}  // namespace ajp::app::play
