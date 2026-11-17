// Le warp écrit directement dans le slot du ring sous Vulkan (KernelOps::warp_blend_ring et
// warp_blend_poses_ring, BackendCaps::ring_targets) : les codes RGBA16 que write_ring écrirait du tenseur de
// warp_blend, au bit près, sans le tenseur ni sa relecture ; le slot attendu puis signalé comme par
// write_ring.
#include <array>
#include <cstdint>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/image.h"
#include "gpu/vulkan/compute/sync.h"
#include "kernels/ref/warp.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

constexpr int kWidth = 24;
constexpr int kHeight = 16;

std::vector<TensorType> types_of(const vk::VulkanBackend& backend) {
  std::vector<TensorType> types{TensorType::F32};
  if (backend.profile().target.f16_storage) types.push_back(TensorType::F16);
  return types;
}

struct Moved {
  std::unique_ptr<TensorBuffer> flow;
  std::unique_ptr<TensorBuffer> mask;
  k::Window area{4, 2, kWidth, kHeight};
  Motion gpu(k::RifeWarp warp) const {
    return {.flow = flow->tensor(), .mask = mask->tensor(), .area = area, .warp = warp};
  }
};

Moved moved(vk::VulkanBackend& backend, TensorType type, std::uint32_t seed, float scale) {
  Moved m;
  std::array<k::PlaneF, 4> flow;
  for (std::size_t c = 0; c < 4; ++c)
    flow[c] =
        stored(synthetic_rgb(32, 20, seed + static_cast<std::uint32_t>(c), -scale, scale), type).planes[0];
  const k::PlaneF mask = stored(synthetic_rgb(32, 20, seed + 4), type).planes[0];
  m.flow = rgb_tensor(backend, 32, 20, type, 4);
  m.mask = rgb_tensor(backend, 32, 20, type, 1);
  write_tensor(backend, m.flow->tensor(), planes_bytes(flow, type));
  write_tensor(backend, m.mask->tensor(), planes_bytes(std::span(&mask, 1), type));
  return m;
}

// Un slot du ring et son sémaphore sur l'appareil du backend, avec la valeur suivante du protocole.
struct Slot {
  vk::StorageImage image;
  vk::TimelineSemaphore semaphore;
  RingSlotDesc desc;
  std::uint64_t value = 0;  // dernière valeur signalée

  RingWrite next() {
    const RingWrite sync{.wait_value = value, .signal_value = value + 1};
    value += 2;  // render rend le slot à la valeur paire suivante
    return sync;
  }
  std::vector<std::uint16_t> read(vk::VulkanBackend& backend, const RingWrite& sync) {
    require_ok(backend.context().wait_on_host(semaphore.at(sync.signal_value)));
    auto codes = require_value(vk::read_rgba16(backend.context(), backend.memory(), image));
    require_ok(semaphore.signal(value));
    return codes;
  }
};

}  // namespace

TEST_CASE("KernelOps : le warp écrit le slot du ring au code de write_ring près",
          "[vk_backend][kernels][ring]") {
  TempDir dir("ring-warp");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  CHECK(backend->capabilities().ring_targets);
  auto image = vk::StorageImage::create(backend->context(), backend->memory(), kWidth, kHeight);
  if (!image && image.error().code == RefusalCode::FeatureUnavailable) SKIP(describe(image.error()));
  REQUIRE(image);
  auto semaphore = require_value(vk::TimelineSemaphore::create(backend->context(), 0));
  const auto bits = [](auto handle) { return vk::detail::to_bits(handle); };
  Slot slot{.image = std::move(*image),
            .semaphore = std::move(semaphore),
            .desc = {.index = 0, .generation = 1, .width = kWidth, .height = kHeight}};
  slot.desc.image = bits(slot.image.image());
  slot.desc.semaphore = bits(slot.semaphore.handle());
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    const k::RgbImage a = stored(synthetic_rgb(kWidth, kHeight, 61, -0.1F, 1.1F), type);
    const k::RgbImage b = stored(synthetic_rgb(kWidth, kHeight, 62, -0.1F, 1.1F), type);
    auto ta = rgb_tensor(*backend, kWidth, kHeight, type);
    auto tb = rgb_tensor(*backend, kWidth, kHeight, type);
    auto out = rgb_tensor(*backend, kWidth, kHeight, type);
    write_tensor(*backend, ta->tensor(), rgb_bytes(a, type));
    write_tensor(*backend, tb->tensor(), rgb_bytes(b, type));
    const Moved now = moved(*backend, type, 63, 3.0F);
    const Moved before = moved(*backend, type, 68, 1.0F);
    for (const k::RifeWarp warp : {k::RifeWarp::Bilinear, k::RifeWarp::CatmullRom}) {
      const Motion m = now.gpu(warp);
      const Motion previous = before.gpu(warp);
      for (const int poses : {0, 1, 2}) {  // sans poses, poses gardées, paire précédente tenue
        for (const float t : {0.25F, 0.5F, 0.75F}) {
          CAPTURE(static_cast<int>(warp), poses, t);
          const Motion* prev = poses == 1 ? &previous : nullptr;
          // La référence : le tenseur de warp_blend, puis write_ring.
          if (poses == 0) {
            require_ok(ops.warp_blend(ta->tensor(), tb->tensor(), m, t, out->tensor()));
          } else {
            require_ok(ops.warp_blend_poses(ta->tensor(), tb->tensor(), m, prev, t, out->tensor()));
          }
          const RingWrite first = slot.next();
          require_ok(ops.write_ring(out->tensor(), slot.desc, first));
          const std::vector<std::uint16_t> want = slot.read(*backend, first);
          // Le même warp, directement dans le slot.
          const RingWrite second = slot.next();
          const RingTarget target{.slot = &slot.desc, .sync = second};
          if (poses == 0) {
            require_ok(ops.warp_blend_ring(ta->tensor(), tb->tensor(), m, t, target));
          } else {
            require_ok(ops.warp_blend_poses_ring(ta->tensor(), tb->tensor(), m, prev, t, target));
          }
          CHECK(slot.read(*backend, second) == want);
        }
      }
    }
  }
}

TEST_CASE("KernelOps : refus nommés du warp dans le slot", "[vk_backend][kernels][ring]") {
  TempDir dir("ring-warp-refus");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  auto image = vk::StorageImage::create(backend->context(), backend->memory(), kWidth, kHeight);
  if (!image && image.error().code == RefusalCode::FeatureUnavailable) SKIP(describe(image.error()));
  REQUIRE(image);
  auto semaphore = require_value(vk::TimelineSemaphore::create(backend->context(), 0));
  const auto bits = [](auto handle) { return vk::detail::to_bits(handle); };
  RingSlotDesc slot{.index = 0,
                    .generation = 1,
                    .width = kWidth,
                    .height = kHeight,
                    .image = bits(image->image()),
                    .semaphore = bits(semaphore.handle())};
  auto ta = rgb_tensor(*backend, kWidth, kHeight, TensorType::F32);
  auto tb = rgb_tensor(*backend, kWidth, kHeight, TensorType::F32);
  const Moved now = moved(*backend, TensorType::F32, 71, 2.0F);
  const Motion m = now.gpu(k::RifeWarp::Bilinear);
  const auto ring = [&](const RingSlotDesc* desc, RingWrite sync) {
    return ops.warp_blend_ring(ta->tensor(), tb->tensor(), m, 0.5F, {.slot = desc, .sync = sync});
  };
  require_refusal(ring(nullptr, {.wait_value = 0, .signal_value = 1}), RefusalCode::InputMismatch);
  require_refusal(ring(&slot, {.wait_value = 2, .signal_value = 2}), RefusalCode::FieldOutOfRange);
  require_refusal(ops.warp_blend_ring(ta->tensor(), tb->tensor(), m, 1.0F,
                                      {.slot = &slot, .sync = {.wait_value = 0, .signal_value = 1}}),
                  RefusalCode::FieldOutOfRange);
  RingSlotDesc wide = slot;
  wide.width = kWidth + 2;
  require_refusal(ring(&wide, {.wait_value = 0, .signal_value = 1}), RefusalCode::ShapeInvalid);
  RingSlotDesc yuv = slot;
  yuv.format = RingFormat::Yuv444p16;
  require_refusal(ring(&yuv, {.wait_value = 0, .signal_value = 1}), RefusalCode::FormatUnsupported);
  // Un mouvement précédent d'une autre fenêtre est refusé, comme par warp_blend_poses.
  Motion shifted = m;
  shifted.area.x = 0;
  require_refusal(ops.warp_blend_poses_ring(ta->tensor(), tb->tensor(), m, &shifted, 0.25F,
                                            {.slot = &slot, .sync = {.wait_value = 0, .signal_value = 1}}),
                  RefusalCode::ShapeInvalid);
}
