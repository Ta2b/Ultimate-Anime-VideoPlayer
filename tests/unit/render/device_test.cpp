// Appareil de rendu et ring de sortie, sur le GPU de la machine (la RTX ici, lavapipe en CI).
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <libplacebo/vulkan.h>

#include "render/compositor.h"
#include "render/output_ring.h"
#include "support.h"
#include "vk_import.h"

namespace ajp::render {
namespace {

#define REQUIRE_DEVICE(device)                  \
  std::string skip_reason;                      \
  auto device = test::make_device(skip_reason); \
  if (!device) SKIP("pas d'appareil Vulkan : " << skip_reason)

}  // namespace

TEST_CASE("l'appareil sans écran remplit GpuDevice", "[gpu]") {
  REQUIRE_DEVICE(device);
  const GpuDevice& desc = device->description();
  CHECK(desc.api == GpuApi::Vulkan);
  CHECK_FALSE(desc.name.empty());
  CHECK_FALSE(desc.driver_version.empty());
  CHECK(desc.pci_vendor != 0);
  CHECK(std::ranges::any_of(desc.uuid, [](std::uint8_t b) { return b != 0; }));
  CHECK(std::ranges::equal(desc.uuid, device->gpu()->uuid));
  REQUIRE(desc.vulkan);
  const VulkanDevice& vk = *desc.vulkan;
  CHECK(vk.instance != nullptr);
  CHECK(vk.physical_device != nullptr);
  CHECK(vk.device == device->vulkan()->device);
  CHECK(vk.get_instance_proc_addr != nullptr);
  CHECK(vk.api_version >= VK_API_VERSION_1_2);
  CHECK(vk.enabled_features != nullptr);
  CHECK(device->surface() == 0);
  // Les familles retenues portent les files graphique et calcul, avec leurs drapeaux.
  const auto family = [&](std::uint32_t index) {
    return std::ranges::find(vk.families, index, &QueueFamily::index);
  };
  REQUIRE(family(vk.graphics.family) != vk.families.end());
  REQUIRE(family(vk.compute.family) != vk.families.end());
  CHECK((family(vk.graphics.family)->flags & queue_flags::kGraphics) != 0);
  CHECK((family(vk.compute.family)->flags & queue_flags::kCompute) != 0);
  if (vk.video_decode) {
    CHECK((family(vk.video_decode->family)->flags & queue_flags::kVideoDecode) != 0);
    CHECK(family(vk.video_decode->family)->video_caps != 0);
  }
  // Le verrou se prend et se rend sur chaque file créée.
  REQUIRE(vk.lock_queue != nullptr);
  REQUIRE(vk.unlock_queue != nullptr);
  for (const QueueFamily& f : vk.families) {
    for (std::uint32_t i = 0; i < f.count; ++i) {
      vk.lock_queue(vk.queue_context, f.index, i);
      vk.unlock_queue(vk.queue_context, f.index, i);
    }
  }
  const bool fp16 =
      std::ranges::find(vk.device_extensions, std::string(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) !=
          vk.device_extensions.end() ||
      vk.api_version >= VK_API_VERSION_1_2;
  CHECK(fp16);
}

TEST_CASE("un UUID inconnu est refusé", "[gpu]") {
  REQUIRE_DEVICE(device);
  DeviceUuid unknown{};
  unknown.fill(0xA5);
  const auto refused =
      RenderDevice::create(RenderDeviceOptions{.device_uuid = unknown, .allow_software = true});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::DeviceUnavailable);
}

TEST_CASE("le ring refuse une taille, un nombre de slots ou un format hors contrat", "[gpu]") {
  REQUIRE_DEVICE(device);
  CHECK(OutputRing::create(*device, {.slots = 0, .width = 16, .height = 16}).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(OutputRing::create(*device, {.slots = 17, .width = 16, .height = 16}).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(OutputRing::create(*device, {.slots = 2, .width = 0, .height = 16}).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(OutputRing::create(*device, {.slots = 2, .width = 16, .height = 16, .format = RingFormat::Yuv444p16})
            .error()
            .code == RefusalCode::FormatUnsupported);
}

TEST_CASE("le ring suit le protocole hold/release et refuse les messages hors séquence", "[gpu]") {
  REQUIRE_DEVICE(device);
  auto ring = OutputRing::create(*device, {.slots = 2, .width = 32, .height = 16, .generation = 7});
  REQUIRE(ring);
  REQUIRE(ring->slots().size() == 2);
  const RingSlotDesc& desc = ring->slots()[0];
  CHECK(desc.index == 0);
  CHECK(desc.generation == 7);
  CHECK(desc.width == 32);
  CHECK(desc.height == 16);
  CHECK(desc.image != 0);
  CHECK(desc.semaphore != 0);
  CHECK(desc.memory.kind == HandleKind::None);
  CHECK(ring->slots()[1].index == 1);

  const test::VulkanCalls vk(device->description());
  // Le sémaphore du slot part de 0 : la valeur que gpu attend au premier tour.
  CHECK(vk.counter(desc.semaphore) == 0);
  vk.signal(desc.semaphore, 1);  // l'écrivain a fini

  CHECK(ring->acquire(RingFrame{.slot = 0, .generation = 8, .ready_value = 1}).error().code ==
        RefusalCode::SequenceOrder);
  CHECK(ring->acquire(RingFrame{.slot = 2, .generation = 7, .ready_value = 1}).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(ring->acquire(RingFrame{.slot = 0, .generation = 7, .ready_value = 2}).error().code ==
        RefusalCode::FieldOutOfRange);
  REQUIRE(ring->acquire(RingFrame{.slot = 0, .generation = 7, .ready_value = 1}));
  CHECK(ring->acquire(RingFrame{.slot = 0, .generation = 7, .ready_value = 1}).error().code ==
        RefusalCode::SequenceOrder);
  CHECK(ring->release(1).error().code == RefusalCode::SequenceOrder);

  const auto freed = ring->release(0);
  REQUIRE(freed);
  CHECK(freed->slot == 0);
  CHECK(freed->semaphore_value == 2);
  CHECK(freed->generation == 7);
  CHECK(ring->release(0).error().code == RefusalCode::SequenceOrder);
  // Fin de lecture côté GPU : libplacebo signale la valeur rendue dans SlotFree.
  pl_gpu_finish(device->gpu());
  CHECK(vk.counter(desc.semaphore) == 2);
  // Le tour suivant exige une valeur impaire au-delà de 2.
  CHECK(ring->acquire(RingFrame{.slot = 0, .generation = 7, .ready_value = 1}).error().code ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("un slot exporté se réimporte et transmet l'image de l'écrivain sans copie", "[gpu]") {
  REQUIRE_DEVICE(device);
  auto ring = OutputRing::create(*device, {.slots = 1,
                                           .width = 16,
                                           .height = 8,
                                           .export_kind = HandleKind::OpaqueFd,
                                           .writer = SlotWriter::SameDevice});
  if (!ring && ring.error().code == RefusalCode::FeatureUnavailable)
    SKIP("export par fd absent : " << ring.error().detail);
  REQUIRE(ring);
  const RingSlotDesc& desc = ring->slots()[0];
  REQUIRE(desc.memory.kind == HandleKind::OpaqueFd);
  REQUIRE(desc.semaphore_handle.kind == HandleKind::OpaqueFd);
  CHECK(desc.memory_size > 0);

  // L'écrivain importe la mémoire et le sémaphore, comme gpu le ferait depuis un autre module.
  const test::VulkanCalls vk(device->description());
  auto imported = test::import_slot(vk, *device, *ring, 0);
  REQUIRE(imported);
  const pl_gpu gpu = device->gpu();
  const auto [w, h] = std::pair{desc.width, desc.height};
  auto target = OffscreenTarget::create(*device, {.width = w, .height = h, .display = test::sdr_display()});
  REQUIRE(target);
  auto compositor = Compositor::create(*device);
  REQUIRE(compositor);
  const SlotColor srgb{.primaries = Primaries::Bt709, .transfer = Transfer::Srgb};

  const std::array<std::array<float, 4>, 2> colors{{{0.25F, 0.5F, 0.75F, 1.0F}, {0.9F, 0.1F, 0.4F, 1.0F}}};
  std::uint64_t release_value = 0;
  for (const auto& color : colors) {
    // Écrivain : attend la libération, écrit, signale une valeur impaire.
    (*imported)->write(gpu, release_value, color, release_value + 1);
    auto slot = ring->acquire(RingFrame{.slot = 0, .generation = 0, .ready_value = release_value + 1});
    REQUIRE(slot);
    REQUIRE(compositor->compose(*slot, srgb, target->frame()));
    const auto freed = ring->release(0);
    REQUIRE(freed);
    release_value = freed->semaphore_value;
    const auto pixels = test::read_back(*target);
    REQUIRE(pixels.size() == static_cast<std::size_t>(w * h * 4));
    for (int c = 0; c < 3; ++c) {
      CHECK(std::abs(test::channel(pixels, w, 3, 5, c) - color[static_cast<std::size_t>(c)]) < 2e-3F);
    }
  }
  CHECK(release_value == 4);
  pl_gpu_finish(gpu);
}

}  // namespace ajp::render
