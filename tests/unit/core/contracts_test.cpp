// Contrats de la vague 2 entre modules : appareil de rendu, ring de sortie, écran.
// Leurs valeurs par défaut doivent être sûres : rien d'inventé, rien d'activé en silence.
#include <cstdint>
#include <stop_token>
#include <type_traits>
#include <variant>

#include <catch2/catch_test_macros.hpp>

#include "core/engines.h"
#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/plan.h"
#include "core/resolve.h"
#include "core/ring.h"

using namespace ajp;

static_assert(std::is_same_v<core::RingFormat, RingFormat>, "un seul RingFormat, celui du ring");
static_assert(std::is_same_v<core::GpuVendor, GpuVendor>, "un seul GpuVendor, celui de l'appareil");
static_assert(sizeof(DeviceUuid) == 16 && sizeof(DeviceLuid) == 8);

TEST_CASE("GpuDevice : le fabricant se déduit de l'identifiant PCI", "[contracts]") {
  CHECK(vendor_from_pci(0x10DE) == GpuVendor::Nvidia);
  CHECK(vendor_from_pci(0x1002) == GpuVendor::Amd);
  CHECK(vendor_from_pci(0x8086) == GpuVendor::Intel);
  CHECK(vendor_from_pci(0x106B) == GpuVendor::Apple);
  CHECK(vendor_from_pci(0x1AF4) == GpuVendor::Other);  // virtio-gpu
  CHECK(vendor_from_pci(0) == GpuVendor::Other);

  const GpuDevice none;
  CHECK(none.vendor() == GpuVendor::Other);
  CHECK_FALSE(none.vulkan);
  CHECK_FALSE(none.luid);
  CHECK(none.uuid == DeviceUuid{});

  GpuDevice rtx{.api = GpuApi::Vulkan, .pci_vendor = pci::kNvidia, .name = "NVIDIA GeForce RTX 4070 Ti"};
  rtx.vulkan.emplace();
  CHECK(rtx.vendor() == GpuVendor::Nvidia);
  // Sans verrou ni file de décodage, rien n'est partagé par défaut.
  CHECK(rtx.vulkan->lock_queue == nullptr);
  CHECK(rtx.vulkan->unlock_queue == nullptr);
  CHECK_FALSE(rtx.vulkan->video_decode);
  CHECK(queue_flags::kVideoDecode == 0x20U);
}

TEST_CASE("Ring : un slot et une image par défaut n'ont rien d'exporté ni d'horodaté", "[contracts]") {
  const RingSlotDesc slot;
  CHECK(slot.format == RingFormat::Rgba16Unorm);
  CHECK(slot.memory.kind == HandleKind::None);
  CHECK(slot.semaphore_handle.kind == HandleKind::None);

  const RingFrame frame;
  CHECK(frame.pts_us == kNoPts);
  CHECK(frame.t_num == 0);
  CHECK(frame.t_den == 1);
  CHECK_FALSE(frame.repeated);
  CHECK_FALSE(frame.scene_cut);
  CHECK(frame.ready_value == 0);
}

TEST_CASE("Ring : un SlotFree porte la génération du ring et passe par la boîte gpu", "[contracts]") {
  Inbox<SlotFree, Flush> inbox({2, 1});
  CHECK(inbox.try_push(SlotFree{.slot = 3, .semaphore_value = 8, .generation = 2}));
  std::stop_source never;
  auto freed = inbox.wait(never.get_token());
  REQUIRE(freed);
  const auto* slot = std::get_if<SlotFree>(&*freed);
  REQUIRE(slot != nullptr);
  CHECK(slot->slot == 3);
  CHECK(slot->semaphore_value == 8);  // ready_value 7 du protocole, plus un
  CHECK(slot->generation == 2);
}

TEST_CASE("DisplayInfo : valeurs par défaut sûres et marquées comme telles", "[contracts]") {
  const core::DisplayInfo display;
  CHECK(display.refresh_hz == 0.0);  // inconnu : resolve() ne cale rien
  CHECK_FALSE(display.hdr);
  CHECK(display.space == core::OutputSpace::Srgb);
  CHECK(display.sdr_white_nits == core::display_defaults::kSdrWhiteNits);
  CHECK(display.min_nits < display.peak_nits);
  CHECK(display.provenance == core::DisplayProvenance{});
  CHECK(display.provenance.peak == core::Provenance::Default);
  CHECK(core::check_display(display));

  core::DisplayInfo measured = display;
  measured.peak_nits = 1015.0;
  measured.provenance.peak = core::Provenance::System;
  CHECK(measured.provenance != display.provenance);
  CHECK(core::check_display(measured));
}
