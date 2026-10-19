// Import d'une image, d'un tampon et d'un sémaphore timeline que Vulkan exporte, comme render
// exportera les slots du ring : CUDA écrit le RGBA16 du slot et Vulkan le relit, Vulkan remplit
// un tampon et CUDA le lit, chacun n'attendant l'autre que par le sémaphore, côté GPU.
#include "gpu/cuda/compute/interop.h"

#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/color.h"
#include "rig.h"
#include "vulkan_rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;

namespace {

ExternalHandle fd_handle(int fd) {
  return {.kind = ExternalHandle::Kind::OpaqueFd, .fd = fd};
}

}  // namespace

TEST_CASE("CUDA écrit une image Vulkan importée, que Vulkan relit après le sémaphore", "[cuda][interop]") {
  AJP_CUDA_RIG(rig);
  std::string reason;
  auto vk = VulkanRig::open(rig.device.info().uuid, reason);
  if (!vk) SKIP(reason);
  constexpr int kW = 333;
  constexpr int kH = 77;

  Exported image = vk->export_image(kW, kH);
  int semaphore_fd = -1;
  VkSemaphore semaphore = vk->export_timeline(semaphore_fd);
  vk->release_to_external(image.image);
  {
    auto imported = ImportedImage::import(fd_handle(image.fd), image.size, true, 0, kW, kH);
    if (!imported) close(image.fd);
    REQUIRE(imported);
    auto sem = ExternalSemaphore::import(fd_handle(semaphore_fd), true);
    if (!sem) close(semaphore_fd);
    REQUIRE(sem);

    const kernels::RgbImage rgb = random_rgb(kW, kH, -0.1F, 1.1F, 7U);
    const DeviceTensor t = upload_rgb(rig, rgb, Elem::F16);
    REQUIRE(write_rgba16(t.tensor, *imported, rig.s()));
    REQUIRE(sem->signal(1, rig.s()));
    // Vulkan attend 1 sur la file, relit, puis signale 2 ; CUDA attend 2 sur son flux.
    const std::vector<std::byte> bytes = vk->read_image(image.image, kW, kH, semaphore, 1, 2);
    REQUIRE(sem->wait(2, rig.s()));
    REQUIRE(rig.stream.synchronize());

    std::vector<std::uint16_t> got(bytes.size() / 2);
    std::memcpy(got.data(), bytes.data(), bytes.size());
    const auto expected = kernels::ref::to_rgba16(download_rgb(rig, t));
    REQUIRE(expected);
    CHECK(got == *expected);

    // Une taille différente de celle de l'image est refusée.
    const DeviceTensor small = alloc_tensor(rig, Elem::F16, kW - 1, kH);
    CHECK(write_rgba16(small.tensor, *imported, rig.s()).error().code == RefusalCode::ShapeInvalid);
  }
  vk->destroy(image);
  vk->destroy(semaphore);
}

TEST_CASE("CUDA lit un tampon Vulkan importé, après le sémaphore", "[cuda][interop]") {
  AJP_CUDA_RIG(rig);
  std::string reason;
  auto vk = VulkanRig::open(rig.device.info().uuid, reason);
  if (!vk) SKIP(reason);
  constexpr VkDeviceSize kBytes = VkDeviceSize{1} << 20;

  Exported buffer = vk->export_buffer(kBytes);
  int semaphore_fd = -1;
  VkSemaphore semaphore = vk->export_timeline(semaphore_fd);
  {
    auto memory = ExternalMemory::import(fd_handle(buffer.fd), buffer.size, true);
    if (!memory) close(buffer.fd);
    REQUIRE(memory);
    auto mapped = memory->map_buffer(0, kBytes);
    REQUIRE(mapped);
    CHECK(memory->map_buffer(0, kBytes).error().code == RefusalCode::InputMismatch);  // une seule projection
    auto sem = ExternalSemaphore::import(fd_handle(semaphore_fd), true);
    if (!sem) close(semaphore_fd);
    REQUIRE(sem);

    vk->fill_buffer(buffer.buffer, kBytes, 0x12345678U, semaphore, 5);
    REQUIRE(sem->wait(5, rig.s()));
    const std::vector<std::byte> bytes = rig.download(*mapped, kBytes);
    vk->finish();
    std::vector<std::uint32_t> words(kBytes / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    CHECK(std::ranges::all_of(words, [](std::uint32_t w) { return w == 0x12345678U; }));
  }
  vk->destroy(buffer);
  vk->destroy(semaphore);
}

TEST_CASE("l'import refuse une poignée ou une plage hors contrat", "[cuda][interop]") {
  AJP_CUDA_RIG(rig);
  CHECK(ExternalMemory::import(fd_handle(-1), 4096, true).error().code == RefusalCode::InputMismatch);
  CHECK(ExternalMemory::import({.kind = ExternalHandle::Kind::OpaqueWin32}, 4096, true).error().code ==
        RefusalCode::InputMismatch);
  CHECK(ExternalMemory::import(fd_handle(3), 0, true).error().code == RefusalCode::InputMismatch);
  CHECK(ExternalSemaphore::import(fd_handle(-1), true).error().code == RefusalCode::InputMismatch);
  CHECK(ImportedImage::import(fd_handle(3), 4096, true, 0, 0, 16).error().code == RefusalCode::ShapeInvalid);

  std::string reason;
  auto vk = VulkanRig::open(rig.device.info().uuid, reason);
  if (!vk) SKIP(reason);
  Exported buffer = vk->export_buffer(4096);
  {
    auto memory = ExternalMemory::import(fd_handle(buffer.fd), buffer.size, true);
    if (!memory) close(buffer.fd);
    REQUIRE(memory);
    CHECK(memory->map_buffer(buffer.size, 1).error().code == RefusalCode::InputMismatch);
    CHECK(memory->map_buffer(0, 0).error().code == RefusalCode::InputMismatch);
  }
  vk->destroy(buffer);
}
