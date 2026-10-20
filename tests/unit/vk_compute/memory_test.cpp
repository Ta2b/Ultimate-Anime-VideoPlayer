// Sous-allocateur borné, tampons, envoi et lecture hôte, image RGBA16.
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/compute/destroyed_log.h"
#include "gpu/vulkan/compute/image.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;

namespace {

constexpr VkDeviceSize kKiB = 1024;
constexpr VkDeviceSize kMiB = 1024 * kKiB;

std::vector<std::byte> pattern(std::size_t count, unsigned seed) {
  std::vector<std::byte> bytes(count);
  for (std::size_t i = 0; i < count; ++i) bytes[i] = static_cast<std::byte>((i * 37U + seed) & 0xFFU);
  return bytes;
}

}  // namespace

TEST_CASE("mémoire : portions d'un même bloc, recousues à la libération", "[vk_compute][memory]") {
  Rig rig = make_rig();
  auto memory =
      require_value(MemoryAllocator::create(*rig.context, {.block_size = kMiB, .budget = 4 * kMiB}));
  std::vector<std::optional<Buffer>> buffers;
  for (int i = 0; i < 4; ++i) {
    buffers.emplace_back(require_value(Buffer::create(*rig.context, *memory, 64 * kKiB, MemoryUse::Device)));
  }
  CHECK(memory->block_count() == 1);
  CHECK(memory->reserved() == kMiB);
  CHECK(memory->used() >= 4 * 64 * kKiB);
  // Deux voisines libérées laissent une place de 128 Kio dans le même bloc.
  buffers[1].reset();
  buffers[2].reset();
  buffers.emplace_back(require_value(Buffer::create(*rig.context, *memory, 128 * kKiB, MemoryUse::Device)));
  CHECK(memory->block_count() == 1);
  buffers.clear();
  CHECK(memory->block_count() == 0);
  CHECK(memory->reserved() == 0);
  CHECK(memory->used() == 0);
}

TEST_CASE("mémoire : budget atteint refusé par son nom", "[vk_compute][memory]") {
  Rig rig = make_rig();
  auto memory =
      require_value(MemoryAllocator::create(*rig.context, {.block_size = kMiB, .budget = 2 * kMiB}));
  Buffer a = require_value(Buffer::create(*rig.context, *memory, 768 * kKiB, MemoryUse::Device));
  Buffer b = require_value(Buffer::create(*rig.context, *memory, 768 * kKiB, MemoryUse::Device));
  CHECK(memory->block_count() == 2);
  const auto third = Buffer::create(*rig.context, *memory, 768 * kKiB, MemoryUse::Device);
  require_refusal(third, RefusalCode::GpuMemoryExhausted);
  CHECK(third.error().detail.find("budget") != std::string::npos);
  require_refusal(Buffer::create(*rig.context, *memory, 3 * kMiB, MemoryUse::Device),
                  RefusalCode::GpuMemoryExhausted);
  // Rien n'a été réservé par les refus.
  CHECK(memory->reserved() == 2 * kMiB);
}

TEST_CASE("mémoire : limites et tailles invalides refusées", "[vk_compute][memory]") {
  Rig rig = make_rig();
  require_refusal(MemoryAllocator::create(*rig.context, {.block_size = 0, .budget = kMiB}),
                  RefusalCode::FieldOutOfRange);
  require_refusal(MemoryAllocator::create(*rig.context, {.block_size = 2 * kMiB, .budget = kMiB}),
                  RefusalCode::FieldOutOfRange);
  require_refusal(Buffer::create(*rig.context, *rig.memory, 0, MemoryUse::Device),
                  RefusalCode::FieldOutOfRange);
  Buffer buffer = make_buffer(rig, 256);
  require_refusal(buffer.view(200, 100), RefusalCode::InputMismatch);
  require_refusal(buffer.view(0, 0), RefusalCode::InputMismatch);
  const std::vector<std::byte> bytes(64);
  require_refusal(Done(upload(*rig.context, *rig.memory, buffer, 224, bytes)), RefusalCode::InputMismatch);
}

TEST_CASE("tampons : envoi et lecture hôte pour chaque usage", "[vk_compute][memory]") {
  Rig rig = make_rig();
  for (const MemoryUse use : {MemoryUse::Device, MemoryUse::Upload, MemoryUse::Readback}) {
    INFO("usage " << static_cast<int>(use));
    Buffer buffer = make_buffer(rig, 1000, use);
    CHECK((buffer.mapped() == nullptr) == (use == MemoryUse::Device));
    const std::vector<std::byte> bytes = pattern(900, static_cast<unsigned>(use));
    const std::uint64_t before = rig.context->host_waits();
    write_bytes(rig, buffer, bytes, 40);
    CHECK(read_bytes(rig, buffer, bytes.size(), 40) == bytes);
    // Seul l'usage Device passe par un transit et une soumission attendue, dans chaque sens.
    CHECK(rig.context->host_waits() - before == (use == MemoryUse::Device ? 2U : 0U));
  }
}

TEST_CASE("image RGBA16 : stockage par la vue UINT, relue exactement", "[vk_compute][memory]") {
  Rig rig = make_rig();
  require_refusal(StorageImage::create(*rig.context, *rig.memory, 0, 4), RefusalCode::InputMismatch);
  StorageImage image = require_value(StorageImage::create(*rig.context, *rig.memory, 7, 5));
  CHECK(image.storage_view() != VK_NULL_HANDLE);
  CHECK(image.sampled_view() != VK_NULL_HANDLE);
  const kernels::RgbImage rgb = synthetic_rgb(7, 5, 11);
  TensorBuffer tensor = upload_rgb(rig, rgb, Precision::F32);
  record_and_run(rig, [&](Recorder& r) { return rig.kernels().output_rgba16(r, tensor.view, image); });
  const std::vector<std::uint16_t> codes = require_value(read_rgba16(*rig.context, *rig.memory, image));
  REQUIRE(codes.size() == 7U * 5U * 4U);
  CHECK(codes[3] == kernels::kOpaqueAlpha16);
}

TEST_CASE("journal des destructions : lu depuis une marque, perdu au-delà de sa capacité",
          "[vk_compute][memory]") {
  DestroyedLog log;
  const std::uint64_t start = log.mark();
  log.note(11);
  log.note(22);
  std::vector<std::uint64_t> gone;
  const DestroyedLog::Read read = log.since(start, gone);
  CHECK(read.complete);
  CHECK(read.mark == start + 2);
  CHECK(gone == std::vector<std::uint64_t>{11, 22});
  gone.clear();
  CHECK(log.since(read.mark, gone).complete);
  CHECK(gone.empty());
  for (std::size_t i = 0; i <= DestroyedLog::kCapacity; ++i) log.note(i);
  CHECK_FALSE(log.since(read.mark, gone).complete);
}

TEST_CASE("tampons : chaque destruction entre au journal du contexte", "[vk_compute][memory]") {
  Rig rig = make_rig();
  auto memory =
      require_value(MemoryAllocator::create(*rig.context, {.block_size = kMiB, .budget = 4 * kMiB}));
  std::optional<Buffer> buffer =
      require_value(Buffer::create(*rig.context, *memory, 64 * kKiB, MemoryUse::Device));
  const std::uint64_t handle = handle_word(buffer->handle());
  const std::uint64_t before = rig.context->destroyed().mark();
  buffer.reset();
  std::vector<std::uint64_t> gone;
  CHECK(rig.context->destroyed().since(before, gone).complete);
  CHECK(gone == std::vector<std::uint64_t>{handle});
}
